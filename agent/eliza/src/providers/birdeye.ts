/**
 * Birdeye API Provider for Solana DEX Data
 * 
 * Fetches market data from Birdeye API:
 * - Token prices
 * - OHLCV historical data
 * - Pool/pair information
 * - Volume and liquidity
 */

export interface OHLCVData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface TokenPrice {
  address: string;
  priceUsd: number;
  updateTime: number;
}

export interface PoolOverview {
  address: string;
  name: string;
  liquidityUsd: number;
  volume24h: number;
  price: number;
  priceChange24h: number;
}

// Token addresses
export const TOKENS = {
  SOL: 'So11111111111111111111111111111111111111112',
  USDC: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
  USDT: 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',
  RAY: '4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R',
  ORCA: 'orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE',
  JUP: 'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN',
  BONK: 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
} as const;

import { logger } from '../services/logger.js';

// Interval mapping
const INTERVAL_MAP: Record<string, string> = {
  '1m': '1m',
  '5m': '5m',
  '15m': '15m',
  '30m': '30m',
  '1h': '1H',
  '4h': '4H',
  '1d': '1D',
  '1w': '1W',
};

export class BirdeyeProvider {
  private baseUrl = 'https://public-api.birdeye.so';
  private apiKey: string;
  private headers: Record<string, string>;

  // Rate limiting: 60 rpm = 1 request per second
  private static lastRequestTime = 0;
  private static readonly MIN_REQUEST_INTERVAL = 1100; // 1.1 seconds between requests (safe margin)

  constructor(apiKey: string) {
    this.apiKey = apiKey;
    this.headers = {
      'X-API-KEY': apiKey,
      'x-chain': 'solana',
      'Content-Type': 'application/json',
    };
  }

  /**
   * Wait to respect rate limit (60 rpm)
   */
  private async waitForRateLimit(): Promise<void> {
    const now = Date.now();
    const timeSinceLastRequest = now - BirdeyeProvider.lastRequestTime;

    if (timeSinceLastRequest < BirdeyeProvider.MIN_REQUEST_INTERVAL) {
      const waitTime = BirdeyeProvider.MIN_REQUEST_INTERVAL - timeSinceLastRequest;
      await new Promise(resolve => setTimeout(resolve, waitTime));
    }

    BirdeyeProvider.lastRequestTime = Date.now();
  }

  private async request<T>(endpoint: string, params?: Record<string, string | number>, retries = 3): Promise<T> {
    const url = new URL(`${this.baseUrl}${endpoint}`);
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        url.searchParams.append(key, String(value));
      });
    }

    let lastError: Error | null = null;

    for (let attempt = 0; attempt < retries; attempt++) {
      try {
        // Respect rate limit before each request
        await this.waitForRateLimit();

        // Add timeout (30 seconds)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000);

        try {
          const response = await fetch(url.toString(), {
            headers: this.headers,
            signal: controller.signal,
          });

          clearTimeout(timeoutId);

          // Handle rate limiting with exponential backoff
          if (response.status === 429) {
            const waitTime = Math.pow(2, attempt + 1) * 2000; // 4s, 8s, 16s for rate limit
            logger.warn(`⚠️ Birdeye rate limited (429). Waiting ${waitTime}ms before retry ${attempt + 1}/${retries}`);
            await new Promise(resolve => setTimeout(resolve, waitTime));
            continue;
          }

          if (!response.ok) {
            throw new Error(`Birdeye API error: ${response.status} ${response.statusText}`);
          }

          const data = await response.json() as { success: boolean; data: T };
          if (!data.success) {
            throw new Error(`Birdeye API returned error: ${JSON.stringify(data)}`);
          }

          return data.data;
        } catch (fetchError: any) {
          clearTimeout(timeoutId);

          // Retry on timeout
          if ((fetchError.name === 'AbortError' || fetchError.cause?.code === 'UND_ERR_CONNECT_TIMEOUT') && attempt < retries - 1) {
            const waitTime = Math.pow(2, attempt) * 2000;
            logger.warn(`⚠️ Birdeye timeout (30s). Retrying in ${waitTime}ms (attempt ${attempt + 1}/${retries})`);
            await new Promise(resolve => setTimeout(resolve, waitTime));
            continue;
          }

          throw fetchError;
        }
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        if (attempt < retries - 1) {
          const waitTime = Math.pow(2, attempt) * 1000;
          await new Promise(resolve => setTimeout(resolve, waitTime));
        }
      }
    }

    throw lastError || new Error('Birdeye API request failed after retries');
  }

  async getTokenPrice(tokenAddress: string): Promise<TokenPrice> {
    const data = await this.request<{ value: number; updateUnixTime: number }>('/defi/price', {
      address: tokenAddress,
    });

    return {
      address: tokenAddress,
      priceUsd: data.value,
      updateTime: data.updateUnixTime,
    };
  }

  async getMultiTokenPrices(tokens: string[]): Promise<Record<string, TokenPrice>> {
    const results: Record<string, TokenPrice> = {};
    
    // Fetch in parallel
    const promises = tokens.map(async (token) => {
      try {
        const price = await this.getTokenPrice(token);
        results[token] = price;
      } catch (error) {
        logger.error(`Failed to fetch price for ${token}`, { error: String(error) });
      }
    });

    await Promise.all(promises);
    return results;
  }

  async getOHLCV(
    tokenAddress: string,
    interval: string = '1h',
    timeFrom?: number,
    timeTo?: number
  ): Promise<OHLCVData[]> {
    const now = Math.floor(Date.now() / 1000);
    const from = timeFrom ?? now - 86400; // Default: last 24h
    const to = timeTo ?? now;

    const data = await this.request<{ items: Array<{
      unixTime: number;
      o: number;
      h: number;
      l: number;
      c: number;
      v: number;
    }> }>('/defi/ohlcv', {
      address: tokenAddress,
      type: INTERVAL_MAP[interval] ?? '1H',
      time_from: from,
      time_to: to,
    });

    return data.items.map((item) => ({
      timestamp: item.unixTime,
      open: item.o,
      high: item.h,
      low: item.l,
      close: item.c,
      volume: item.v,
    }));
  }

  async getPoolOverview(poolAddress: string): Promise<PoolOverview> {
    logger.info(`🔍 Fetching pool overview for: ${poolAddress}`);

    try {
      const data = await this.request<{
        address: string;
        name: string;
        liquidity: number;
        v24hUSD: number;
        price: number;
        v24hChangePercent: number;
      }>('/defi/v3/pair/overview/single', { address: poolAddress });

      logger.info(`📊 Pool data received: ${JSON.stringify(data, null, 2)}`);

      return {
        address: data.address || poolAddress,
        name: data.name || 'Unknown',
        liquidityUsd: data.liquidity || 0,
        volume24h: data.v24hUSD || 0,
        price: data.price || 0,
        priceChange24h: data.v24hChangePercent || 0,
      };
    } catch (error) {
      logger.error('❌ Pool overview error', { error: String(error) });
      throw error;
    }
  }
}

