/**
 * CoinGecko Fetcher - Cross-exchange price verification
 */

import type { TokenPrice } from './types.js';
import { logger } from '../logger.js';

const COINGECKO_API = 'https://api.coingecko.com/api/v3';

// Get API key at runtime
function getApiKey(): string {
  return process.env.COINGECKO_API_KEY || '';
}

// CoinGecko ID mapping
const COINGECKO_IDS: Record<string, string> = {
  SOL: 'solana',
  BTC: 'bitcoin',
  ETH: 'ethereum',
  USDC: 'usd-coin',
  USDT: 'tether',
  WIF: 'dogwifcoin',
  BONK: 'bonk',
  JUP: 'jupiter-exchange-solana',
  RAY: 'raydium',
};

async function fetchWithTimeout(url: string, options: RequestInit = {}, timeout = 30000): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  try {
    const resp = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(id);
    return resp;
  } catch (e: any) {
    clearTimeout(id);
    // Log timeout errors for debugging
    if (e.name === 'AbortError' || e.cause?.code === 'UND_ERR_CONNECT_TIMEOUT') {
      logger.info(`[CoinGecko] Timeout (${timeout}ms) for ${url}`);
    }
    throw e;
  }
}

export interface CoinGeckoPrice extends TokenPrice {
  marketCap?: number;
  marketCapRank?: number;
  high24h?: number;
  low24h?: number;
  ath?: number;
  athChangePercentage?: number;
}

export async function fetchCoinGeckoPrices(symbols: string[]): Promise<CoinGeckoPrice[]> {
  const prices: CoinGeckoPrice[] = [];
  const ids = symbols.map(s => COINGECKO_IDS[s]).filter(Boolean);
  
  if (ids.length === 0) return prices;

  try {
    const apiKey = getApiKey();
    const headers: Record<string, string> = {};
    if (apiKey) {
      headers['x-cg-demo-api-key'] = apiKey;
    }

    const resp = await fetchWithTimeout(
      `${COINGECKO_API}/simple/price?ids=${ids.join(',')}&vs_currencies=usd&include_24hr_vol=true&include_24hr_change=true&include_market_cap=true`,
      { headers }
    );
    
    const data = await resp.json() as Record<string, {
      usd: number;
      usd_24h_vol: number;
      usd_24h_change: number;
      usd_market_cap: number;
    }>;

    for (const [id, info] of Object.entries(data)) {
      const symbol = Object.entries(COINGECKO_IDS).find(([, v]) => v === id)?.[0];
      if (!symbol) continue;

      prices.push({
        symbol,
        price: info.usd,
        change24h: info.usd_24h_change,
        volume24h: info.usd_24h_vol,
        marketCap: info.usd_market_cap,
        source: 'coingecko',
        timestamp: Date.now(),
      });
    }
  } catch (e) {
    logger.error('[CoinGecko] Fetch error:', { error: String(e) });
  }
  
  return prices;
}

// Get detailed market data
export async function fetchCoinGeckoMarkets(symbols: string[]): Promise<CoinGeckoPrice[]> {
  const prices: CoinGeckoPrice[] = [];
  const ids = symbols.map(s => COINGECKO_IDS[s]).filter(Boolean);
  
  if (ids.length === 0) return prices;

  try {
    const apiKey = getApiKey();
    const headers: Record<string, string> = {};
    if (apiKey) {
      headers['x-cg-demo-api-key'] = apiKey;
    }

    const resp = await fetchWithTimeout(
      `${COINGECKO_API}/coins/markets?vs_currency=usd&ids=${ids.join(',')}&order=market_cap_desc&sparkline=false`,
      { headers }
    );
    
    const data = await resp.json() as Array<{
      id: string;
      symbol: string;
      current_price: number;
      market_cap: number;
      market_cap_rank: number;
      total_volume: number;
      high_24h: number;
      low_24h: number;
      price_change_percentage_24h: number;
      ath: number;
      ath_change_percentage: number;
    }>;

    for (const coin of data) {
      const symbol = Object.entries(COINGECKO_IDS).find(([, v]) => v === coin.id)?.[0];
      if (!symbol) continue;

      prices.push({
        symbol,
        price: coin.current_price,
        change24h: coin.price_change_percentage_24h,
        volume24h: coin.total_volume,
        marketCap: coin.market_cap,
        marketCapRank: coin.market_cap_rank,
        high24h: coin.high_24h,
        low24h: coin.low_24h,
        ath: coin.ath,
        athChangePercentage: coin.ath_change_percentage,
        source: 'coingecko',
        timestamp: Date.now(),
      });
    }
  } catch (e) {
    logger.error('[CoinGecko] Markets fetch error:', { error: String(e) });
  }
  
  return prices;
}

