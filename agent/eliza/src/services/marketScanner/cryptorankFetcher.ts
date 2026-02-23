/**
 * CryptoRank Fetcher — Market data, sentiment, and global metrics from CryptoRank V2 API
 *
 * Provides:
 * - Token prices with multi-timeframe percent changes (24h, 7d, 30d, 3m, 6m)
 * - Market cap, volume, FDV, ATH/ATL data
 * - Global market metrics (BTC dominance, fear/greed, altcoin index)
 *
 * Auth: X-API-Key header
 * Rate limits: handled via resilience queue
 */

import type { TokenPrice } from './types.js';
import { resilientFetchJson, getQueue } from '../resilience.js';
import { logger } from '../logger.js';

const CRYPTORANK_API = 'https://api.cryptorank.io/v2';

function getApiKey(): string {
  return process.env.CRYPTORANK_API_KEY || '';
}

// CryptoRank numeric IDs for our core tokens
const CRYPTORANK_IDS: Record<string, number> = {
  BTC: 1,
  ETH: 3,
  SOL: 5663,
  BONK: 179862,
  JUP: 186828,
  WIF: 186547,
  RAY: 12688,
  USDC: 5176,
  USDT: 16,
};

// Register a rate-limited queue for CryptoRank (2 concurrent, conservative)
const getCryptoRankQueue = () => getQueue('cryptorank', { concurrency: 2 });

export interface CryptoRankPrice extends TokenPrice {
  marketCap?: number;
  fullyDilutedValuation?: number;
  high24h?: number;
  low24h?: number;
  rank?: number;
  percentChange?: {
    h24?: number;
    d7?: number;
    d30?: number;
    m3?: number;
    m6?: number;
  };
  ath?: { date: number; value: number; percentChange: number };
  atl?: { date: number; value: number; percentChange: number };
  circulatingSupply?: number;
}

interface CryptoRankCurrency {
  id: number;
  key: string;
  symbol: string;
  name: string;
  rank: number;
  price: string;
  high24h: string;
  low24h: string;
  volume24h: string;
  marketCap: string;
  fullyDilutedValuation: string;
  circulatingSupply: string;
  percentChange?: {
    h24: string;
    d7: string;
    d30: string;
    m3: string;
    m6: string;
  };
  ath?: { date: number; value: string; percentChange: string };
  atl?: { date: number; value: string; percentChange: string };
}

interface CryptoRankCurrencyResponse {
  data: CryptoRankCurrency;
  status: { usedCredits: number };
}

interface CryptoRankListResponse {
  data: CryptoRankCurrency[];
  status: { usedCredits: number };
}

export interface CryptoRankGlobalData {
  totalMarketCap: number;
  totalMarketCapChange: number;
  totalVolume24h: number;
  totalVolume24hChange: number;
  btcDominance: number;
  btcDominanceChange: number;
  ethDominance: number;
  ethDominanceChange: number;
  fearGreed: number;
  fearGreedChange: number;
  altcoinIndex: number;
  altcoinIndexChange: number;
  activeCurrencies: number;
}

function makeHeaders(): Record<string, string> {
  const key = getApiKey();
  if (!key) return {};
  return { 'X-API-Key': key };
}

/**
 * Fetch prices for specific tokens by their CryptoRank IDs.
 * Uses individual /currencies/:id calls to get percentChange data
 * (the list endpoint doesn't include percentChange).
 */
export async function fetchCryptoRankPrices(symbols: string[]): Promise<CryptoRankPrice[]> {
  const apiKey = getApiKey();
  if (!apiKey) {
    logger.debug('[CryptoRank] No API key configured, skipping');
    return [];
  }

  const prices: CryptoRankPrice[] = [];
  const ids = symbols
    .map(s => ({ symbol: s, id: CRYPTORANK_IDS[s] }))
    .filter(entry => entry.id !== undefined);

  if (ids.length === 0) return prices;

  const results = await Promise.allSettled(
    ids.map(({ symbol, id }) =>
      resilientFetchJson<CryptoRankCurrencyResponse>(
        `${CRYPTORANK_API}/currencies/${id}`,
        { headers: makeHeaders() },
        {
          queue: getCryptoRankQueue(),
          retries: 2,
          fetchTimeout: 10000,
          label: `cryptorank/${symbol}`,
        },
      ).then(resp => ({ symbol, data: resp.data })),
    ),
  );

  for (const result of results) {
    if (result.status !== 'fulfilled') continue;
    const { symbol, data } = result.value;

    prices.push({
      symbol,
      price: parseFloat(data.price),
      change24h: data.percentChange ? parseFloat(data.percentChange.h24) : undefined,
      volume24h: parseFloat(data.volume24h),
      marketCap: parseFloat(data.marketCap),
      fullyDilutedValuation: parseFloat(data.fullyDilutedValuation),
      high24h: parseFloat(data.high24h),
      low24h: parseFloat(data.low24h),
      rank: data.rank,
      circulatingSupply: parseFloat(data.circulatingSupply),
      percentChange: data.percentChange
        ? {
            h24: parseFloat(data.percentChange.h24),
            d7: parseFloat(data.percentChange.d7),
            d30: parseFloat(data.percentChange.d30),
            m3: parseFloat(data.percentChange.m3),
            m6: parseFloat(data.percentChange.m6),
          }
        : undefined,
      ath: data.ath
        ? { date: data.ath.date, value: parseFloat(data.ath.value), percentChange: parseFloat(data.ath.percentChange) }
        : undefined,
      atl: data.atl
        ? { date: data.atl.date, value: parseFloat(data.atl.value), percentChange: parseFloat(data.atl.percentChange) }
        : undefined,
      source: 'cryptorank',
      timestamp: Date.now(),
    });
  }

  logger.info(`[CryptoRank] Fetched ${prices.length}/${ids.length} token prices`);
  return prices;
}

/**
 * Fetch bulk prices from the /currencies list endpoint.
 * Faster (single call) but doesn't include percentChange detail per token.
 */
export async function fetchCryptoRankBulkPrices(symbols: string[]): Promise<CryptoRankPrice[]> {
  const apiKey = getApiKey();
  if (!apiKey) return [];

  try {
    const resp = await resilientFetchJson<CryptoRankListResponse>(
      `${CRYPTORANK_API}/currencies?limit=100`,
      { headers: makeHeaders() },
      {
        queue: getCryptoRankQueue(),
        retries: 2,
        fetchTimeout: 15000,
        label: 'cryptorank/currencies',
      },
    );

    const symbolSet = new Set(symbols.map(s => s.toUpperCase()));
    return resp.data
      .filter(c => symbolSet.has(c.symbol))
      .map(c => ({
        symbol: c.symbol,
        price: parseFloat(c.price),
        volume24h: parseFloat(c.volume24h),
        marketCap: parseFloat(c.marketCap),
        fullyDilutedValuation: parseFloat(c.fullyDilutedValuation),
        high24h: parseFloat(c.high24h),
        low24h: parseFloat(c.low24h),
        rank: c.rank,
        source: 'cryptorank' as const,
        timestamp: Date.now(),
      }));
  } catch (e) {
    logger.error('[CryptoRank] Bulk fetch error:', { error: String(e) });
    return [];
  }
}

/**
 * Fetch global market data — BTC dominance, total market cap, fear/greed, altcoin index
 */
export async function fetchCryptoRankGlobal(): Promise<CryptoRankGlobalData | null> {
  const apiKey = getApiKey();
  if (!apiKey) return null;

  try {
    const resp = await resilientFetchJson<{ data: Record<string, any>; status: { usedCredits: number } }>(
      `${CRYPTORANK_API}/global`,
      { headers: makeHeaders() },
      {
        queue: getCryptoRankQueue(),
        retries: 2,
        fetchTimeout: 10000,
        label: 'cryptorank/global',
      },
    );

    const d = resp.data;
    return {
      totalMarketCap: parseFloat(d.totalMarketCap),
      totalMarketCapChange: d.totalMarketCapChange,
      totalVolume24h: parseFloat(d.totalVolume24h),
      totalVolume24hChange: d.totalVolume24hChange,
      btcDominance: d.btcDominance,
      btcDominanceChange: d.btcDominanceChange,
      ethDominance: d.ethDominance,
      ethDominanceChange: d.ethDominanceChange,
      fearGreed: d.fearGreed,
      fearGreedChange: d.fearGreedChange,
      altcoinIndex: d.altcoinIndex,
      altcoinIndexChange: d.altcoinIndexChange,
      activeCurrencies: d.activeCurrencies,
    };
  } catch (e) {
    logger.error('[CryptoRank] Global data fetch error:', { error: String(e) });
    return null;
  }
}
