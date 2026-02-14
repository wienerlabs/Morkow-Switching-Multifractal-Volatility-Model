/**
 * DEX Price Fetcher - Raydium, Orca, Meteora, DexScreener + fallbacks
 */

import type { DEXPrice, LPPool, NewToken } from './types.js';
import { logger } from '../logger.js';

// API Endpoints
const BIRDEYE_API = 'https://public-api.birdeye.so';
const JUPITER_PRICE_API = 'https://api.jup.ag/price/v2';
const DEXSCREENER_API = 'https://api.dexscreener.com/latest/dex';
const RAYDIUM_API = 'https://api.raydium.io/v2/main/pairs';
const ORCA_API = 'https://api.orca.so/v1/whirlpool/list';
const METEORA_API = 'https://dlmm-api.meteora.ag/pair/all';

// Get API key at runtime (not at module load time)
function getBirdeyeApiKey(): string {
  return process.env.BIRDEYE_API_KEY || '';
}

// Well-known Solana token addresses
const TOKEN_ADDRESSES: Record<string, string> = {
  SOL: 'So11111111111111111111111111111111111111112',
  USDC: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
  USDT: 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',
  WIF: 'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm',
  BONK: 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
  JUP: 'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN',
  RAY: '4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R',
  BTC: '9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E', // Wrapped BTC
  ETH: '7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs', // Wrapped ETH
};

interface DexScreenerPair {
  baseToken: { symbol: string; address: string };
  quoteToken: { symbol: string };
  priceUsd: string;
  priceChange: { h24: number; h1: number };
  volume: { h24: number };
  liquidity: { usd: number };
  dexId: string;
  pairAddress: string;
  pairCreatedAt: number;
  txns: { h24: { buys: number; sells: number } };
}

// Fetch with timeout and retry
async function fetchWithRetry(url: string, options: RequestInit = {}, retries = 3, timeoutMs = 30000): Promise<Response> {
  for (let i = 0; i < retries; i++) {
    try {
      const controller = new AbortController();
      const id = setTimeout(() => controller.abort(), timeoutMs);
      const resp = await fetch(url, { ...options, signal: controller.signal });
      clearTimeout(id);
      if (resp.ok) return resp;
      if (resp.status >= 500 && i < retries - 1) {
        const waitTime = Math.pow(2, i) * 1000; // Exponential backoff: 1s, 2s, 4s
        logger.info(`[DEXFetcher] Server error (${resp.status}), retrying in ${waitTime}ms (attempt ${i + 1}/${retries})`);
        await new Promise(r => setTimeout(r, waitTime));
        continue;
      }
      return resp;
    } catch (e: any) {
      const isTimeout = e.name === 'AbortError' || e.cause?.code === 'UND_ERR_CONNECT_TIMEOUT';

      if (i < retries - 1) {
        const waitTime = Math.pow(2, i) * 2000; // Exponential backoff: 2s, 4s, 8s
        if (isTimeout) {
          logger.info(`[DEXFetcher] Timeout (${timeoutMs}ms), retrying in ${waitTime}ms (attempt ${i + 1}/${retries})`);
        } else {
          logger.info(`[DEXFetcher] Connection error, retrying in ${waitTime}ms (attempt ${i + 1}/${retries})`);
        }
        await new Promise(r => setTimeout(r, waitTime));
        continue;
      }

      // Last retry failed
      throw e;
    }
  }
  throw new Error('Max retries exceeded');
}

// ============= BIRDEYE (Primary) =============

export async function fetchBirdeyePricesReal(symbols: string[]): Promise<DEXPrice[]> {
  const prices: DEXPrice[] = [];
  const apiKey = getBirdeyeApiKey();
  if (!apiKey) {
    logger.warn('[Birdeye] No API key, skipping');
    return prices;
  }

  const addresses = symbols.map(s => TOKEN_ADDRESSES[s]).filter(Boolean);
  if (addresses.length === 0) return prices;

  try {
    const resp = await fetchWithRetry(
      `${BIRDEYE_API}/defi/multi_price?list_address=${addresses.join(',')}`,
      { headers: { 'X-API-KEY': apiKey, 'x-chain': 'solana' } }
    );
    const data = await resp.json() as { success: boolean; data: Record<string, { value: number; updateUnixTime: number }> };

    if (!data.success) return prices;

    for (const [address, info] of Object.entries(data.data || {})) {
      const symbol = Object.entries(TOKEN_ADDRESSES).find(([, v]) => v === address)?.[0];
      if (!symbol) continue;
      prices.push({
        symbol,
        address,
        price: info.value,
        source: 'birdeye',
        dex: 'birdeye',
        timestamp: info.updateUnixTime * 1000,
      });
    }
  } catch (e) {
    logger.error('[Birdeye] Multi-price error:', { error: String(e) });
  }
  return prices;
}

// ============= JUPITER (Aggregator) =============

export async function fetchJupiterPrices(symbols: string[]): Promise<DEXPrice[]> {
  const prices: DEXPrice[] = [];
  // Only fetch SOL-native tokens (not BTC/ETH which are wrapped with different prices)
  const solanaSymbols = symbols.filter(s => !['BTC', 'ETH'].includes(s));
  const ids = solanaSymbols.map(s => TOKEN_ADDRESSES[s]).filter(Boolean);
  if (ids.length === 0) return prices;

  try {
    // Use Jupiter Price API v2 (more stable endpoint)
    const resp = await fetchWithRetry(
      `${JUPITER_PRICE_API}?ids=${ids.join(',')}`,
      {},
      2  // Only 2 retries for Jupiter
    );
    const data = await resp.json() as { data: Record<string, { id: string; price: string }> };

    for (const [address, info] of Object.entries(data.data || {})) {
      const symbol = Object.entries(TOKEN_ADDRESSES).find(([, v]) => v === address)?.[0];
      if (!symbol) continue;
      prices.push({
        symbol,
        address,
        price: parseFloat(info.price),
        source: 'jupiter',
        dex: 'jupiter',
        timestamp: Date.now(),
      });
    }
  } catch (e) {
    logger.error('[Jupiter] Price API error:', { error: String(e) });
  }
  return prices;
}

// ============= DEXSCREENER (Backup) =============

// Symbols to skip for DEX (wrapped versions have different prices)
const SKIP_FOR_DEX = ['BTC', 'ETH'];

export async function fetchDexScreenerPrices(symbols: string[]): Promise<DEXPrice[]> {
  const prices: DEXPrice[] = [];
  // Filter out BTC/ETH - wrapped versions have different prices than spot
  const dexSymbols = symbols.filter(s => !SKIP_FOR_DEX.includes(s));

  for (const symbol of dexSymbols) {
    const address = TOKEN_ADDRESSES[symbol];
    if (!address) continue;

    try {
      const resp = await fetchWithRetry(`${DEXSCREENER_API}/tokens/${address}`);
      const data = await resp.json() as { pairs: DexScreenerPair[] };

      if (data.pairs && data.pairs.length > 0) {
        const bestPair = data.pairs.reduce((a, b) =>
          (a.liquidity?.usd || 0) > (b.liquidity?.usd || 0) ? a : b
        );

        // Use actual DEX from pair data, not 'dexscreener'
        const actualDex = bestPair.dexId?.toLowerCase() || 'unknown';
        const price = parseFloat(bestPair.priceUsd);

        // DEBUG: Log what we're getting
        logger.info(`[DexScreener] ${symbol}: dexId=${bestPair.dexId}, priceUsd=${bestPair.priceUsd}, parsed=${price}`);

        // Skip if price is invalid or DEX is unknown
        if (!price || price <= 0 || actualDex === 'unknown') {
          logger.warn(`[DexScreener] Skipping ${symbol}: invalid price (${price}) or DEX (${actualDex})`);
          continue;
        }

        prices.push({
          symbol,
          address,
          price,
          change24h: bestPair.priceChange?.h24,
          volume24h: bestPair.volume?.h24,
          liquidity: bestPair.liquidity?.usd,
          source: 'dexscreener',
          dex: actualDex as any, // Use actual DEX (orca, raydium, etc.)
          poolAddress: bestPair.pairAddress,
          timestamp: Date.now(),
        });

        logger.info(`[DexScreener] Added ${symbol} from ${actualDex}: $${price.toFixed(6)}`);
      }
    } catch (e) {
      logger.error(`[DexScreener] ${symbol} error:`, { error: String(e) });
    }
  }
  return prices;
}

// ============= RAYDIUM =============

interface RaydiumPair {
  name: string;
  ammId: string;
  baseMint: string;
  quoteMint: string;
  price: number;
  liquidity: number;
  volume24h: number;
  apr24h: number;
}

export async function fetchRaydiumPrices(symbols: string[]): Promise<DEXPrice[]> {
  const prices: DEXPrice[] = [];
  try {
    const resp = await fetchWithRetry(RAYDIUM_API);
    const pairs = await resp.json() as RaydiumPair[];

    // Find SOL pairs for requested symbols
    const solMint = TOKEN_ADDRESSES.SOL;
    const usdcMint = TOKEN_ADDRESSES.USDC;

    for (const symbol of symbols) {
      const tokenMint = TOKEN_ADDRESSES[symbol];
      if (!tokenMint) continue;

      // Find pair with USDC or SOL quote
      const pair = pairs.find(p =>
        (p.baseMint === tokenMint && (p.quoteMint === usdcMint || p.quoteMint === solMint)) ||
        (p.quoteMint === tokenMint && (p.baseMint === usdcMint || p.baseMint === solMint))
      );

      if (pair && pair.price > 0) {
        prices.push({
          symbol,
          address: tokenMint,
          price: pair.price,
          volume24h: pair.volume24h,
          liquidity: pair.liquidity,
          source: 'raydium',
          dex: 'raydium',
          poolAddress: pair.ammId,
          timestamp: Date.now(),
        });
      }
    }
  } catch (e) {
    logger.error('[Raydium] Error:', { error: String(e) });
  }
  return prices;
}

// ============= ORCA =============

interface OrcaWhirlpool {
  address: string;
  tokenA: { mint: string; symbol: string };
  tokenB: { mint: string; symbol: string };
  price: number;
  tvl: number;
  volume: { day: number };
  feeApr: { day: number };
  totalApr: { day: number };
}

// USDC/USDT mints for price validation
const STABLE_MINTS = [
  TOKEN_ADDRESSES.USDC,
  TOKEN_ADDRESSES.USDT,
];

export async function fetchOrcaPrices(symbols: string[]): Promise<DEXPrice[]> {
  const prices: DEXPrice[] = [];
  try {
    const resp = await fetchWithRetry(ORCA_API);
    const data = await resp.json() as { whirlpools: OrcaWhirlpool[] };

    for (const symbol of symbols) {
      const tokenMint = TOKEN_ADDRESSES[symbol];
      if (!tokenMint) continue;

      // Skip wrapped BTC/ETH - their prices don't match CEX prices
      if (symbol === 'BTC' || symbol === 'ETH') continue;

      // Find pool where tokenA or tokenB matches AND paired with USDC/USDT
      const pool = data.whirlpools?.find(p => {
        const isOurToken = p.tokenA?.mint === tokenMint || p.tokenB?.mint === tokenMint;
        const isPairedWithStable =
          STABLE_MINTS.includes(p.tokenA?.mint) || STABLE_MINTS.includes(p.tokenB?.mint);
        return isOurToken && isPairedWithStable;
      });

      if (pool && pool.price > 0) {
        // If our token is tokenB, invert price
        const price = pool.tokenA?.mint === tokenMint ? pool.price : 1 / pool.price;

        // Sanity check - skip obviously wrong prices
        if (symbol === 'SOL' && (price < 50 || price > 500)) continue;
        if (symbol === 'BONK' && price > 0.001) continue;

        prices.push({
          symbol,
          address: tokenMint,
          price,
          volume24h: pool.volume?.day,
          liquidity: pool.tvl,
          source: 'orca',
          dex: 'orca',
          poolAddress: pool.address,
          timestamp: Date.now(),
        });
      }
    }
  } catch (e) {
    logger.error('[Orca] Error:', { error: String(e) });
  }
  return prices;
}

// ============= METEORA =============

interface MeteoraPair {
  address: string;
  name: string;
  mint_x: string;
  mint_y: string;
  current_price: number;
  liquidity: string;
  trade_volume_24h: number;
  apr: number;
  apy: number;
}

export async function fetchMeteoraPrices(symbols: string[]): Promise<DEXPrice[]> {
  const prices: DEXPrice[] = [];
  try {
    const resp = await fetchWithRetry(METEORA_API);
    const pairs = await resp.json() as MeteoraPair[];

    for (const symbol of symbols) {
      const tokenMint = TOKEN_ADDRESSES[symbol];
      if (!tokenMint) continue;

      // Find pair where mint_x or mint_y matches
      const pair = pairs.find(p => p.mint_x === tokenMint || p.mint_y === tokenMint);

      if (pair && pair.current_price > 0) {
        // If our token is mint_y, invert price
        const price = pair.mint_x === tokenMint ? pair.current_price : 1 / pair.current_price;
        prices.push({
          symbol,
          address: tokenMint,
          price,
          volume24h: pair.trade_volume_24h,
          liquidity: parseFloat(pair.liquidity) || 0,
          source: 'meteora',
          dex: 'meteora',
          poolAddress: pair.address,
          timestamp: Date.now(),
        });
      }
    }
  } catch (e) {
    logger.error('[Meteora] Error:', { error: String(e) });
  }
  return prices;
}

// ============= COMBINED DEX FETCHER (All Sources) =============

// Helper to wrap promise with timeout
function withTimeout<T>(promise: Promise<T>, ms: number, fallback: T): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>(resolve => setTimeout(() => resolve(fallback), ms))
  ]);
}

export async function fetchBirdeyePrices(symbols: string[]): Promise<DEXPrice[]> {
  // DexScreener is fast and reliable - always fetch it
  // Other APIs are optional and may be slow
  const dexscreener = await withTimeout(fetchDexScreenerPrices(symbols), 10000, []);

  // Optionally fetch from other sources with shorter timeout
  const results = await Promise.allSettled([
    withTimeout(fetchRaydiumPrices(symbols), 8000, []),
    withTimeout(fetchOrcaPrices(symbols), 8000, []),
    withTimeout(fetchMeteoraPrices(symbols), 8000, []),
    withTimeout(fetchBirdeyePricesReal(symbols), 5000, []),
    withTimeout(fetchJupiterPrices(symbols), 5000, []),
  ]);

  const raydium = results[0].status === 'fulfilled' ? results[0].value : [];
  const orca = results[1].status === 'fulfilled' ? results[1].value : [];
  const meteora = results[2].status === 'fulfilled' ? results[2].value : [];
  const birdeye = results[3].status === 'fulfilled' ? results[3].value : [];
  const jupiter = results[4].status === 'fulfilled' ? results[4].value : [];

  logger.info(`[DEX] Raydium: ${raydium.length}, Orca: ${orca.length}, Meteora: ${meteora.length}, DexScreener: ${dexscreener.length}`);

  // Combine all - each source gets its own entry
  const priceMap = new Map<string, DEXPrice>();

  for (const p of dexscreener) priceMap.set(`${p.symbol}_dexscreener`, p);
  for (const p of raydium) priceMap.set(`${p.symbol}_raydium`, p);
  for (const p of orca) priceMap.set(`${p.symbol}_orca`, p);
  for (const p of meteora) priceMap.set(`${p.symbol}_meteora`, p);
  for (const p of jupiter) priceMap.set(`${p.symbol}_jupiter`, p);
  for (const p of birdeye) priceMap.set(`${p.symbol}_birdeye`, p);

  return Array.from(priceMap.values());
}

// Get all DEX prices separately for comparison
export async function fetchAllDEXPrices(symbols: string[]): Promise<{
  raydium: DEXPrice[];
  orca: DEXPrice[];
  meteora: DEXPrice[];
  dexscreener: DEXPrice[];
  birdeye: DEXPrice[];
  jupiter: DEXPrice[];
}> {
  const results = await Promise.allSettled([
    withTimeout(fetchRaydiumPrices(symbols), 10000, []),
    withTimeout(fetchOrcaPrices(symbols), 10000, []),
    withTimeout(fetchMeteoraPrices(symbols), 10000, []),
    withTimeout(fetchDexScreenerPrices(symbols), 10000, []),
    withTimeout(fetchBirdeyePricesReal(symbols), 5000, []),
    withTimeout(fetchJupiterPrices(symbols), 5000, []),
  ]);

  return {
    raydium: results[0].status === 'fulfilled' ? results[0].value : [],
    orca: results[1].status === 'fulfilled' ? results[1].value : [],
    meteora: results[2].status === 'fulfilled' ? results[2].value : [],
    dexscreener: results[3].status === 'fulfilled' ? results[3].value : [],
    birdeye: results[4].status === 'fulfilled' ? results[4].value : [],
    jupiter: results[5].status === 'fulfilled' ? results[5].value : [],
  };
}

export async function fetchDexScreenerPools(baseToken = 'SOL'): Promise<DEXPrice[]> {
  const prices: DEXPrice[] = [];
  const address = TOKEN_ADDRESSES[baseToken];
  if (!address) return prices;

  try {
    const resp = await fetchWithRetry(`${DEXSCREENER_API}/tokens/${address}`);
    const data = await resp.json() as { pairs: DexScreenerPair[] };

    for (const pair of (data.pairs || []).slice(0, 10)) {
      prices.push({
        symbol: pair.baseToken.symbol,
        price: parseFloat(pair.priceUsd),
        change24h: pair.priceChange?.h24,
        volume24h: pair.volume?.h24,
        liquidity: pair.liquidity?.usd,
        source: 'dexscreener',
        dex: pair.dexId as DEXPrice['dex'],
        poolAddress: pair.pairAddress,
        timestamp: Date.now(),
      });
    }
  } catch (e) {
    logger.error('[DexScreener] Fetch error:', { error: String(e) });
  }
  return prices;
}

// ============= LP POOLS (Birdeye + DexScreener) =============

async function fetchBirdeyeLPPools(): Promise<LPPool[]> {
  const pools: LPPool[] = [];
  const apiKey = getBirdeyeApiKey();
  if (!apiKey) return pools;

  try {
    const resp = await fetchWithRetry(
      `${BIRDEYE_API}/defi/v3/pool/list?sort_by=liquidity&sort_type=desc&limit=20`,
      { headers: { 'X-API-KEY': apiKey, 'x-chain': 'solana' } }
    );
    const data = await resp.json() as {
      success: boolean;
      data: { items: Array<{
        address: string; name: string; source: string;
        token_1_symbol: string; token_2_symbol: string;
        liquidity: number; trade_24h_usd: number;
        apy: number; fee_rate: number;
      }> }
    };

    if (!data.success) return pools;

    for (const item of data.data?.items || []) {
      pools.push({
        name: item.name || `${item.token_1_symbol}/${item.token_2_symbol}`,
        address: item.address,
        dex: item.source,
        token0: item.token_1_symbol,
        token1: item.token_2_symbol,
        apy: item.apy || 0,
        apr: item.apy ? item.apy * 0.9 : 0,
        tvl: item.liquidity,
        volume24h: item.trade_24h_usd,
        fees24h: item.trade_24h_usd * (item.fee_rate / 100),
        feeRate: item.fee_rate,
        utilization: item.trade_24h_usd / (item.liquidity || 1),
        riskScore: item.liquidity > 10_000_000 ? 2 : item.liquidity > 1_000_000 ? 4 : 6,
      });
    }
  } catch (e) {
    logger.error('[Birdeye] Pool fetch error:', { error: String(e) });
  }
  return pools;
}

async function fetchDexScreenerLPPools(): Promise<LPPool[]> {
  const pools: LPPool[] = [];

  // Fetch pools for multiple base tokens to get variety
  const baseTokens = ['SOL', 'JUP', 'RAY', 'ORCA', 'BONK', 'USDC'];

  for (const baseToken of baseTokens) {
    const address = TOKEN_ADDRESSES[baseToken];
    if (!address) continue;

    try {
      const resp = await fetchWithRetry(`${DEXSCREENER_API}/tokens/${address}`);
      const data = await resp.json() as { pairs: DexScreenerPair[] };

      if (!data.pairs) continue;

      const sortedPairs = data.pairs
        .filter(p => p.liquidity?.usd > 100000)
        .sort((a, b) => (b.liquidity?.usd || 0) - (a.liquidity?.usd || 0))
        .slice(0, 10); // Top 10 per token

    for (const pair of sortedPairs) {
      const volume = pair.volume?.h24 || 0;
      const tvl = pair.liquidity?.usd || 0;
      const feeRate = pair.dexId === 'meteora' ? 0.25 : pair.dexId === 'raydium' ? 0.25 : 0.3;
      const fees24h = volume * (feeRate / 100);
      const apr = tvl > 0 ? (fees24h * 365 / tvl) * 100 : 0;

        pools.push({
          name: `${pair.baseToken.symbol}/${pair.quoteToken.symbol}`,
          address: pair.pairAddress,
          dex: pair.dexId,
          token0: pair.baseToken.symbol,
          token1: pair.quoteToken.symbol,
          apy: apr * 1.1,
          apr,
          tvl,
          volume24h: volume,
          fees24h,
          feeRate,
          utilization: tvl > 0 ? volume / tvl : 0,
          riskScore: tvl > 10_000_000 ? 2 : tvl > 1_000_000 ? 4 : 6,
        });
      }
    } catch (e) {
      logger.error(`[DexScreener] ${baseToken} pool fetch error:`, { error: String(e) });
    }
  }

  return pools;
}

async function fetchRaydiumLPPools(): Promise<LPPool[]> {
  const pools: LPPool[] = [];
  try {
    const resp = await fetchWithRetry(RAYDIUM_API);
    const pairs = await resp.json() as RaydiumPair[];

    // Filter for high liquidity SOL pairs
    const solPairs = pairs
      .filter(p => p.liquidity > 100000 && (p.name.includes('SOL') || p.name.includes('USDC')))
      .sort((a, b) => b.liquidity - a.liquidity)
      .slice(0, 15);

    for (const pair of solPairs) {
      const [token0, token1] = pair.name.split('/');
      const apr = pair.apr24h || 0;

      pools.push({
        name: pair.name,
        address: pair.ammId,
        dex: 'raydium',
        token0,
        token1,
        apy: apr * 1.1,
        apr,
        tvl: pair.liquidity,
        volume24h: pair.volume24h,
        fees24h: pair.volume24h * 0.0025,
        feeRate: 0.25,
        utilization: pair.liquidity > 0 ? pair.volume24h / pair.liquidity : 0,
        riskScore: pair.liquidity > 10_000_000 ? 2 : pair.liquidity > 1_000_000 ? 4 : 6,
      });
    }
  } catch (e) {
    logger.error('[Raydium] Pool fetch error:', { error: String(e) });
  }
  return pools;
}

async function fetchOrcaLPPools(): Promise<LPPool[]> {
  const pools: LPPool[] = [];
  try {
    const resp = await fetchWithRetry(ORCA_API);
    const data = await resp.json() as { whirlpools: OrcaWhirlpool[] };

    // Filter for high TVL pools
    const topPools = (data.whirlpools || [])
      .filter(p => p.tvl > 100000)
      .sort((a, b) => b.tvl - a.tvl)
      .slice(0, 15);

    for (const pool of topPools) {
      const feeApr = (pool.feeApr?.day || 0) * 100;
      const totalApr = (pool.totalApr?.day || 0) * 100 || feeApr;

      pools.push({
        name: `${pool.tokenA?.symbol}/${pool.tokenB?.symbol}`,
        address: pool.address,
        dex: 'orca',
        token0: pool.tokenA?.symbol || '',
        token1: pool.tokenB?.symbol || '',
        apy: totalApr * 1.1,
        apr: totalApr,
        tvl: pool.tvl,
        volume24h: pool.volume?.day || 0,
        fees24h: (pool.volume?.day || 0) * 0.003,
        feeRate: 0.3,
        utilization: pool.tvl > 0 ? (pool.volume?.day || 0) / pool.tvl : 0,
        riskScore: pool.tvl > 10_000_000 ? 2 : pool.tvl > 1_000_000 ? 4 : 6,
      });
    }
  } catch (e) {
    logger.error('[Orca] Pool fetch error:', { error: String(e) });
  }
  return pools;
}

async function fetchMeteoraLPPools(): Promise<LPPool[]> {
  const pools: LPPool[] = [];
  try {
    const resp = await fetchWithRetry(METEORA_API);
    const pairs = await resp.json() as MeteoraPair[];

    // Filter for high liquidity pools
    const topPairs = pairs
      .filter(p => parseFloat(p.liquidity) > 10000 && p.name.includes('SOL'))
      .sort((a, b) => parseFloat(b.liquidity) - parseFloat(a.liquidity))
      .slice(0, 15);

    for (const pair of topPairs) {
      const [token0, token1] = pair.name.split('-');
      const tvl = parseFloat(pair.liquidity) || 0;

      pools.push({
        name: pair.name.replace('-', '/'),
        address: pair.address,
        dex: 'meteora',
        token0: token0 || '',
        token1: token1 || '',
        apy: pair.apy * 100,
        apr: pair.apr * 100,
        tvl,
        volume24h: pair.trade_volume_24h,
        fees24h: pair.trade_volume_24h * 0.002,
        feeRate: 0.2,
        utilization: tvl > 0 ? pair.trade_volume_24h / tvl : 0,
        riskScore: tvl > 10_000_000 ? 2 : tvl > 1_000_000 ? 4 : 6,
      });
    }
  } catch (e) {
    logger.error('[Meteora] Pool fetch error:', { error: String(e) });
  }
  return pools;
}

export async function fetchTopLPPools(): Promise<LPPool[]> {
  // DexScreener is fast - always fetch first
  const dexscreenerPools = await withTimeout(fetchDexScreenerLPPools(), 10000, []);

  // Other APIs are optional - use Promise.allSettled with timeout
  const results = await Promise.allSettled([
    withTimeout(fetchRaydiumLPPools(), 10000, []),
    withTimeout(fetchOrcaLPPools(), 10000, []),
    withTimeout(fetchMeteoraLPPools(), 10000, []),
    withTimeout(fetchBirdeyeLPPools(), 5000, []),
  ]);

  const raydiumPools = results[0].status === 'fulfilled' ? results[0].value : [];
  const orcaPools = results[1].status === 'fulfilled' ? results[1].value : [];
  const meteoraPools = results[2].status === 'fulfilled' ? results[2].value : [];
  const birdeyePools = results[3].status === 'fulfilled' ? results[3].value : [];

  logger.info(`[LP Pools] Raydium: ${raydiumPools.length}, Orca: ${orcaPools.length}, Meteora: ${meteoraPools.length}, DexScreener: ${dexscreenerPools.length}`);

  // Merge all pools - prefer native APIs over DexScreener
  const poolMap = new Map<string, LPPool>();
  for (const p of dexscreenerPools) poolMap.set(p.address, p);
  for (const p of birdeyePools) poolMap.set(p.address, p);
  for (const p of raydiumPools) poolMap.set(p.address, p);
  for (const p of orcaPools) poolMap.set(p.address, p);
  for (const p of meteoraPools) poolMap.set(p.address, p);

  return Array.from(poolMap.values())
    .sort((a, b) => b.tvl - a.tvl)
    .slice(0, 50); // Increased from 30 to 50 for more variety
}

// ============= NEW TOKENS (Birdeye + DexScreener) =============

async function fetchBirdeyeNewTokens(minLiquidity: number): Promise<NewToken[]> {
  const tokens: NewToken[] = [];
  const apiKey = getBirdeyeApiKey();
  if (!apiKey) return tokens;

  try {
    const resp = await fetchWithRetry(
      `${BIRDEYE_API}/defi/v3/token/new_listing?limit=50&min_liquidity=${minLiquidity}`,
      { headers: { 'X-API-KEY': apiKey, 'x-chain': 'solana' } }
    );
    const data = await resp.json() as {
      success: boolean;
      data: { items: Array<{
        address: string; symbol: string; name: string;
        created_at: number; price: number; price_change_24h: number;
        liquidity: number; volume_24h: number; holder: number;
      }> }
    };

    if (!data.success) return tokens;

    for (const item of (data.data?.items || []).slice(0, 10)) {
      tokens.push({
        symbol: item.symbol,
        name: item.name,
        address: item.address,
        launchTime: item.created_at,
        price: item.price,
        priceChange24h: item.price_change_24h,
        liquidity: item.liquidity,
        volume24h: item.volume_24h,
        holders: item.holder,
        riskLevel: item.liquidity > 500000 ? 'medium' : 'high',
      });
    }
  } catch (e) {
    logger.error('[Birdeye] New tokens fetch error:', { error: String(e) });
  }
  return tokens;
}

async function fetchDexScreenerNewTokens(minLiquidity: number): Promise<NewToken[]> {
  const tokens: NewToken[] = [];

  try {
    const resp = await fetchWithRetry(`${DEXSCREENER_API}/search?q=solana`);
    const data = await resp.json() as { pairs: DexScreenerPair[] };

    if (!data.pairs) return tokens;

    const now = Date.now();
    const oneDayAgo = now - 24 * 60 * 60 * 1000;

    const newPairs = data.pairs
      .filter(p =>
        p.pairCreatedAt > oneDayAgo &&
        (p.liquidity?.usd || 0) >= minLiquidity
      )
      .sort((a, b) => (b.priceChange?.h24 || 0) - (a.priceChange?.h24 || 0))
      .slice(0, 10);

    for (const pair of newPairs) {
      const liq = pair.liquidity?.usd || 0;
      tokens.push({
        symbol: pair.baseToken.symbol,
        name: pair.baseToken.symbol,
        address: pair.baseToken.address,
        launchTime: pair.pairCreatedAt,
        price: parseFloat(pair.priceUsd),
        priceChange24h: pair.priceChange?.h24 || 0,
        liquidity: liq,
        volume24h: pair.volume?.h24 || 0,
        holders: pair.txns?.h24?.buys || 0,
        riskLevel: liq > 500000 ? 'medium' : 'high',
      });
    }
  } catch (e) {
    logger.error('[DexScreener] New tokens fetch error:', { error: String(e) });
  }
  return tokens;
}

export async function fetchNewTokens(minLiquidity = 50000): Promise<NewToken[]> {
  const [birdeyeTokens, dexscreenerTokens] = await Promise.all([
    fetchBirdeyeNewTokens(minLiquidity),
    fetchDexScreenerNewTokens(minLiquidity),
  ]);

  logger.info(`[New Tokens] Birdeye: ${birdeyeTokens.length}, DexScreener: ${dexscreenerTokens.length}`);

  // Merge by address
  const tokenMap = new Map<string, NewToken>();
  for (const t of dexscreenerTokens) tokenMap.set(t.address, t);
  for (const t of birdeyeTokens) tokenMap.set(t.address, t);

  return Array.from(tokenMap.values())
    .sort((a, b) => b.priceChange24h - a.priceChange24h)
    .slice(0, 10);
}

// ============= TRENDING TOKENS (DexScreener) =============

export async function fetchTrendingTokens(): Promise<string[]> {
  // Get trending/boosted tokens from DexScreener
  try {
    const resp = await fetchWithRetry('https://api.dexscreener.com/token-boosts/top/v1', {}, 2, 5000);
    if (!resp.ok) return [];

    const data = await resp.json() as Array<{ tokenAddress: string; chainId: string; description?: string }>;

    // Filter for Solana tokens and get unique symbols
    const solanaTokens = data
      .filter((t: { chainId: string }) => t.chainId === 'solana')
      .slice(0, 10);

    // For each token, we need to get the symbol from the pair data
    const symbols = new Set<string>();

    for (const token of solanaTokens.slice(0, 5)) {
      try {
        const pairResp = await fetchWithRetry(
          `${DEXSCREENER_API}/tokens/${token.tokenAddress}`,
          {},
          1,
          3000
        );
        if (pairResp.ok) {
          const pairData = await pairResp.json() as { pairs?: DexScreenerPair[] };
          if (pairData.pairs?.[0]) {
            symbols.add(pairData.pairs[0].baseToken.symbol);
          }
        }
      } catch {
        // Skip on error
      }
    }

    return Array.from(symbols);
  } catch (e) {
    logger.error('[DexScreener] Trending tokens error:', { error: String(e) });
    return [];
  }
}
