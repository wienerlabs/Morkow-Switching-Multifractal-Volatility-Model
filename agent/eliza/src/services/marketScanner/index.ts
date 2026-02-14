/**
 * Market Scanner - Main Module
 * Scans CEX + DEX markets for opportunities
 * Uses: Birdeye, Jupiter, DexScreener, CoinGecko, Binance, Coinbase, Kraken
 *
 * NEW: Also scans perps markets for funding rate opportunities
 */

import type { MarketSnapshot, ScannerConfig } from './types.js';
import { fetchAllCEXPrices } from './cexFetcher.js';
import { fetchBirdeyePrices, fetchTopLPPools, fetchNewTokens, fetchTrendingTokens } from './dexFetcher.js';
import { fetchCoinGeckoPrices } from './coingeckoFetcher.js';
import { detectArbitrage } from './arbitrageDetector.js';
import { selectBestStrategy } from './strategySelector.js';
import { renderDashboard } from './dashboard.js';
import { getPerpsScanner } from '../perps/perpsScanner.js';
import { fetchLendingMarkets } from './lendingScanner.js';
import { fetchSpotTokens } from './spotScanner.js';

export * from './types.js';
export { fetchAllCEXPrices } from './cexFetcher.js';
export { fetchBirdeyePrices, fetchTopLPPools, fetchNewTokens, fetchAllDEXPrices, fetchTrendingTokens } from './dexFetcher.js';
export { fetchCoinGeckoPrices } from './coingeckoFetcher.js';
export { detectArbitrage } from './arbitrageDetector.js';
export { selectBestStrategy } from './strategySelector.js';
export { renderDashboard } from './dashboard.js';
export { fetchSpotTokens } from './spotScanner.js';

// Core tokens always monitored
const CORE_TOKENS = ['SOL', 'BTC', 'ETH', 'WIF', 'BONK', 'JUP'];

const DEFAULT_CONFIG: ScannerConfig = {
  tokens: CORE_TOKENS,
  refreshInterval: 30000, // 30 seconds
  minArbitrageSpread: 0.5, // Min 0.5% spread after fees
  minLiquidity: 50000,
  maxRiskScore: 7,
};

// Cache for trending tokens (refresh every 5 minutes)
let trendingTokensCache: string[] = [];
let lastTrendingFetch = 0;
const TRENDING_CACHE_TTL = 5 * 60 * 1000; // 5 minutes

export async function scanMarkets(config: Partial<ScannerConfig> = {}): Promise<MarketSnapshot> {
  const cfg = { ...DEFAULT_CONFIG, ...config };

  // Fetch trending tokens periodically
  if (Date.now() - lastTrendingFetch > TRENDING_CACHE_TTL) {
    try {
      const trending = await fetchTrendingTokens();
      if (trending.length > 0) {
        trendingTokensCache = trending;
        lastTrendingFetch = Date.now();
        console.log(`[Scanner] Trending tokens: ${trending.join(', ')}`);
      }
    } catch (e) {
      // Ignore errors, use cached or empty
    }
  }

  // Combine core tokens with trending tokens
  const allTokens = [...new Set([...cfg.tokens, ...trendingTokensCache])];

  console.log('[Scanner] Fetching from all sources...');
  console.log(`[Scanner] Tokens: ${allTokens.join(', ')}`);
  console.log('[Scanner] CEX: Binance ✅, Coinbase ✅, Kraken ✅');
  console.log('[Scanner] DEX: DexScreener ✅ (Birdeye ❌ suspended, Jupiter ❌ requires paid API)');
  console.log('[Scanner] Verification: CoinGecko ✅');

  // Initialize perps scanner (will use cached clients if available)
  const perpsScanner = getPerpsScanner();

  // Fetch all data in parallel - ALL SOURCES including perps, lending, and spot
  const [cexPrices, dexPrices, coingeckoPrices, lpPools, newTokens, perpsData, lendingMarkets, spotTokens] = await Promise.all([
    fetchAllCEXPrices(allTokens),
    fetchBirdeyePrices(allTokens),  // Falls back to DexScreener when Birdeye/Jupiter fail
    fetchCoinGeckoPrices(allTokens),
    fetchTopLPPools(),
    fetchNewTokens(cfg.minLiquidity),
    perpsScanner.scan().catch(() => ({ perpsOpportunities: [], fundingArbitrage: [] })),
    fetchLendingMarkets().catch(() => []),
    fetchSpotTokens().catch(() => []),
  ]);

  // Log results
  console.log(`[Scanner] Results: CEX ${cexPrices.length} | DEX ${dexPrices.length} | CoinGecko ${coingeckoPrices.length} | Pools ${lpPools.length}`);
  console.log(`[Scanner] Perps: ${perpsData.perpsOpportunities.length} opportunities | ${perpsData.fundingArbitrage.length} funding arb`);
  console.log(`[Scanner] Lending: ${lendingMarkets.length} markets`);
  console.log(`[Scanner] Spot: ${spotTokens.length} tokens`);

  // Detect arbitrage opportunities
  const arbitrage = detectArbitrage(cexPrices, dexPrices, cfg.minArbitrageSpread);

  // Filter pools by risk
  const filteredPools = lpPools
    .filter(p => p.riskScore <= cfg.maxRiskScore)
    .sort((a, b) => b.apy - a.apy);

  // Select best strategy (now includes perps)
  const bestStrategy = selectBestStrategy(
    arbitrage,
    filteredPools,
    newTokens,
    perpsData.perpsOpportunities,
    perpsData.fundingArbitrage
  );

  return {
    timestamp: Date.now(),
    cexPrices,
    dexPrices,
    arbitrage,
    lpPools: filteredPools,
    newTokens,
    perpsOpportunities: perpsData.perpsOpportunities,
    fundingArbitrage: perpsData.fundingArbitrage,
    lendingMarkets,
    spotTokens,
    bestStrategy,
  };
}

export async function startScanner(config: Partial<ScannerConfig> = {}): Promise<void> {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  
  console.log('[Scanner] Starting Market Scanner...');
  console.log(`[Scanner] Tokens: ${cfg.tokens.join(', ')}`);
  console.log(`[Scanner] Refresh: ${cfg.refreshInterval / 1000}s`);
  
  const scan = async () => {
    try {
      const snapshot = await scanMarkets(cfg);
      renderDashboard(snapshot);
    } catch (error) {
      console.error('[Scanner] Error:', error);
    }
  };
  
  // Initial scan
  await scan();
  
  // Periodic scans
  setInterval(scan, cfg.refreshInterval);
}

// CLI entry point
if (import.meta.url === `file://${process.argv[1]}`) {
  startScanner().catch(console.error);
}

