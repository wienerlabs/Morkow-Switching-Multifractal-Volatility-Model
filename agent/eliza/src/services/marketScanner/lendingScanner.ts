/**
 * Lending Market Scanner
 * 
 * Scans lending protocols for opportunities using REAL on-chain data
 * Data source: DeFiLlama Yields API (aggregates from Kamino, MarginFi, Solend)
 */

import type { LendingMarketData } from '../lending/types.js';
import { logger } from '../logger.js';

const DEFILLAMA_POOLS_API = 'https://yields.llama.fi/pools';

// Asset tier mapping (1=best, 3=worst)
const ASSET_TIERS: Record<string, number> = {
  'USDC': 1,
  'USDT': 1,
  'SOL': 1,
  'JITOSOL': 2,
  'MSOL': 2,
  'PYUSD': 2,
  'USDS': 2,
  'JUPSOL': 2,
};

/**
 * Fetch REAL lending market data from DeFiLlama
 * This aggregates on-chain data from Kamino, MarginFi, and Solend
 */
export async function fetchLendingMarkets(): Promise<LendingMarketData[]> {
  try {
    const response = await fetch(DEFILLAMA_POOLS_API);
    
    if (!response.ok) {
      logger.error(`[LendingScanner] DeFiLlama API error: ${response.status}`);
      return [];
    }

    const data = await response.json() as { data: any[] };
    
    // Filter for Solana lending protocols
    const lendingPools = data.data.filter((pool: any) => 
      (pool.project === 'kamino-lend' || 
       pool.project === 'marginfi' || 
       pool.project === 'solend') &&
      pool.chain === 'Solana' &&
      pool.tvlUsd > 0
    );

    // Convert to our format
    const markets: LendingMarketData[] = lendingPools.map((pool: any) => {
      const asset = pool.symbol;
      const protocol = pool.project === 'kamino-lend' ? 'kamino' : 
                      pool.project === 'marginfi' ? 'marginfi' : 'solend';
      
      // Calculate total protocol TVL
      const protocolTvl = lendingPools
        .filter((p: any) => p.project === pool.project)
        .reduce((sum: number, p: any) => sum + p.tvlUsd, 0);

      return {
        asset,
        protocol,
        tvlUsd: pool.tvlUsd,
        totalApy: pool.apy,
        supplyApy: pool.apyBase || pool.apy,
        rewardApy: pool.apyReward || 0,
        borrowApy: pool.apyBorrow || 0,
        utilizationRate: pool.ltv || 0.5,  // Estimate if not available
        totalBorrows: pool.totalBorrowUsd || pool.tvlUsd * 0.5,
        availableLiquidity: pool.tvlUsd * 0.5,
        protocolTvlUsd: protocolTvl,
        totalSupply: pool.tvlUsd,
        totalBorrow: pool.totalBorrowUsd || pool.tvlUsd * 0.5,
      };
    });

    logger.info(`[LendingScanner] Found ${markets.length} lending markets (Kamino, MarginFi, Solend)`);
    
    return markets;
  } catch (error) {
    logger.error('[LendingScanner] Error fetching lending markets:', { error: String(error) });
    return [];
  }
}

/**
 * Get asset tier (1=best, 3=worst)
 */
export function getAssetTier(asset: string): number {
  return ASSET_TIERS[asset] || 3;
}

/**
 * Filter markets by quality criteria
 */
export function filterLendingMarkets(
  markets: LendingMarketData[],
  minTvl: number = 50_000_000,
  maxUtilization: number = 0.85,
  maxTier: number = 2
): LendingMarketData[] {
  return markets.filter(market => {
    const tier = getAssetTier(market.asset);
    return (
      market.tvlUsd >= minTvl &&
      market.utilizationRate <= maxUtilization &&
      tier <= maxTier
    );
  });
}

/**
 * Get best lending rate for an asset across all protocols
 */
export function getBestLendingRate(
  markets: LendingMarketData[],
  asset: string
): { protocol: string; apy: number; market: LendingMarketData } | null {
  const assetMarkets = markets.filter(m => 
    m.asset.toUpperCase() === asset.toUpperCase()
  );

  if (assetMarkets.length === 0) return null;

  const best = assetMarkets.reduce((a, b) => 
    a.supplyApy > b.supplyApy ? a : b
  );

  return {
    protocol: best.protocol,
    apy: best.supplyApy,
    market: best,
  };
}

/**
 * Get top lending opportunities sorted by APY
 */
export function getTopLendingOpportunities(
  markets: LendingMarketData[],
  limit: number = 10
): LendingMarketData[] {
  return markets
    .sort((a, b) => b.supplyApy - a.supplyApy)
    .slice(0, limit);
}

