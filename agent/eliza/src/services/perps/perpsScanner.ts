/**
 * Perps Scanner
 * 
 * Scans perpetual futures markets for trading opportunities:
 * - Negative funding rate opportunities (get paid to hold)
 * - Funding rate arbitrage across venues
 * - High conviction directional trades based on OI/momentum
 * 
 * Integrates with the main TradingAgent loop.
 */
import { logger } from '../logger.js';
import type { PerpsOpportunity, FundingArbitrageOpportunity } from '../marketScanner/types.js';
import type { PerpsVenue, FundingRate, PositionSide } from '../../types/perps.js';
import { FundingRateAggregator, getFundingRateAggregator } from './fundingRateAggregator.js';
import { DriftClient } from './driftClient.js';
import { AdrenaClient } from './adrenaClient.js';

// ============= CONFIGURATION =============

export interface PerpsScannnerConfig {
  // Funding thresholds
  minNegativeFundingBps: number;      // Min negative funding to consider (default: 5 = 0.05%)
  minFundingArbSpreadBps: number;     // Min spread for funding arb (default: 10 = 0.1%)
  
  // Risk filters
  minOiUsd: number;                   // Min open interest (default: $1M)
  maxOiChangePct: number;             // Max OI change to avoid crowded trades (default: 50%)
  minLiquidityUsd: number;            // Min liquidity depth (default: $500K)
  
  // Position sizing
  defaultLeverage: number;            // Default leverage for opportunities (default: 3)
  holdingPeriodHours: number;         // Expected hold time (default: 24)
  
  // Markets to scan
  markets: string[];                  // e.g., ['SOL-PERP', 'BTC-PERP', 'ETH-PERP']
}

export const DEFAULT_PERPS_SCANNER_CONFIG: PerpsScannnerConfig = {
  minNegativeFundingBps: 5,           // 0.05% hourly minimum
  minFundingArbSpreadBps: 10,         // 0.10% spread minimum
  minOiUsd: 500_000,                  // $500K minimum OI
  maxOiChangePct: 50,                 // Max 50% OI change
  minLiquidityUsd: 200_000,           // $200K liquidity minimum
  defaultLeverage: 3,
  holdingPeriodHours: 24,
  markets: ['SOL-PERP', 'BTC-PERP', 'ETH-PERP'],
};

// ============= PERPS SCANNER =============

export class PerpsScanner {
  private config: PerpsScannnerConfig;
  private fundingAggregator: FundingRateAggregator;
  private driftClient: DriftClient | null = null;
  private adrenaClient: AdrenaClient | null = null;
  
  // Cache for OI data
  private oiCache: Map<string, { oi: number; timestamp: number }> = new Map();
  private previousOiCache: Map<string, number> = new Map();

  constructor(config: Partial<PerpsScannnerConfig> = {}) {
    this.config = { ...DEFAULT_PERPS_SCANNER_CONFIG, ...config };
    this.fundingAggregator = getFundingRateAggregator();
    logger.info('PerpsScanner initialized', { config: this.config });
  }

  /**
   * Set venue clients for data fetching
   */
  setClients(drift: DriftClient | null, adrena: AdrenaClient | null): void {
    this.driftClient = drift;
    this.adrenaClient = adrena;
    
    // Also set clients on the aggregator
    this.fundingAggregator.setClients({
      drift: drift || undefined,
      adrena: adrena || undefined,
    });
    
    logger.debug('PerpsScanner clients set', {
      drift: !!drift,
      adrena: !!adrena,
    });
  }

  /**
   * Scan for all perps opportunities
   */
  async scan(): Promise<{
    perpsOpportunities: PerpsOpportunity[];
    fundingArbitrage: FundingArbitrageOpportunity[];
  }> {
    logger.info('Starting perps scan...');
    const startTime = Date.now();

    try {
      // Fetch all funding rates
      const fundingRates = await this.fundingAggregator.fetchAllFundingRates();
      
      // Find single-venue opportunities (negative funding)
      const perpsOpportunities = await this.findNegativeFundingOpportunities(fundingRates);
      
      // Find cross-venue arbitrage opportunities
      const fundingArbitrage = await this.findFundingArbitrageOpportunities();

      logger.info('Perps scan complete', {
        durationMs: Date.now() - startTime,
        perpsOpportunities: perpsOpportunities.length,
        fundingArbitrage: fundingArbitrage.length,
      });

      return { perpsOpportunities, fundingArbitrage };
    } catch (error) {
      logger.error('Perps scan failed', { error });
      return { perpsOpportunities: [], fundingArbitrage: [] };
    }
  }

  /**
   * Find negative funding opportunities (get paid to hold)
   * - Negative funding for longs means shorts pay longs
   * - Positive funding for shorts means longs pay shorts
   */
  private async findNegativeFundingOpportunities(
    fundingRates: FundingRate[]
  ): Promise<PerpsOpportunity[]> {
    const opportunities: PerpsOpportunity[] = [];

    for (const rate of fundingRates) {
      // Skip if rate is too small
      const rateBps = Math.abs(rate.rate) * 10000;
      if (rateBps < this.config.minNegativeFundingBps) {
        continue;
      }

      // Determine trade direction based on funding
      // Negative rate = shorts pay longs → go LONG
      // Positive rate = longs pay shorts → go SHORT
      const side: PositionSide = rate.rate < 0 ? 'long' : 'short';

      // Get market data for risk metrics
      const marketData = await this.getMarketData(rate.venue, rate.market);
      if (!marketData) continue;

      // Filter by OI and liquidity
      if (marketData.openInterest < this.config.minOiUsd) continue;
      if (marketData.liquidityDepth < this.config.minLiquidityUsd) continue;

      // Calculate expected return
      const hourlyReturn = Math.abs(rate.rate);
      const expectedReturnPct = hourlyReturn * this.config.holdingPeriodHours * 100;
      const estimatedProfitUsd = (1000 * expectedReturnPct) / 100; // For $1000 position

      // Calculate risk score
      const riskScore = this.calculateRiskScore(marketData, rate);

      // Determine confidence
      const confidence = this.determineConfidence(rateBps, marketData, riskScore);

      opportunities.push({
        market: rate.market,
        venue: rate.venue,
        side,
        fundingRate: rate.rate,
        annualizedRate: rate.annualizedRate,
        nextFundingTime: rate.nextFundingTime,
        markPrice: marketData.markPrice,
        indexPrice: marketData.indexPrice,
        perpSpotBasis: marketData.perpSpotBasis,
        expectedReturnPct,
        holdingPeriodHours: this.config.holdingPeriodHours,
        estimatedProfitUsd,
        openInterest: marketData.openInterest,
        oiChangePct24h: marketData.oiChangePct24h,
        liquidityDepth: marketData.liquidityDepth,
        maxLeverage: marketData.maxLeverage,
        confidence,
        riskScore,
      });
    }

    // Sort by expected return (descending)
    opportunities.sort((a, b) => b.expectedReturnPct - a.expectedReturnPct);

    return opportunities;
  }

  /**
   * Find funding rate arbitrage opportunities between venues
   */
  private async findFundingArbitrageOpportunities(): Promise<FundingArbitrageOpportunity[]> {
    const arbOpps = await this.fundingAggregator.findArbitrageOpportunities(
      this.config.minFundingArbSpreadBps
    );

    return arbOpps.map(opp => ({
      ...opp,
      confidence: this.determineArbConfidence(opp.estimatedProfitBps),
    }));
  }

  /**
   * Get market data for a specific venue and market
   */
  private async getMarketData(
    venue: PerpsVenue,
    market: string
  ): Promise<{
    markPrice: number;
    indexPrice: number;
    perpSpotBasis: number;
    openInterest: number;
    oiChangePct24h: number;
    liquidityDepth: number;
    maxLeverage: number;
  } | null> {
    try {
      if (venue === 'drift' && this.driftClient) {
        const markets = await this.driftClient.getMarkets();
        const marketInfo = markets.find(m =>
          m.symbol.toUpperCase().includes(market.toUpperCase().replace('-PERP', ''))
        );

        if (!marketInfo) return null;

        const totalOi = (marketInfo.openInterestLong + marketInfo.openInterestShort) * marketInfo.markPrice;
        const oiChangePct = this.calculateOiChange(market, totalOi);

        return {
          markPrice: marketInfo.markPrice,
          indexPrice: marketInfo.indexPrice,
          perpSpotBasis: ((marketInfo.markPrice - marketInfo.indexPrice) / marketInfo.indexPrice) * 100,
          openInterest: totalOi,
          oiChangePct24h: oiChangePct,
          liquidityDepth: totalOi * 0.1, // Estimate: 10% of OI
          maxLeverage: marketInfo.maxLeverage,
        };
      }

      if (venue === 'adrena' && this.adrenaClient) {
        const stats = await this.adrenaClient.getMarketStats();
        const marketStats = stats.find(s =>
          s.market.toUpperCase().includes(market.toUpperCase().replace('-PERP', ''))
        );

        if (!marketStats) return null;

        // Adrena doesn't expose OI directly, estimate from market cap
        const estimatedOi = 5_000_000; // Default estimate
        const oiChangePct = this.calculateOiChange(market, estimatedOi);

        return {
          markPrice: marketStats.markPrice,
          indexPrice: marketStats.markPrice, // Use mark as index (no separate index)
          perpSpotBasis: 0, // No basis data available
          openInterest: estimatedOi,
          oiChangePct24h: oiChangePct,
          liquidityDepth: estimatedOi * 0.1,
          maxLeverage: marketStats.maxLeverage,
        };
      }

      return null;
    } catch (error) {
      logger.warn('Failed to get market data', { venue, market, error });
      return null;
    }
  }

  /**
   * Calculate OI change percentage
   */
  private calculateOiChange(market: string, currentOi: number): number {
    const cacheKey = `${market}`;
    const cached = this.oiCache.get(cacheKey);
    const previousOi = this.previousOiCache.get(cacheKey);

    // Update cache
    if (!cached || Date.now() - cached.timestamp > 3600000) { // 1 hour
      if (cached) {
        this.previousOiCache.set(cacheKey, cached.oi);
      }
      this.oiCache.set(cacheKey, { oi: currentOi, timestamp: Date.now() });
    }

    if (!previousOi) return 0;
    return ((currentOi - previousOi) / previousOi) * 100;
  }

  /**
   * Calculate risk score (1-10, lower is safer)
   */
  private calculateRiskScore(
    marketData: { openInterest: number; oiChangePct24h: number; liquidityDepth: number },
    rate: FundingRate
  ): number {
    let score = 5; // Base score

    // Lower score for high OI (more liquid)
    if (marketData.openInterest > 10_000_000) score -= 1;
    if (marketData.openInterest > 50_000_000) score -= 1;

    // Higher score for rapid OI changes (crowded trade risk)
    if (Math.abs(marketData.oiChangePct24h) > 30) score += 1;
    if (Math.abs(marketData.oiChangePct24h) > 50) score += 1;

    // Higher score for extreme funding (mean reversion risk)
    const rateBps = Math.abs(rate.rate) * 10000;
    if (rateBps > 50) score += 1;  // >0.5% hourly is extreme
    if (rateBps > 100) score += 1; // >1% hourly is very extreme

    // Clamp to 1-10
    return Math.max(1, Math.min(10, score));
  }

  /**
   * Determine confidence level for single-venue opportunity
   */
  private determineConfidence(
    rateBps: number,
    marketData: { openInterest: number; oiChangePct24h: number },
    riskScore: number
  ): 'high' | 'medium' | 'low' {
    if (rateBps >= 10 && marketData.openInterest > 5_000_000 && riskScore <= 4) {
      return 'high';
    }
    if (rateBps < 5 || marketData.openInterest < 2_000_000 || riskScore >= 7) {
      return 'low';
    }
    return 'medium';
  }

  /**
   * Determine confidence for arbitrage opportunity
   */
  private determineArbConfidence(spreadBps: number): 'high' | 'medium' | 'low' {
    if (spreadBps >= 20) return 'high';
    if (spreadBps >= 10) return 'medium';
    return 'low';
  }

  /**
   * Get scanner configuration
   */
  getConfig(): PerpsScannnerConfig {
    return { ...this.config };
  }
}

// ============= SINGLETON =============

let scannerInstance: PerpsScanner | null = null;

export function getPerpsScanner(config?: Partial<PerpsScannnerConfig>): PerpsScanner {
  if (!scannerInstance) {
    scannerInstance = new PerpsScanner(config);
  }
  return scannerInstance;
}

export function resetPerpsScanner(): void {
  scannerInstance = null;
}
