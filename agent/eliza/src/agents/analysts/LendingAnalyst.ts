/**
 * LendingAnalyst - Lending Protocol Analyst
 * 
 * Evaluates lending opportunities with ML-enhanced predictions
 * Follows the same pattern as LPAnalyst and other analysts
 */

import { BaseAnalyst, DEFAULT_ANALYST_CONFIG, type AnalystConfig } from './BaseAnalyst.js';
import type { LendingMarketData } from '../../services/lending/types.js';
import { RiskManager } from '../../services/riskManager.js';
import { getLendingModelLoader, createFeatureExtractor, type PredictionResult } from '../../services/lending/ml/index.js';
import { logger } from '../../services/logger.js';

/**
 * Input for LendingAnalyst
 */
export interface LendingAnalysisInput {
  markets: LendingMarketData[];
  volatility24h: number;
  portfolioValueUsd?: number;
}

/**
 * Output from LendingAnalyst
 */
export interface LendingOpportunityResult {
  type: 'lending';
  name: string;
  expectedReturn: number;
  riskScore: number;
  confidence: number;
  riskAdjustedReturn: number;
  approved: boolean;
  rejectReason?: string;
  warnings: string[];
  raw: LendingMarketData;
  mlPrediction?: PredictionResult;
}

/**
 * Configuration for LendingAnalyst
 */
export interface LendingAnalystConfig extends AnalystConfig {
  minNetApy: number;              // Minimum net APY (2%)
  maxUtilization: number;         // Maximum utilization rate (85%)
  minTvl: number;                 // Minimum TVL in USD
  maxTier: number;                // Maximum asset tier (1=best, 3=worst)
  minConfidenceThreshold: number; // Minimum ML confidence (60%)
}

export const DEFAULT_LENDING_CONFIG: LendingAnalystConfig = {
  ...DEFAULT_ANALYST_CONFIG,
  minConfidence: 0.50,            // 50% min confidence
  minNetApy: 0.02,                // 2% min net APY
  maxUtilization: 0.85,           // 85% max utilization
  minTvl: 50_000_000,             // $50M min TVL
  maxTier: 2,                     // Tier 1-2 only
  minConfidenceThreshold: 0.50,   // 50% threshold
};

/**
 * LendingAnalyst - ML-Enhanced Lending Evaluator
 */
export class LendingAnalyst extends BaseAnalyst<LendingAnalysisInput, LendingOpportunityResult> {
  private riskManager: RiskManager;
  private lendingConfig: LendingAnalystConfig;
  private modelLoader: ReturnType<typeof getLendingModelLoader>;
  private featureExtractor: ReturnType<typeof createFeatureExtractor>;
  private initialized = false;

  constructor(config: Partial<LendingAnalystConfig> = {}) {
    super({ ...DEFAULT_LENDING_CONFIG, ...config });
    this.lendingConfig = { ...DEFAULT_LENDING_CONFIG, ...config };
    
    // Initialize RiskManager
    this.riskManager = new RiskManager({
      maxPositionPct: 0.30,  // 30% max position for lending
      maxDailyLossPct: 0.05,
      dataQualityScore: 0.85,  // High quality on-chain data
    });

    // Initialize ML components
    this.modelLoader = getLendingModelLoader({
      minConfidence: this.lendingConfig.minConfidenceThreshold,
      minNetApy: this.lendingConfig.minNetApy,
    });
    this.featureExtractor = createFeatureExtractor();

    logger.info(`ℹ️ [${new Date().toISOString()}] [AGENT] [LendingAnalyst] Initialized`, {
      config: this.lendingConfig,
    });
  }

  getName(): string {
    return 'LendingAnalyst';
  }

  /**
   * Initialize ML model (call before analyze)
   */
  async initialize(): Promise<boolean> {
    if (this.initialized) return true;
    
    try {
      await this.modelLoader.initialize();
      this.initialized = true;
      logger.info(`ℹ️ [${new Date().toISOString()}] [AGENT] [LendingAnalyst] ML model loaded`);
      return true;
    } catch (error) {
      logger.error(`❌ [${new Date().toISOString()}] [AGENT] [LendingAnalyst] Failed to load ML model`, { error: String(error) });
      return false;
    }
  }

  /**
   * Analyze lending opportunities
   */
  async analyze(input: LendingAnalysisInput): Promise<LendingOpportunityResult[]> {
    // Ensure ML model is initialized
    if (!this.initialized) {
      await this.initialize();
    }

    const results: LendingOpportunityResult[] = [];
    const volatility = input.volatility24h ?? this.config.volatility24h;

    for (const market of input.markets) {
      const result = await this.evaluateLendingMarket(market, volatility);
      results.push(result);
    }

    // Sort by risk-adjusted return
    results.sort((a, b) => b.riskAdjustedReturn - a.riskAdjustedReturn);

    if (this.config.verbose) {
      const approved = results.filter(r => r.approved);
      logger.info(`ℹ️ [${new Date().toISOString()}] [AGENT] [LendingAnalyst] Analyzed ${results.length} markets, ${approved.length} approved`);
    }

    return results;
  }

  /**
   * Evaluate a single lending market with ML
   */
  private async evaluateLendingMarket(
    market: LendingMarketData,
    volatility: number
  ): Promise<LendingOpportunityResult> {
    const warnings: string[] = [];
    let approved = true;
    let rejectReason: string | undefined;

    // Extract features and run ML prediction
    const features = this.featureExtractor.extractFeatures(market);
    const netApy = this.featureExtractor.calculateNetApy(market);
    const mlPrediction = await this.modelLoader.predict(features, netApy);

    // Calculate expected return (net APY)
    const expectedReturn = netApy * 100;  // Convert to percentage

    // Calculate risk score (1-10)
    let riskScore = 5;  // Base risk

    // Increase risk for high utilization
    if (market.utilizationRate > 0.80) riskScore += 2;
    if (market.utilizationRate > 0.90) riskScore += 2;

    // Decrease risk for high TVL
    if (market.tvlUsd > 100_000_000) riskScore -= 1;
    if (market.tvlUsd > 200_000_000) riskScore -= 1;

    // Adjust for asset tier
    const assetTier = this.getAssetTier(market.asset);
    if (assetTier === 1) riskScore -= 1;  // Tier 1 assets (USDC, SOL) are safer
    if (assetTier === 3) riskScore += 2;  // Tier 3 assets are riskier

    riskScore = Math.max(1, Math.min(10, riskScore));  // Clamp to 1-10

    // Risk-adjusted return
    const riskAdjustedReturn = expectedReturn / riskScore;

    // Validation checks
    if (market.tvlUsd < this.lendingConfig.minTvl) {
      approved = false;
      rejectReason = `TVL too low ($${(market.tvlUsd / 1_000_000).toFixed(2)}M < $${(this.lendingConfig.minTvl / 1_000_000).toFixed(0)}M)`;
    } else if (market.utilizationRate > this.lendingConfig.maxUtilization) {
      approved = false;
      rejectReason = `Utilization too high (${(market.utilizationRate * 100).toFixed(1)}% > ${(this.lendingConfig.maxUtilization * 100).toFixed(0)}%)`;
    } else if (assetTier > this.lendingConfig.maxTier) {
      approved = false;
      rejectReason = `Asset tier too low (${assetTier} > ${this.lendingConfig.maxTier})`;
    } else if (netApy < this.lendingConfig.minNetApy) {
      approved = false;
      rejectReason = `Net APY too low (${(netApy * 100).toFixed(2)}% < ${(this.lendingConfig.minNetApy * 100).toFixed(0)}%)`;
    } else if (!mlPrediction.shouldLend) {
      approved = false;
      rejectReason = `ML model rejected (confidence: ${(mlPrediction.confidence * 100).toFixed(1)}%)`;
    }

    // Warnings
    if (market.utilizationRate > 0.75) {
      warnings.push(`High utilization: ${(market.utilizationRate * 100).toFixed(1)}%`);
    }
    if (assetTier === 3) {
      warnings.push(`Low-tier asset: ${market.asset}`);
    }
    if (netApy > 0.50) {  // 50% APY is suspiciously high
      warnings.push(`Unusually high APY: ${(netApy * 100).toFixed(2)}%`);
    }

    return {
      type: 'lending',
      name: `${market.asset} on ${market.protocol}`,
      expectedReturn,
      riskScore,
      confidence: mlPrediction.confidence,
      riskAdjustedReturn,
      approved,
      rejectReason,
      warnings,
      raw: market,
      mlPrediction,
    };
  }

  /**
   * Get asset tier (1=best, 3=worst)
   */
  private getAssetTier(asset: string): number {
    const tier1 = ['USDC', 'USDT', 'SOL'];
    const tier2 = ['JITOSOL', 'MSOL', 'PYUSD', 'USDS', 'JUPSOL'];

    if (tier1.includes(asset)) return 1;
    if (tier2.includes(asset)) return 2;
    return 3;
  }
}


