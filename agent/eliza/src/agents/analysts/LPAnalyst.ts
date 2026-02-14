/**
 * LPAnalyst - Liquidity Pool Rebalancing Analyst
 * 
 * Evaluates LP pool opportunities with ML-enhanced predictions
 * Extracted from TradingAgent for independent execution
 */

import { BaseAnalyst, DEFAULT_ANALYST_CONFIG, type AnalystConfig } from './BaseAnalyst.js';
import type { LPPool } from '../../services/marketScanner/types.js';
import { getSentimentIntegration, type SentimentAdjustedScore } from '../../services/sentiment/sentimentIntegration.js';
import { RiskManager } from '../../services/riskManager.js';
import { lpRebalancerModel, type PredictionResult } from '../../inference/model.js';
import { logger } from '../../services/logger.js';

/**
 * Input for LPAnalyst
 */
export interface LPAnalysisInput {
  pools: LPPool[];
  volatility24h: number;
  portfolioValueUsd?: number;
  mlPredictions?: Map<string, PredictionResult>; // Pool address -> ML prediction
}

/**
 * Output from LPAnalyst
 */
export interface LPOpportunityResult {
  type: 'lp';
  name: string;
  expectedReturn: number;
  riskScore: number;
  confidence: number;
  riskAdjustedReturn: number;
  approved: boolean;
  rejectReason?: string;
  warnings: string[];
  raw: LPPool;
  sentimentAdjustment?: SentimentAdjustedScore;
  mlPrediction?: PredictionResult;
}

/**
 * Configuration for LPAnalyst
 */
export interface LPAnalystConfig extends AnalystConfig {
  maxApy: number;              // Maximum APY (scam filter)
  minTvl: number;              // Minimum TVL in USD
  minVolumeTvlRatio: number;   // Minimum volume/TVL ratio
  allowedTokens: string[];     // Whitelist of allowed tokens
  minConfidenceThreshold: number; // Minimum confidence to approve (84%)
}

export const DEFAULT_LP_CONFIG: LPAnalystConfig = {
  ...DEFAULT_ANALYST_CONFIG,
  minConfidence: 0.50,            // 50% min confidence
  maxApy: 500,                    // 500% max APY (scam filter)
  minTvl: 100_000,                // $100k min TVL
  minVolumeTvlRatio: 0.3,         // 0.3 min volume/TVL ratio
  allowedTokens: ['SOL', 'USDC', 'USDT', 'JUP', 'BONK', 'mSOL', 'stSOL', 'jitoSOL', 'RAY', 'ORCA'],
  minConfidenceThreshold: 0.50,   // 50% threshold
};

/**
 * LPAnalyst - ML-Enhanced LP Pool Evaluator
 */
export class LPAnalyst extends BaseAnalyst<LPAnalysisInput, LPOpportunityResult> {
  private riskManager: RiskManager;
  private lpConfig: LPAnalystConfig;

  constructor(config: Partial<LPAnalystConfig> = {}) {
    super({ ...DEFAULT_LP_CONFIG, ...config });
    this.lpConfig = { ...DEFAULT_LP_CONFIG, ...config };
    
    // Initialize RiskManager
    this.riskManager = new RiskManager({
      maxPositionPct: 0.037,  // 3.7% Half-Kelly position sizing (production)
      maxDailyLossPct: 0.05,
      dataQualityScore: 0.56,
    });

    logger.info(`ℹ️ [${new Date().toISOString()}] [AGENT] [LPAnalyst] Initialized`, {
      config: this.lpConfig,
    });
  }

  getName(): string {
    return 'LPAnalyst';
  }

  /**
   * Analyze LP pool opportunities
   */
  async analyze(input: LPAnalysisInput): Promise<LPOpportunityResult[]> {
    const results: LPOpportunityResult[] = [];
    const volatility = input.volatility24h ?? this.config.volatility24h;

    for (const pool of input.pools) {
      const mlPrediction = input.mlPredictions?.get(pool.address);
      const result = await this.evaluateLPPoolWithML(pool, volatility, mlPrediction);
      results.push(result);
    }

    return results;
  }

  /**
   * Evaluate single LP pool with ML prediction
   */
  private async evaluateLPPoolWithML(
    pool: LPPool,
    volatility: number,
    mlPrediction?: PredictionResult
  ): Promise<LPOpportunityResult> {
    const name = `${pool.name} [${pool.dex}]`;

    // ============= SCAM FILTER =============
    // 1. MAX APY CHECK
    if (pool.apy > this.lpConfig.maxApy) {
      return this.rejectPool(pool, name, `APY too high (${pool.apy.toFixed(0)}% > ${this.lpConfig.maxApy}%)`);
    }

    // 2. MIN TVL CHECK
    if (pool.tvl < this.lpConfig.minTvl) {
      return this.rejectPool(pool, name, `TVL too low ($${(pool.tvl/1_000_000).toFixed(1)}M < $${(this.lpConfig.minTvl/1_000_000).toFixed(1)}M)`);
    }

    // 3. VOLUME/TVL RATIO
    const volumeTvlRatio = pool.volume24h / pool.tvl;
    if (volumeTvlRatio < this.lpConfig.minVolumeTvlRatio) {
      return this.rejectPool(pool, name, `Volume/TVL too low (${volumeTvlRatio.toFixed(2)} < ${this.lpConfig.minVolumeTvlRatio})`);
    }

    // 4. ALLOWED TOKENS
    const tokens = pool.name.split('/').map(t => t.trim().toUpperCase());
    const hasUnknownToken = tokens.some(t => !this.lpConfig.allowedTokens.includes(t));
    if (hasUnknownToken) {
      return this.rejectPool(pool, name, `Unknown token in pair`);
    }

    // ============= ML-ENHANCED CONFIDENCE =============
    let mlConfidence: number;
    const warnings: string[] = [];

    if (mlPrediction) {
      // Use ML model confidence (84% accuracy)
      mlConfidence = mlPrediction.confidence;

      // If ML says HOLD with high confidence, reduce our confidence
      if (mlPrediction.decision === 'HOLD' && mlPrediction.probability < 0.3) {
        mlConfidence = 1 - mlPrediction.confidence; // Invert
        warnings.push(`ML: HOLD (${(mlPrediction.probability * 100).toFixed(1)}%)`);
      } else if (mlPrediction.decision === 'REBALANCE') {
        warnings.push(`ML: REBALANCE (${(mlPrediction.probability * 100).toFixed(1)}%)`);
      }
    } else {
      // Fallback to heuristic confidence
      const tvlConf = pool.tvl > 50_000_000 ? 0.95 : pool.tvl > 20_000_000 ? 0.9 : 0.8;
      const volConf = volumeTvlRatio > 1 ? 0.95 : volumeTvlRatio > 0.5 ? 0.9 : 0.8;
      mlConfidence = (tvlConf + volConf) / 2;
      warnings.push('ML: unavailable (using heuristics)');
    }

    // ============= SENTIMENT INTEGRATION (15% weight for LP) =============
    // Use first token in pair for sentiment (e.g., SOL from SOL/USDC)
    const primaryToken = tokens[0] || 'SOL';
    const sentimentIntegration = getSentimentIntegration();
    const sentimentAdjustment = await sentimentIntegration.getAdjustedScore(
      primaryToken,
      mlConfidence,
      'lp'
    );

    // Use sentiment-adjusted confidence
    const confidence = sentimentAdjustment.finalScore;

    if (sentimentAdjustment.sentimentAvailable) {
      warnings.push(`Sentiment: ${sentimentAdjustment.signal} (${(sentimentAdjustment.rawSentiment * 100).toFixed(0)}%)`);
    }

    // ============= DETAILED SENTIMENT LOGGING =============
    if (this.config.verbose) {
      this.logSentimentEvaluation(pool, name, primaryToken, mlConfidence, sentimentAdjustment, confidence);
    }

    // ============= RISK SCORING =============
    let riskLevel: 'low' | 'medium' | 'high';
    let riskScore: number;

    if (pool.tvl > 50_000_000 && pool.apy < 50) {
      riskLevel = 'low';
      riskScore = 2;
    } else if (pool.tvl >= 20_000_000 && pool.apy < 100) {
      riskLevel = 'low';
      riskScore = 3;
    } else if (pool.tvl >= 5_000_000 && pool.apy <= 200) {
      riskLevel = 'medium';
      riskScore = pool.apy > 100 ? 6 : 4;
    } else {
      riskLevel = 'high';
      riskScore = 8;
    }

    const dailyReturn = pool.apy / 365;

    const positionCalc = this.riskManager.calculatePositionSize({
      modelConfidence: confidence,
      currentVolatility24h: volatility,
      portfolioValueUsd: this.config.portfolioValueUsd,
    });

    const riskCheck = this.riskManager.checkTradeAllowed({
      proposedPositionPct: positionCalc.positionPct,
      currentVolatility24h: volatility,
      expectedReturnPct: dailyReturn,
    });

    const riskAdjustedReturn = dailyReturn * (1 - riskScore / 20);

    warnings.push(`Risk: ${riskLevel}`);

    return {
      type: 'lp',
      name,
      expectedReturn: pool.apy,
      riskScore,
      confidence,
      riskAdjustedReturn,
      approved: riskCheck.allowed && confidence >= this.lpConfig.minConfidenceThreshold,
      rejectReason: !riskCheck.allowed ? riskCheck.reason : undefined,
      warnings,
      raw: pool,
      sentimentAdjustment,
      mlPrediction,
    };
  }

  /**
   * Helper to create rejected pool evaluation
   */
  private rejectPool(pool: LPPool, name: string, reason: string): LPOpportunityResult {
    return {
      type: 'lp',
      name,
      expectedReturn: pool.apy,
      riskScore: 10,
      confidence: 0,
      riskAdjustedReturn: 0,
      approved: false,
      rejectReason: reason,
      warnings: [],
      raw: pool,
    };
  }

  /**
   * Log sentiment evaluation details
   */
  private logSentimentEvaluation(
    pool: LPPool,
    name: string,
    primaryToken: string,
    mlConfidence: number,
    sentimentAdjustment: SentimentAdjustedScore,
    confidence: number
  ): void {
    const mlPassesThreshold = mlConfidence >= this.lpConfig.minConfidenceThreshold;
    const finalPassesThreshold = confidence >= this.lpConfig.minConfidenceThreshold;
    const delta = confidence - mlConfidence;

    logger.info(`\n┌─────────────────────────────────────────────────────────┐`);
    logger.info(`│  💧 LP POOL OPPORTUNITY EVALUATION WITH SENTIMENT       │`);
    logger.info(`├─────────────────────────────────────────────────────────┤`);
    logger.info(`│  Pool:                ${name.substring(0, 35).padEnd(35)}│`);
    logger.info(`│  Token:               ${primaryToken.padEnd(35)}│`);
    logger.info(`│  APY:                 ${(pool.apy.toFixed(2) + '%').padEnd(35)}│`);
    logger.info(`│  TVL:                 ${'$' + (pool.tvl / 1_000_000).toFixed(2) + 'M'.padEnd(33)}│`);
    logger.info(`├─────────────────────────────────────────────────────────┤`);
    logger.info(`│  ML Confidence:       ${mlConfidence.toFixed(3).padEnd(35)}│`);
    if (sentimentAdjustment.sentimentAvailable) {
      logger.info(`│  Sentiment Score:     ${sentimentAdjustment.rawSentiment.toFixed(3)} (${sentimentAdjustment.signal})`.padEnd(60) + `│`);
      logger.info(`│  Normalized:          ${sentimentAdjustment.normalizedSentiment.toFixed(3).padEnd(35)}│`);
      logger.info(`│  Sentiment Weight:    ${'15%'.padEnd(35)}│`);
      logger.info(`│  Final Confidence:    ${confidence.toFixed(3).padEnd(35)}│`);
      logger.info(`│  Delta:               ${(delta >= 0 ? '+' : '') + delta.toFixed(3)}`.padEnd(60) + `│`);
    } else {
      logger.info(`│  Sentiment:           ${'unavailable (using ML only)'.padEnd(35)}│`);
    }
    logger.info(`├─────────────────────────────────────────────────────────┤`);
    logger.info(`│  Threshold:           ${this.lpConfig.minConfidenceThreshold.toFixed(3).padEnd(35)}│`);
    logger.info(`│  Decision:            ${(finalPassesThreshold ? 'PROVIDE LP ✅' : 'SKIP ❌').padEnd(35)}│`);
    if (mlPassesThreshold && !finalPassesThreshold) {
      logger.info(`│  ⚠️  SENTIMENT PREVENTED LP ENTRY                       │`);
    }
    if (!mlPassesThreshold && finalPassesThreshold) {
      logger.info(`│  ✨ SENTIMENT ENABLED LP ENTRY                          │`);
    }
    logger.info(`└─────────────────────────────────────────────────────────┘`);
  }

  /**
   * Update LP-specific configuration
   */
  updateLPConfig(config: Partial<LPAnalystConfig>): void {
    this.lpConfig = { ...this.lpConfig, ...config };
    this.updateConfig(config);
  }

  /**
   * Get LP-specific configuration
   */
  getLPConfig(): LPAnalystConfig {
    return { ...this.lpConfig };
  }
}

