/**
 * MomentumAnalyst - Funding Rate Arbitrage Analyst
 * 
 * Evaluates funding rate arbitrage opportunities (delta-neutral strategy)
 * Extracted from TradingAgent for independent execution
 */

import { BaseAnalyst, DEFAULT_ANALYST_CONFIG, type AnalystConfig } from './BaseAnalyst.js';
import type { FundingArbitrageOpportunity } from '../../services/marketScanner/types.js';
import { getSentimentIntegration, type SentimentAdjustedScore } from '../../services/sentiment/sentimentIntegration.js';
import { RiskManager } from '../../services/riskManager.js';
import { logger } from '../../services/logger.js';

/**
 * Input for MomentumAnalyst
 */
export interface MomentumAnalysisInput {
  opportunities: FundingArbitrageOpportunity[];
  volatility24h: number;
  portfolioValueUsd?: number;
}

/**
 * Output from MomentumAnalyst
 */
export interface MomentumOpportunityResult {
  type: 'funding_arb';
  name: string;
  expectedReturn: number;
  riskScore: number;
  confidence: number;
  riskAdjustedReturn: number;
  approved: boolean;
  rejectReason?: string;
  warnings: string[];
  raw: FundingArbitrageOpportunity;
  sentimentAdjustment?: SentimentAdjustedScore;
}

/**
 * Configuration for MomentumAnalyst
 */
export interface MomentumAnalystConfig extends AnalystConfig {
  minSpreadBps: number;           // Minimum spread in basis points
  minConfidenceThreshold: number; // Minimum confidence to approve
  deltaNeutralPositionPct: number; // Position size for delta-neutral strategy
}

export const DEFAULT_MOMENTUM_CONFIG: MomentumAnalystConfig = {
  ...DEFAULT_ANALYST_CONFIG,
  minConfidence: 0.80,            // 80% min confidence
  minSpreadBps: 5,                // 5 bps minimum spread
  minConfidenceThreshold: 0.80,   // 80% threshold
  deltaNeutralPositionPct: 10,    // 10% position for delta-neutral
};

/**
 * MomentumAnalyst - Funding Rate Arbitrage Evaluator
 */
export class MomentumAnalyst extends BaseAnalyst<MomentumAnalysisInput, MomentumOpportunityResult> {
  private riskManager: RiskManager;
  private momentumConfig: MomentumAnalystConfig;

  constructor(config: Partial<MomentumAnalystConfig> = {}) {
    super({ ...DEFAULT_MOMENTUM_CONFIG, ...config });
    this.momentumConfig = { ...DEFAULT_MOMENTUM_CONFIG, ...config };
    
    // Initialize RiskManager
    this.riskManager = new RiskManager({
      maxPositionPct: 0.37,
      maxDailyLossPct: 0.05,
      dataQualityScore: 0.56,
    });

    logger.info(`ℹ️ [${new Date().toISOString()}] [AGENT] [MomentumAnalyst] Initialized`, {
      config: this.momentumConfig,
    });
  }

  getName(): string {
    return 'MomentumAnalyst';
  }

  /**
   * Analyze funding arbitrage opportunities
   */
  async analyze(input: MomentumAnalysisInput): Promise<MomentumOpportunityResult[]> {
    const results: MomentumOpportunityResult[] = [];
    const volatility = input.volatility24h ?? this.config.volatility24h;

    for (const arb of input.opportunities) {
      const result = await this.evaluateFundingArbitrage(arb, volatility);
      results.push(result);
    }

    return results;
  }

  /**
   * Evaluate single funding arbitrage opportunity
   */
  private async evaluateFundingArbitrage(
    arb: FundingArbitrageOpportunity,
    volatility: number
  ): Promise<MomentumOpportunityResult> {
    const name = `${arb.market} ${arb.longVenue}↔${arb.shortVenue}`;
    const annualizedReturn = arb.annualizedSpread * 100;

    // Confidence based on spread size
    let baseConfidence = 0.6;
    if (arb.estimatedProfitBps >= 20) baseConfidence = 0.9;
    else if (arb.estimatedProfitBps >= 10) baseConfidence = 0.8;

    // ============= SENTIMENT INTEGRATION (5% weight for funding arb - delta neutral) =============
    const token = arb.market.split('-')[0] || 'SOL';
    const sentimentIntegration = getSentimentIntegration();
    const sentimentAdjustment = await sentimentIntegration.getAdjustedScore(
      token,
      baseConfidence,
      'arbitrage' // Use arbitrage weight (5%) since it's delta-neutral
    );

    const confidence = sentimentAdjustment.finalScore;
    const warnings: string[] = [];

    if (sentimentAdjustment.sentimentAvailable) {
      warnings.push(`Sentiment: ${sentimentAdjustment.signal} (${(sentimentAdjustment.rawSentiment * 100).toFixed(0)}%)`);
    }

    // ============= DETAILED SENTIMENT LOGGING =============
    if (this.config.verbose) {
      this.logSentimentEvaluation(arb, token, baseConfidence, sentimentAdjustment, confidence);
    }

    // Risk score for delta-neutral (lower risk)
    const riskScore = arb.estimatedProfitBps >= 20 ? 3 : 4;

    // Risk check
    const riskCheck = this.riskManager.checkTradeAllowed({
      proposedPositionPct: this.momentumConfig.deltaNeutralPositionPct,
      currentVolatility24h: volatility,
      expectedReturnPct: annualizedReturn / 365, // Daily
    });

    const riskAdjustedReturn = (annualizedReturn / 365) * (1 - riskScore / 20);

    let rejectReason: string | undefined;
    if (!riskCheck.allowed) {
      rejectReason = riskCheck.reason;
    } else if (arb.estimatedProfitBps < this.momentumConfig.minSpreadBps) {
      rejectReason = `Spread too low: ${arb.estimatedProfitBps.toFixed(1)}bps (min ${this.momentumConfig.minSpreadBps}bps)`;
    }

    return {
      type: 'funding_arb',
      name,
      expectedReturn: annualizedReturn,
      riskScore,
      confidence,
      riskAdjustedReturn,
      approved: !rejectReason && confidence >= this.momentumConfig.minConfidenceThreshold,
      rejectReason,
      warnings: [...warnings, ...riskCheck.warnings],
      raw: arb,
      sentimentAdjustment,
    };
  }

  /**
   * Log sentiment evaluation details
   */
  private logSentimentEvaluation(
    arb: FundingArbitrageOpportunity,
    token: string,
    baseConfidence: number,
    sentimentAdjustment: any,
    confidence: number
  ): void {
    const mlPassesThreshold = baseConfidence >= this.momentumConfig.minConfidenceThreshold;
    const finalPassesThreshold = confidence >= this.momentumConfig.minConfidenceThreshold;
    const delta = confidence - baseConfidence;

    logger.info(`\n┌─────────────────────────────────────────────────────────┐`);
    logger.info(`│  🔄 FUNDING ARB OPPORTUNITY EVALUATION WITH SENTIMENT   │`);
    logger.info(`├─────────────────────────────────────────────────────────┤`);
    logger.info(`│  Market:              ${arb.market.padEnd(35)}│`);
    logger.info(`│  Token:               ${token.padEnd(35)}│`);
    logger.info(`│  Route:               ${(arb.longVenue + '↔' + arb.shortVenue).padEnd(35)}│`);
    logger.info(`│  Spread:              ${(arb.estimatedProfitBps.toFixed(1) + 'bps').padEnd(35)}│`);
    logger.info(`├─────────────────────────────────────────────────────────┤`);
    logger.info(`│  ML Confidence:       ${baseConfidence.toFixed(3).padEnd(35)}│`);
    if (sentimentAdjustment.sentimentAvailable) {
      logger.info(`│  Sentiment Score:     ${sentimentAdjustment.rawSentiment.toFixed(3)} (${sentimentAdjustment.signal})`.padEnd(60) + `│`);
      logger.info(`│  Normalized:          ${sentimentAdjustment.normalizedSentiment.toFixed(3).padEnd(35)}│`);
      logger.info(`│  Sentiment Weight:    ${'5% (delta-neutral)'.padEnd(35)}│`);
      logger.info(`│  Final Confidence:    ${confidence.toFixed(3).padEnd(35)}│`);
      logger.info(`│  Delta:               ${(delta >= 0 ? '+' : '') + delta.toFixed(3)}`.padEnd(60) + `│`);
    } else {
      logger.info(`│  Sentiment:           ${'unavailable (using ML only)'.padEnd(35)}│`);
    }
    logger.info(`├─────────────────────────────────────────────────────────┤`);
    logger.info(`│  Threshold:           ${this.momentumConfig.minConfidenceThreshold.toFixed(3).padEnd(35)}│`);
    logger.info(`│  Decision:            ${(finalPassesThreshold ? 'TRADE ✅' : 'SKIP ❌').padEnd(35)}│`);
    if (mlPassesThreshold && !finalPassesThreshold) {
      logger.info(`│  ⚠️  SENTIMENT PREVENTED TRADE                          │`);
    }
    if (!mlPassesThreshold && finalPassesThreshold) {
      logger.info(`│  ✨ SENTIMENT ENABLED TRADE                             │`);
    }
    logger.info(`└─────────────────────────────────────────────────────────┘`);
  }

  /**
   * Update momentum-specific configuration
   */
  updateMomentumConfig(config: Partial<MomentumAnalystConfig>): void {
    this.momentumConfig = { ...this.momentumConfig, ...config };
    this.updateConfig(config);
  }

  /**
   * Get momentum-specific configuration
   */
  getMomentumConfig(): MomentumAnalystConfig {
    return { ...this.momentumConfig };
  }
}
