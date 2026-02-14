/**
 * ArbitrageAnalyst
 *
 * Standalone analyst for evaluating arbitrage opportunities.
 * Extracted from TradingAgent.evaluateArbitrage() to run independently.
 *
 * Capabilities:
 * - ML-powered profitability prediction (ONNX model)
 * - Cross-DEX spread calculation
 * - Gas cost analysis
 * - Slippage estimation
 * - MEV risk scoring
 * - Sentiment integration (5% weight for arbitrage)
 */

import { BaseAnalyst, type Opportunity, type AnalystConfig, DEFAULT_ANALYST_CONFIG } from './BaseAnalyst.js';
import { RiskManager, type RiskCheckResult } from '../../services/riskManager.js';
import { getSentimentIntegration, type SentimentAdjustedScore } from '../../services/sentiment/sentimentIntegration.js';
import type { ArbitrageOpportunity } from '../../services/marketScanner/types.js';
import { logger } from '../../services/logger.js';
import { ArbitrageFeatureExtractor, arbitrageModelLoader } from '../../services/arbitrage/index.js';

// ============= TYPES =============

/**
 * Input data for arbitrage analysis
 */
export interface ArbitrageAnalysisInput {
  /** List of raw arbitrage opportunities from market scanner */
  opportunities: ArbitrageOpportunity[];
  /** Current 24h volatility (overrides config if provided) */
  volatility24h?: number;
}

/**
 * Enhanced opportunity result specific to arbitrage
 */
export interface ArbitrageOpportunityResult extends Opportunity {
  type: 'arbitrage';
  raw: ArbitrageOpportunity;
  /** Sentiment adjustment details */
  sentimentAdjustment?: SentimentAdjustedScore;
  /** Trading direction analysis */
  direction: {
    isCexToDex: boolean;
    isDexToCex: boolean;
    isDexToDex: boolean;
    isCexToCex: boolean;
  };
}

/**
 * Configuration specific to arbitrage analysis
 *
 * Based on defined risk parameters:
 * - minSpread: 0.5% minimum spread
 * - maxSlippage: 0.5% max slippage
 * - minConfidence: 80% min confidence
 * - minProfitAfterGas: 0.5% min profit after gas
 * - dexToDex: enabled
 * - dexToCex: DISABLED (deposit latency risk)
 * - useJitoBundles: true (MEV protection)
 */
export interface ArbitrageAnalystConfig extends AnalystConfig {
  /** Minimum spread percentage for approval (0.5% = 0.005) */
  minSpreadPct: number;
  /** Maximum spread percentage (above this is suspicious) */
  maxSpreadPct: number;
  /** Maximum allowed slippage percentage */
  maxSlippagePct: number;
  /** Minimum profit percentage after gas (not flat USD) */
  minProfitAfterGasPct: number;
  /** CEX exchanges for direction detection */
  cexExchanges: string[];
  /** Direction rules */
  allowDexToDex: boolean;
  allowDexToCex: boolean;
  allowCexToDex: boolean;
  allowCexToCex: boolean;
  /** Use Jito bundles for MEV protection */
  useJitoBundles: boolean;
  /** Maximum concurrent positions */
  maxConcurrentPositions: number;
}

export const DEFAULT_ARBITRAGE_CONFIG: ArbitrageAnalystConfig = {
  ...DEFAULT_ANALYST_CONFIG,
  minConfidence: 0.60,            // 60% minimum ML confidence
  minSpreadPct: 0.10,             // 10 bps minimum spread
  maxSpreadPct: 10.0,             // 10% max (scam filter)
  maxSlippagePct: 2.0,            // 2% max slippage
  minProfitAfterGasPct: 0.05,     // 5 bps min profit after gas
  cexExchanges: ['binance', 'coinbase', 'kraken'],
  allowDexToDex: true,
  allowDexToCex: true,
  allowCexToDex: false,           // Requires CEX balance
  allowCexToCex: false,           // Requires cross-exchange transfer
  useJitoBundles: true,
  maxConcurrentPositions: 5,
};

// ============= ANALYST CLASS =============

export class ArbitrageAnalyst extends BaseAnalyst<ArbitrageAnalysisInput, ArbitrageOpportunityResult> {
  private riskManager: RiskManager;
  private arbConfig: ArbitrageAnalystConfig;
  private featureExtractor: ArbitrageFeatureExtractor;
  private mlInitialized = false;

  constructor(config: Partial<ArbitrageAnalystConfig> = {}) {
    const mergedConfig = { ...DEFAULT_ARBITRAGE_CONFIG, ...config };
    super(mergedConfig);
    this.arbConfig = mergedConfig;
    this.riskManager = new RiskManager();
    this.featureExtractor = new ArbitrageFeatureExtractor();
    logger.info('[ArbitrageAnalyst] Initialized', { config: this.arbConfig });
  }

  /**
   * Initialize ML model (lazy loading)
   */
  private async initializeML(): Promise<boolean> {
    if (this.mlInitialized) return true;

    try {
      const success = await arbitrageModelLoader.initialize();
      if (success) {
        this.mlInitialized = true;
        logger.info('[ArbitrageAnalyst] ML model initialized');
      }
      return success;
    } catch (error) {
      logger.warn('[ArbitrageAnalyst] ML model initialization failed, using fallback', { error });
      return false;
    }
  }

  getName(): string {
    return 'ArbitrageAnalyst';
  }

  /**
   * Analyze arbitrage opportunities from market data
   */
  async analyze(input: ArbitrageAnalysisInput): Promise<ArbitrageOpportunityResult[]> {
    // Initialize ML model (lazy loading)
    await this.initializeML();

    const volatility = input.volatility24h ?? this.arbConfig.volatility24h;
    const results: ArbitrageOpportunityResult[] = [];

    if (this.arbConfig.verbose) {
      logger.info(`[ArbitrageAnalyst] Evaluating ${input.opportunities.length} opportunities`, {
        mlEnabled: this.mlInitialized
      });
    }

    for (const arb of input.opportunities) {
      const result = await this.evaluateArbitrage(arb, volatility);
      results.push(result);
    }

    const approved = results.filter((r) => r.approved);
    if (this.arbConfig.verbose) {
      logger.info(`[ArbitrageAnalyst] Results: ${approved.length}/${results.length} approved`);
    }

    return results;
  }

  /**
   * Get ML confidence using ONNX model inference
   */
  private async getMLConfidence(arb: ArbitrageOpportunity): Promise<number> {
    if (!this.mlInitialized || !arbitrageModelLoader.isInitialized()) {
      // Fallback to rule-based if ML not available
      const confidenceMap: Record<string, number> = { high: 0.9, medium: 0.7, low: 0.5 };
      return confidenceMap[arb.confidence] || 0.5;
    }

    try {
      // Extract features from opportunity
      const features = await this.featureExtractor.extractFeatures(arb);

      // Run ML inference
      const prediction = await arbitrageModelLoader.predict(
        features,
        this.arbConfig.minConfidence,
        `arb_${arb.symbol}_${Date.now()}`
      );

      if (this.arbConfig.verbose) {
        logger.debug('[ArbitrageAnalyst] ML prediction', {
          symbol: arb.symbol,
          probability: prediction.probability,
          isProfitable: prediction.isProfitable
        });
      }

      return prediction.probability;
    } catch (error) {
      logger.warn('[ArbitrageAnalyst] ML inference failed, using fallback', {
        symbol: arb.symbol,
        error
      });
      // Fallback to rule-based
      const confidenceMap: Record<string, number> = { high: 0.9, medium: 0.7, low: 0.5 };
      return confidenceMap[arb.confidence] || 0.5;
    }
  }

  /**
   * Evaluate a single arbitrage opportunity
   * Uses ML model for confidence scoring instead of rule-based mapping
   */
  private async evaluateArbitrage(
    arb: ArbitrageOpportunity,
    volatility: number
  ): Promise<ArbitrageOpportunityResult> {
    // Direction analysis
    const direction = this.analyzeDirection(arb);

    // Get ML-based confidence (replaces rule-based mapping)
    const baseConfidence = await this.getMLConfidence(arb);

    // Sentiment integration (5% weight for arbitrage)
    const sentimentIntegration = getSentimentIntegration();
    const sentimentAdjustment = await sentimentIntegration.getAdjustedScore(
      arb.symbol,
      baseConfidence,
      'arbitrage'
    );

    const confidence = sentimentAdjustment.finalScore;
    const warnings: string[] = sentimentAdjustment.sentimentAvailable
      ? [`Sentiment: ${sentimentAdjustment.signal} (${(sentimentAdjustment.rawSentiment * 100).toFixed(0)}%)`]
      : [];

    // Log sentiment evaluation
    this.logSentimentEvaluation(arb, baseConfidence, sentimentAdjustment);

    // Risk check
    const riskCheck = this.riskManager.checkTradeAllowed({
      proposedPositionPct: 2.0,
      currentVolatility24h: volatility,
      expectedReturnPct: arb.spreadPct,
    });

    // Risk score based on spread and direction
    let riskScore = arb.spreadPct > 2 ? 7 : arb.spreadPct > 1 ? 5 : 3;
    if (direction.isCexToDex) riskScore -= 1; // CEX→DEX is safer

    // Risk-adjusted return
    const riskAdjustedReturn = arb.netProfit > 0 ? arb.spreadPct * (1 - riskScore / 20) : 0;

    // Calculate profit percentage (netProfit / estimated trade value)
    // Estimate trade value from spread: if spread is 1.5% and netProfit is $15, tradeValue ≈ $1000
    const estimatedTradeValue = arb.spreadPct > 0 ? (arb.netProfit / arb.spreadPct) * 100 : 0;
    const profitAfterGasPct = estimatedTradeValue > 0 ? (arb.netProfit / estimatedTradeValue) * 100 : 0;

    // Determine rejection reason
    const rejectReason = this.determineRejectReason(arb, direction, riskCheck, profitAfterGasPct);

    // Check if direction is allowed based on config
    const directionAllowed = this.isDirectionAllowed(direction);

    const approved =
      directionAllowed &&
      riskCheck.allowed &&
      arb.spreadPct >= this.arbConfig.minSpreadPct &&
      arb.spreadPct <= this.arbConfig.maxSpreadPct &&
      profitAfterGasPct >= this.arbConfig.minProfitAfterGasPct &&
      confidence >= this.arbConfig.minConfidence;

    return {
      type: 'arbitrage',
      name: `${arb.symbol} ${arb.buyExchange}→${arb.sellExchange}`,
      expectedReturn: arb.spreadPct,
      riskScore,
      confidence,
      riskAdjustedReturn,
      approved,
      rejectReason,
      warnings: [...warnings, ...riskCheck.warnings],
      raw: arb,
      sentimentAdjustment,
      direction,
    };
  }

  /**
   * Analyze trade direction (CEX→DEX, DEX→CEX, etc.)
   */
  private analyzeDirection(arb: ArbitrageOpportunity): ArbitrageOpportunityResult['direction'] {
    const cexExchanges = this.arbConfig.cexExchanges;
    const isCexBuy = cexExchanges.includes(arb.buyExchange.toLowerCase());
    const isCexSell = cexExchanges.includes(arb.sellExchange.toLowerCase());

    return {
      isCexToDex: isCexBuy && !isCexSell,
      isDexToCex: !isCexBuy && isCexSell,
      isDexToDex: !isCexBuy && !isCexSell,
      isCexToCex: isCexBuy && isCexSell,
    };
  }

  /**
   * Check if a direction is allowed based on config
   */
  private isDirectionAllowed(direction: ArbitrageOpportunityResult['direction']): boolean {
    if (direction.isCexToDex) return this.arbConfig.allowCexToDex;
    if (direction.isDexToCex) return this.arbConfig.allowDexToCex;
    if (direction.isDexToDex) return this.arbConfig.allowDexToDex;
    if (direction.isCexToCex) return this.arbConfig.allowCexToCex;
    return false;
  }

  /**
   * Determine rejection reason based on checks
   */
  private determineRejectReason(
    arb: ArbitrageOpportunity,
    direction: ArbitrageOpportunityResult['direction'],
    riskCheck: RiskCheckResult,
    profitAfterGasPct: number
  ): string | undefined {
    // Direction checks based on config
    if (!this.isDirectionAllowed(direction)) {
      if (direction.isDexToCex) {
        return 'DEX→CEX disabled (deposit latency risk)';
      }
      if (direction.isDexToDex && !this.arbConfig.allowDexToDex) {
        return 'DEX→DEX disabled';
      }
      if (direction.isCexToCex) {
        return 'CEX→CEX disabled (transfer required)';
      }
      if (direction.isCexToDex && !this.arbConfig.allowCexToDex) {
        return 'CEX→DEX disabled';
      }
    }

    // Risk check
    if (!riskCheck.allowed) {
      return riskCheck.reason;
    }

    // Spread checks
    if (arb.spreadPct < this.arbConfig.minSpreadPct) {
      return `Spread too low: ${arb.spreadPct.toFixed(2)}% (min ${this.arbConfig.minSpreadPct}%)`;
    }
    if (arb.spreadPct > this.arbConfig.maxSpreadPct) {
      return `Spread unrealistic: ${arb.spreadPct.toFixed(1)}% (data error)`;
    }

    // Profit check (percentage-based)
    if (profitAfterGasPct < this.arbConfig.minProfitAfterGasPct) {
      return `Profit too low: ${profitAfterGasPct.toFixed(2)}% (min ${this.arbConfig.minProfitAfterGasPct}%)`;
    }

    return undefined;
  }

  /**
   * Log sentiment evaluation details
   */
  private logSentimentEvaluation(
    arb: ArbitrageOpportunity,
    baseConfidence: number,
    sentimentAdjustment: SentimentAdjustedScore
  ): void {
    if (!this.arbConfig.verbose) return;

    const confidence = sentimentAdjustment.finalScore;
    const delta = confidence - baseConfidence;

    // Check direction
    const direction = this.analyzeDirection(arb);
    const directionAllowed = this.isDirectionAllowed(direction);
    const directionType = direction.isDexToDex ? 'DEX→DEX' :
                          direction.isDexToCex ? 'DEX→CEX' :
                          direction.isCexToDex ? 'CEX→DEX' : 'CEX→CEX';

    console.log(`\n┌─────────────────────────────────────────────────────────┐`);
    console.log(`│  ⚡ ARBITRAGE OPPORTUNITY EVALUATION                    │`);
    console.log(`├─────────────────────────────────────────────────────────┤`);
    console.log(`│  Symbol:              ${arb.symbol.padEnd(35)}│`);
    console.log(`│  Route:               ${(arb.buyExchange + '→' + arb.sellExchange).padEnd(35)}│`);
    console.log(`│  Direction:           ${(directionType + (directionAllowed ? ' ✅' : ' ❌ BLOCKED')).padEnd(35)}│`);
    console.log(`│  Spread:              ${(arb.spreadPct.toFixed(2) + '%').padEnd(35)}│`);
    console.log(`├─────────────────────────────────────────────────────────┤`);
    console.log(`│  ML Confidence:       ${(baseConfidence.toFixed(3) + (baseConfidence >= this.arbConfig.minConfidence ? ' ✅' : ' ❌')).padEnd(35)}│`);
    if (sentimentAdjustment.sentimentAvailable) {
      console.log(`│  Sentiment:           ${sentimentAdjustment.rawSentiment.toFixed(3)} (${sentimentAdjustment.signal})`.padEnd(60) + `│`);
      console.log(`│  Final Confidence:    ${confidence.toFixed(3).padEnd(35)}│`);
    } else {
      console.log(`│  Sentiment:           ${'unavailable (using ML only)'.padEnd(35)}│`);
    }
    console.log(`├─────────────────────────────────────────────────────────┤`);
    console.log(`│  Min Confidence:      ${(this.arbConfig.minConfidence.toFixed(2) + ' (' + (confidence >= this.arbConfig.minConfidence ? 'PASS' : 'FAIL') + ')').padEnd(35)}│`);
    console.log(`│  Min Spread:          ${(this.arbConfig.minSpreadPct.toFixed(2) + '% (' + (arb.spreadPct >= this.arbConfig.minSpreadPct ? 'PASS' : 'FAIL') + ')').padEnd(35)}│`);
    console.log(`│  Direction Allowed:   ${(directionAllowed ? 'YES' : 'NO - ' + directionType + ' disabled').padEnd(35)}│`);
    console.log(`└─────────────────────────────────────────────────────────┘`);
  }

  /**
   * Update arbitrage-specific configuration
   */
  updateArbitrageConfig(config: Partial<ArbitrageAnalystConfig>): void {
    this.arbConfig = { ...this.arbConfig, ...config };
    this.config = this.arbConfig;
  }
}

