/**
 * SpotAnalyst - ML-Enhanced Spot Trading Analyst
 *
 * Evaluates spot trading opportunities using:
 * - XGBoost ML model (70% weight)
 * - Rule-based signals (30% weight)
 * - Token whitelist filtering
 * - Risk management
 */

import { BaseAnalyst, DEFAULT_ANALYST_CONFIG, type AnalystConfig } from './BaseAnalyst.js';
import type { ApprovedToken } from '../../services/trading/types.js';
import { SpotMLModel } from '../../services/spot/ml/spotMLModel.js';
import { SpotFeatureExtractor, type TokenMarketData } from '../../services/spot/ml/featureExtractor.js';
import { RiskManager } from '../../services/riskManager.js';
import { logger } from '../../services/logger.js';

/**
 * Input for SpotAnalyst
 */
export interface SpotAnalysisInput {
  tokens: ApprovedToken[];
  marketData: Map<string, TokenMarketData>;  // token address -> market data
  volatility24h: number;
  portfolioValueUsd?: number;
}

/**
 * Output from SpotAnalyst
 */
export interface SpotOpportunityResult {
  type: 'spot';
  name: string;
  token: ApprovedToken;
  expectedReturn: number;
  riskScore: number;
  confidence: number;
  riskAdjustedReturn: number;
  approved: boolean;
  rejectReason?: string;
  warnings: string[];
  raw: {
    mlProbability: number;
    ruleScore: number;
    features: any;
  };
  // Trading information
  trading: {
    direction: 'LONG';  // Always LONG for spot
    leverage: 1;  // Always 1x for spot
    entryPrice: number;
    tp1: { price: number; percentage: number; exitPct: number };
    tp2: { price: number; percentage: number; exitPct: number };
    tp3: { price: number; percentage: number; exitPct: number };
    stopLoss: { price: number; percentage: number };
    positionSizeUsd: number;
    expectedReturnUsd: number;
    maxLossUsd: number;
    riskRewardRatio: number;
  };
}

/**
 * Configuration for SpotAnalyst
 */
export interface SpotAnalystConfig extends AnalystConfig {
  minConfidenceThreshold: number;  // Minimum confidence to approve (0.45 = 45%)
  mlWeight: number;                 // Weight for ML prediction (0.7 = 70%)
  ruleWeight: number;               // Weight for rule-based score (0.3 = 30%)
  minLiquidity: number;             // Minimum liquidity ($200K)
  maxPositions: number;             // Max concurrent positions (4)
}

export const DEFAULT_SPOT_CONFIG: SpotAnalystConfig = {
  ...DEFAULT_ANALYST_CONFIG,
  minConfidenceThreshold: 0.50,   // 50% minimum confidence
  mlWeight: 0.70,                 // 70% ML
  ruleWeight: 0.30,               // 30% rules
  minLiquidity: 50_000,           // $50K minimum liquidity
  maxPositions: 4,
};

/**
 * SpotAnalyst - ML-Enhanced Spot Trading Evaluator
 */
export class SpotAnalyst extends BaseAnalyst<SpotAnalysisInput, SpotOpportunityResult> {
  private spotConfig: SpotAnalystConfig;
  private mlModel: SpotMLModel;
  private featureExtractor: SpotFeatureExtractor;
  private riskManager: RiskManager;
  private modelInitialized: boolean = false;

  constructor(config: Partial<SpotAnalystConfig> = {}) {
    super({ ...DEFAULT_SPOT_CONFIG, ...config });
    this.spotConfig = { ...DEFAULT_SPOT_CONFIG, ...config };

    this.mlModel = new SpotMLModel();
    this.featureExtractor = new SpotFeatureExtractor();

    this.riskManager = new RiskManager({
      maxPositionPct: 0.12,  // 12% max per position
      maxDailyLossPct: 0.06,
      dataQualityScore: 0.75,
    });

    logger.info('[SpotAnalyst] Initialized', { config: this.spotConfig });
  }

  getName(): string {
    return 'SpotAnalyst';
  }

  /**
   * Initialize ML model (call once before first use)
   */
  async initialize(): Promise<void> {
    if (this.modelInitialized) {
      return;
    }

    try {
      await this.mlModel.load();
      this.modelInitialized = true;
      logger.info('[SpotAnalyst] ML model loaded successfully');
    } catch (error) {
      logger.error('[SpotAnalyst] Failed to load ML model', { error });
      throw error;
    }
  }

  /**
   * Analyze spot trading opportunities
   */
  async analyze(input: SpotAnalysisInput): Promise<SpotOpportunityResult[]> {
    // Ensure ML model is loaded
    if (!this.modelInitialized) {
      await this.initialize();
    }

    const results: SpotOpportunityResult[] = [];
    const volatility = input.volatility24h ?? this.config.volatility24h;

    logger.info('[SpotAnalyst] Analyzing tokens', { count: input.tokens.length });

    for (const token of input.tokens) {
      const marketData = input.marketData.get(token.address);

      if (!marketData) {
        logger.warn('[SpotAnalyst] No market data for token', { symbol: token.symbol });
        continue;
      }

      const result = await this.evaluateToken(token, marketData, volatility);
      results.push(result);
    }

    const approved = results.filter(r => r.approved);
    logger.info('[SpotAnalyst] Analysis complete', {
      total: results.length,
      approved: approved.length,
      rejected: results.length - approved.length,
    });

    // Print summary of all tokens scanned
    console.log('\n' + '='.repeat(80));
    console.log('📊 SPOT ANALYST SCAN SUMMARY');
    console.log('='.repeat(80));
    console.log(`Total Tokens Scanned: ${results.length}`);
    console.log(`Approved: ${approved.length} | Rejected: ${results.length - approved.length}`);
    console.log('='.repeat(80) + '\n');

    results.forEach((result, index) => {
      const status = result.approved ? '✅ APPROVED' : '❌ REJECTED';
      console.log(`[${index + 1}] ${result.token.symbol} - ${status}`);
      console.log(`    ML Probability:  ${(Number(result.raw.mlProbability) * 100).toFixed(1)}%`);
      console.log(`    Rule Score:      ${Number(result.raw.ruleScore).toFixed(0)}/160 (${(Number(result.raw.ruleScore) / 160 * 100).toFixed(1)}%)`);
      console.log(`    Final Confidence: ${(Number(result.confidence) * 100).toFixed(1)}%`);
      console.log(`    Threshold:       ${(this.spotConfig.minConfidenceThreshold * 100).toFixed(0)}%`);
      if (result.rejectReason) {
        console.log(`    Reject Reason:   ${result.rejectReason}`);
      }
      if (result.approved) {
        console.log(`    Position Size:   $${result.trading.positionSizeUsd.toFixed(2)}`);
        console.log(`    Expected Return: $${result.trading.expectedReturnUsd.toFixed(2)}`);
      }
      console.log('');
    });
    console.log('='.repeat(80) + '\n');

    return results;
  }

  /**
   * Evaluate a single token
   */
  private async evaluateToken(
    token: ApprovedToken,
    marketData: TokenMarketData,
    volatility: number
  ): Promise<SpotOpportunityResult> {
    const warnings: string[] = [];
    let rejectReason: string | undefined;

    // 1. Extract features for ML model
    const features = await this.featureExtractor.extractFeatures(token, marketData);

    // 2. Get ML prediction
    const mlPrediction = await this.mlModel.predict(features, 0.5);

    // 3. Calculate rule-based score (for backup/validation)
    const ruleScore = this.calculateRuleBasedScore(features);
    const ruleConfidence = ruleScore / 160; // Max score is 160

    // 4. Combine ML + Rules (70% ML, 30% rules)
    const finalConfidence = (Number(mlPrediction.confidence) * Number(this.spotConfig.mlWeight)) +
                           (Number(ruleConfidence) * Number(this.spotConfig.ruleWeight));

    // 5. Calculate expected return (based on historical TP1 target)
    const expectedReturn = 0.12; // 12% target (TP1)

    // 6. Calculate risk score (1-10)
    let riskScore = 5; // Base risk

    // Adjust for token tier
    if (token.tier === 1) riskScore -= 1;
    if (token.tier === 3) riskScore += 1;

    // Adjust for volatility
    if (volatility > 0.05) riskScore += 1;
    if (volatility > 0.10) riskScore += 1;

    // Adjust for liquidity
    if (token.liquidity < 500_000) riskScore += 1;
    if (token.liquidity > 1_000_000) riskScore -= 1;

    // Adjust for ML confidence
    if (mlPrediction.confidence < 0.4) riskScore += 2;
    if (mlPrediction.confidence > 0.7) riskScore -= 1;

    riskScore = Math.max(1, Math.min(10, riskScore));

    // 7. Risk-adjusted return
    const riskAdjustedReturn = expectedReturn * (1 - riskScore / 20);

    // 8. Validation checks

    // Check confidence threshold
    if (finalConfidence < this.spotConfig.minConfidenceThreshold) {
      rejectReason = `Low confidence: ${(finalConfidence * 100).toFixed(1)}% < ${(this.spotConfig.minConfidenceThreshold * 100).toFixed(0)}%`;
    }

    // Check liquidity
    if (token.liquidity < this.spotConfig.minLiquidity) {
      rejectReason = `Low liquidity: $${(token.liquidity / 1000).toFixed(0)}K < $${(this.spotConfig.minLiquidity / 1000).toFixed(0)}K`;
    }

    // Check risk manager
    const positionCalc = this.riskManager.calculatePositionSize({
      modelConfidence: finalConfidence,
      currentVolatility24h: volatility,
      portfolioValueUsd: 100_000, // Default
    });

    const riskCheck = this.riskManager.checkTradeAllowed({
      proposedPositionPct: positionCalc.positionPct,
      currentVolatility24h: volatility,
      expectedReturnPct: expectedReturn,
    });

    if (!riskCheck.allowed) {
      rejectReason = riskCheck.reason;
    }

    // Warnings
    if (Number(mlPrediction.confidence) < 0.5) {
      warnings.push(`ML confidence below 50%: ${(Number(mlPrediction.confidence) * 100).toFixed(1)}%`);
    }

    if (Number(ruleConfidence) < 0.3) {
      warnings.push(`Rule-based score low: ${(Number(ruleConfidence) * 100).toFixed(1)}%`);
    }

    if (Number(volatility) > 0.08) {
      warnings.push(`High volatility: ${(Number(volatility) * 100).toFixed(1)}%`);
    }

    if (token.tier === 3) {
      warnings.push('Tier 3 token (lower quality)');
    }

    const approved = !rejectReason;

    // Calculate trading parameters using REAL market data
    const entryPrice = marketData.currentPrice;
    const tp1Price = entryPrice * 1.12;  // +12%
    const tp2Price = entryPrice * 1.25;  // +25%
    const tp3Price = entryPrice * 1.40;  // +40%
    const stopLossPrice = entryPrice * 0.92;  // -8%

    // Calculate position size based on confidence and volatility
    const basePositionPct = 0.03;  // 3% base
    const confidenceMultiplier = finalConfidence;
    const volatilityAdjustment = Math.max(0.5, 1 - volatility);
    const positionPct = basePositionPct * confidenceMultiplier * volatilityAdjustment;
    const portfolioValue = this.config.portfolioValueUsd;
    const positionSizeUsd = portfolioValue * positionPct;

    // Calculate returns and losses
    const expectedReturnUsd = positionSizeUsd * expectedReturn;  // If TP1 hit
    const maxLossUsd = positionSizeUsd * 0.08;  // If stop loss hit
    const riskRewardRatio = expectedReturn / 0.08;  // 12% / 8% = 1.5

    const result: SpotOpportunityResult = {
      type: 'spot',
      name: `${token.symbol} Spot Entry`,
      token,
      expectedReturn,
      riskScore,
      confidence: finalConfidence,
      riskAdjustedReturn,
      approved,
      rejectReason,
      warnings,
      raw: {
        mlProbability: mlPrediction.probability,
        ruleScore,
        features,
      },
      trading: {
        direction: 'LONG',
        leverage: 1,
        entryPrice,
        tp1: { price: tp1Price, percentage: 12, exitPct: 40 },
        tp2: { price: tp2Price, percentage: 25, exitPct: 35 },
        tp3: { price: tp3Price, percentage: 40, exitPct: 25 },
        stopLoss: { price: stopLossPrice, percentage: -8 },
        positionSizeUsd,
        expectedReturnUsd,
        maxLossUsd,
        riskRewardRatio,
      },
    };

    // Formatting helpers
    const formatUsd = (num: number) => {
      // Handle very small numbers (< $0.01)
      if (num < 0.01 && num > 0) {
        return num.toFixed(6); // Show 6 decimals for small prices
      }
      return num.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    };
    const formatCompact = (num: number) => {
      if (num >= 1_000_000) return `$${(num / 1_000_000).toFixed(2)}M`;
      if (num >= 1_000) return `$${(num / 1_000).toFixed(2)}K`;
      return `$${num.toFixed(2)}`;
    };

    // Log detailed evaluation with COMPLETE trading information
    console.log(`\n┌──────────────────────────────────────────────────────────┐`);
    console.log(`│  🎯 SPOT TRADE OPPORTUNITY: ${token.symbol.padEnd(27)} │`);
    console.log(`├──────────────────────────────────────────────────────────┤`);
    console.log(`│  Token Info:`.padEnd(61) + '│');
    console.log(`│    Address:    ${token.address}`.padEnd(61) + '│');
    console.log(`│    Price:      $${formatUsd(entryPrice)}`.padEnd(61) + '│');
    console.log(`│    Market Cap: ${formatCompact(token.marketCap)}`.padEnd(61) + '│');
    console.log(`│    Liquidity:  ${formatCompact(token.liquidity)}`.padEnd(61) + '│');
    console.log(`│    Volume 24h: ${formatCompact(token.volume24h)}`.padEnd(61) + '│');
    console.log(`├──────────────────────────────────────────────────────────┤`);
    console.log(`│  Trading Setup:`.padEnd(61) + '│');
    console.log(`│    Direction:     LONG (Spot)`.padEnd(61) + '│');
    console.log(`│    Leverage:      1x`.padEnd(61) + '│');
    console.log(`│    Confidence:    ${(finalConfidence * 100).toFixed(0)}%`.padEnd(61) + '│');
    console.log(`│    Position Size: $${formatUsd(positionSizeUsd)}`.padEnd(61) + '│');
    console.log(`├──────────────────────────────────────────────────────────┤`);
    console.log(`│  Entry & Exit:`.padEnd(61) + '│');
    console.log(`│    ENTRY:       $${formatUsd(entryPrice)}`.padEnd(61) + '│');
    console.log(`│    TP1 (+12%):  $${formatUsd(tp1Price)} → Exit 40%`.padEnd(61) + '│');
    console.log(`│    TP2 (+25%):  $${formatUsd(tp2Price)} → Exit 35%`.padEnd(61) + '│');
    console.log(`│    TP3 (+40%):  $${formatUsd(tp3Price)} → Exit 25%`.padEnd(61) + '│');
    console.log(`│    STOP LOSS:   $${formatUsd(stopLossPrice)} (-8%)`.padEnd(61) + '│');
    console.log(`├──────────────────────────────────────────────────────────┤`);
    console.log(`│  Risk/Reward:`.padEnd(61) + '│');
    console.log(`│    Expected Return: $${formatUsd(expectedReturnUsd)} (if TP1 hit)`.padEnd(61) + '│');
    console.log(`│    Max Loss:        $${formatUsd(maxLossUsd)} (if stopped out)`.padEnd(61) + '│');
    console.log(`│    Risk Score:      ${riskScore}/10`.padEnd(61) + '│');
    console.log(`│    R/R Ratio:       ${riskRewardRatio.toFixed(1)}:1`.padEnd(61) + '│');
    console.log(`├──────────────────────────────────────────────────────────┤`);
    console.log(`│  Analysis Details:`.padEnd(61) + '│');
    console.log(`│    ML Prediction:   ${(Number(mlPrediction.probability) * 100).toFixed(1)}% BUY probability`.padEnd(61) + '│');
    console.log(`│    ML Confidence:   ${(Number(mlPrediction.confidence) * 100).toFixed(1)}%`.padEnd(61) + '│');
    console.log(`│    Rule Score:      ${Number(ruleScore).toFixed(0)}/160 (${(Number(ruleConfidence) * 100).toFixed(1)}%)`.padEnd(61) + '│');
    console.log(`│    Combined Score:  ${(Number(finalConfidence) * 100).toFixed(1)}% (${(Number(this.spotConfig.mlWeight) * 100).toFixed(0)}% ML + ${(Number(this.spotConfig.ruleWeight) * 100).toFixed(0)}% Rules)`.padEnd(61) + '│');
    console.log(`├──────────────────────────────────────────────────────────┤`);
    console.log(`│  Decision:          ${approved ? '✅ APPROVED' : '❌ REJECTED'}`.padEnd(61) + '│');
    console.log(`│  Threshold:         ${(this.spotConfig.minConfidenceThreshold * 100).toFixed(0)}%`.padEnd(61) + '│');
    if (rejectReason) {
      console.log(`│  Reason:            ${rejectReason.substring(0, 38)}`.padEnd(61) + '│');
    }
    if (warnings.length > 0) {
      console.log(`├──────────────────────────────────────────────────────────┤`);
      console.log(`│  ⚠️  Warnings:`.padEnd(61) + '│');
      warnings.forEach(w => {
        console.log(`│    - ${w.substring(0, 51)}`.padEnd(61) + '│');
      });
    }
    console.log(`└──────────────────────────────────────────────────────────┘`);


    return result;
  }

  /**
   * Calculate rule-based score (max 160 points)
   * This is the backup/validation system
   */
  private calculateRuleBasedScore(features: any): number {
    let score = 0;

    // Technical signals (100 points max)

    // RSI oversold (RSI < 35): +20 pts
    if ((features.rsi_14 || 50) < 35) score += 20;

    // Price dip 10-25% from 7-day high: +25 pts
    const priceVs7d = features.price_vs_7d_high || 0;
    if (priceVs7d >= -0.25 && priceVs7d <= -0.10) score += 25;

    // Volume spike (1.5x avg): +15 pts
    if ((features.volume_vs_7d_avg || 1) >= 1.5) score += 15;

    // Support test (within 5%): +15 pts
    if (Math.abs(features.distance_to_support || 1) <= 0.05) score += 15;

    // Above 50-day MA: +10 pts
    if (features.above_ma50 === 1) score += 10;

    // MACD bullish: +10 pts
    if (features.macd_bullish === 1) score += 10;

    // Bollinger touch: +5 pts
    if (features.bb_touch_lower === 1) score += 5;

    // Sentiment signals (30 points max)

    // Positive sentiment: +10 pts
    if ((features.sentiment_score || 0) > 0.2) score += 10;

    // Sentiment improving: +10 pts
    if ((features.sentiment_velocity || 0) > 0) score += 10;

    // Social volume spike: +5 pts
    if ((features.social_volume_normalized || 1) > 1.5) score += 5;

    // Influencer mentions: +5 pts
    if ((features.influencer_mentions || 0) > 0) score += 5;

    // Market context (30 points max)

    // SOL above 20-day MA: +10 pts
    if (features.sol_above_ma20 === 1) score += 10;

    // Not in bear market: +10 pts
    if (features.market_regime_bear !== 1) score += 10;

    // Low correlation to SOL (diversification): +5 pts
    if (Math.abs(features.correlation_to_sol || 0) < 0.5) score += 5;

    // Low market volatility: +5 pts
    if ((features.market_volatility || 0) < 0.05) score += 5;

    return score;
  }
}
