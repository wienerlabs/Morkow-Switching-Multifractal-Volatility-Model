/**
 * Risk Manager Service
 *
 * Data-driven risk management for LP rebalancing trades.
 * Limits are calculated from historical backtest analysis.
 * Includes regime-aware position sizing.
 * Integrates with benchmark monitoring for auto-pause.
 *
 * Source: models/lp_rebalancer/metadata/risk_analysis.json
 */
import { logger } from './logger.js';
import { MarketRegime, RegimeDetector, RegimeResult, ALAMS_REGIME_POSITION_SCALE } from './analysis/regimeDetector.js';
import { benchmarkMonitor } from './monitoring/benchmarkMonitor.js';

/**
 * Risk limits derived from backtest analysis
 * Updated: 2026-01-06
 * Sample size: 21 trades over 30 days
 * Data quality score: 56%
 */
export interface RiskLimits {
  // Position sizing (Half-Kelly based)
  maxPositionPct: number;
  minPositionPct: number;

  // Daily limits
  maxDailyLossPct: number;
  maxDailyTrades: number;

  // Volatility filters
  maxVolatility24h: number;
  minVolatility24h: number;

  // Timing
  minCooldownHours: number;

  // Regime-specific position scaling factors
  regimePositionScaling: {
    BULL: number;      // Multiplier for bull market (e.g., 1.0 = full position)
    BEAR: number;      // Multiplier for bear market (e.g., 0.5 = half position)
    SIDEWAYS: number;  // Multiplier for sideways market
    UNKNOWN: number;   // Multiplier when regime is unknown
  };

  // Metadata
  dataQualityScore: number;
  lastUpdated: string;
}

/**
 * Default limits from risk_analysis.json
 * Rationale for each value included in comments
 */
export const DEFAULT_RISK_LIMITS: RiskLimits = {
  // Configured for $100 capital: 20% max = $20 per position
  maxPositionPct: 20.0,  // 20% max position size
  minPositionPct: 5.0,   // 5% min position size

  // worst_day=-1.48% × 1.5 + std=2.82%
  maxDailyLossPct: 5.0,
  maxDailyTrades: 2,

  // Profitable trades avg_vol=0.8785 ± 0.2749 (but capped for safety)
  // Note: Original values too high, using practical limits
  maxVolatility24h: 0.15,  // 15% daily volatility cap
  minVolatility24h: 0.02,  // Need some movement for opportunity

  // Based on avg 1.0 trades/day optimal spacing
  minCooldownHours: 1,

  // Regime-specific position scaling
  // Conservative in BEAR markets where model may be less reliable
  regimePositionScaling: {
    BULL: 1.0,       // Full position size in bull markets
    BEAR: 0.5,       // 50% position size in bear markets (more conservative)
    SIDEWAYS: 0.75,  // 75% position size in sideways markets
    UNKNOWN: 0.5,    // Conservative when regime is unknown
  },

  // Confidence in these limits
  dataQualityScore: 0.56,
  lastUpdated: '2026-01-06',
};

export interface TradeState {
  dailyPnL: number;          // Current day's P&L in %
  dailyTradeCount: number;   // Trades executed today
  lastTradeTime: Date | null;
  currentPositionPct: number;
}

export interface RiskCheckResult {
  allowed: boolean;
  reason: string;
  suggestedPositionPct?: number;
  warnings: string[];
}

/**
 * Risk Manager for LP Rebalancing
 * 
 * Enforces data-driven limits based on backtest analysis:
 * - Position sizing via Half-Kelly criterion
 * - Daily loss limits based on historical worst day
 * - Volatility filters from profitable trade analysis
 * - Cooldown periods for optimal trade spacing
 */
export class RiskManager {
  private limits: RiskLimits;
  private state: TradeState;
  
  constructor(limits: Partial<RiskLimits> = {}) {
    this.limits = { ...DEFAULT_RISK_LIMITS, ...limits };
    this.state = {
      dailyPnL: 0,
      dailyTradeCount: 0,
      lastTradeTime: null,
      currentPositionPct: 0,
    };
    
    logger.info('RiskManager initialized', {
      maxPosition: this.limits.maxPositionPct,
      maxDailyLoss: this.limits.maxDailyLossPct,
      dataQuality: this.limits.dataQualityScore,
    });
  }
  
  /**
   * Check if a trade is allowed under current risk limits
   */
  checkTradeAllowed(params: {
    proposedPositionPct: number;
    currentVolatility24h: number;
    expectedReturnPct?: number;
  }): RiskCheckResult {
    const warnings: string[] = [];
    const { proposedPositionPct, currentVolatility24h, expectedReturnPct } = params;

    // 0. Benchmark pause check - NEW
    if (benchmarkMonitor.isTradingPaused()) {
      const pauseReason = benchmarkMonitor.getPauseReason();
      return {
        allowed: false,
        reason: `Trading paused due to benchmark violations: ${pauseReason || 'Unknown'}`,
        warnings: ['Resume trading requires manual approval via CLI'],
      };
    }

    // 1. Daily loss limit check
    if (this.state.dailyPnL <= -this.limits.maxDailyLossPct) {
      return {
        allowed: false,
        reason: `Daily loss limit reached: ${this.state.dailyPnL.toFixed(2)}% (limit: -${this.limits.maxDailyLossPct}%)`,
        warnings,
      };
    }
    
    // 2. Daily trade count check
    if (this.state.dailyTradeCount >= this.limits.maxDailyTrades) {
      return {
        allowed: false,
        reason: `Max daily trades reached: ${this.state.dailyTradeCount} (limit: ${this.limits.maxDailyTrades})`,
        warnings,
      };
    }
    
    // 3. Cooldown check
    if (this.state.lastTradeTime) {
      const hoursSinceLast = (Date.now() - this.state.lastTradeTime.getTime()) / (1000 * 60 * 60);
      if (hoursSinceLast < this.limits.minCooldownHours) {
        const remaining = this.limits.minCooldownHours - hoursSinceLast;
        return {
          allowed: false,
          reason: `Cooldown active: ${remaining.toFixed(1)}h remaining (min: ${this.limits.minCooldownHours}h)`,
          warnings,
        };
      }
    }
    
    // 4. Volatility check
    if (currentVolatility24h > this.limits.maxVolatility24h) {
      return {
        allowed: false,
        reason: `Volatility too high: ${(currentVolatility24h * 100).toFixed(1)}% (max: ${(this.limits.maxVolatility24h * 100).toFixed(1)}%)`,
        warnings,
      };
    }
    
    if (currentVolatility24h < this.limits.minVolatility24h) {
      warnings.push(`Low volatility: ${(currentVolatility24h * 100).toFixed(1)}% (min recommended: ${(this.limits.minVolatility24h * 100).toFixed(1)}%)`);
    }

    // 5. Position size check and adjustment
    let adjustedPosition = proposedPositionPct;

    if (proposedPositionPct > this.limits.maxPositionPct) {
      adjustedPosition = this.limits.maxPositionPct;
      warnings.push(`Position capped: ${proposedPositionPct.toFixed(1)}% → ${adjustedPosition.toFixed(1)}%`);
    }

    if (proposedPositionPct < this.limits.minPositionPct) {
      adjustedPosition = this.limits.minPositionPct;
      warnings.push(`Position raised to min: ${proposedPositionPct.toFixed(1)}% → ${adjustedPosition.toFixed(1)}%`);
    }

    // 6. Expected return sanity check (if provided)
    if (expectedReturnPct !== undefined && expectedReturnPct < 0) {
      warnings.push(`Negative expected return: ${expectedReturnPct.toFixed(2)}%`);
    }

    // 7. Data quality warning
    if (this.limits.dataQualityScore < 0.7) {
      warnings.push(`Low data quality (${(this.limits.dataQualityScore * 100).toFixed(0)}%) - limits may be unreliable`);
    }

    return {
      allowed: true,
      reason: 'All risk checks passed',
      suggestedPositionPct: adjustedPosition,
      warnings,
    };
  }

  /**
   * Record a completed trade and update state
   */
  recordTrade(pnlPct: number): void {
    this.state.dailyPnL += pnlPct;
    this.state.dailyTradeCount += 1;
    this.state.lastTradeTime = new Date();

    logger.info('Trade recorded', {
      pnlPct,
      dailyPnL: this.state.dailyPnL,
      dailyTradeCount: this.state.dailyTradeCount,
    });
  }

  /**
   * Reset daily counters (call at start of new trading day)
   */
  resetDaily(): void {
    this.state.dailyPnL = 0;
    this.state.dailyTradeCount = 0;
    logger.info('Daily risk counters reset');
  }

  /**
   * Calculate position size based on confidence and volatility
   */
  calculatePositionSize(params: {
    modelConfidence: number;      // 0-1 probability
    currentVolatility24h: number; // e.g., 0.05 for 5%
    portfolioValueUsd: number;
  }): { positionPct: number; positionUsd: number; rationale: string } {
    const { modelConfidence, currentVolatility24h, portfolioValueUsd } = params;

    // Base position from Half-Kelly
    let positionPct = this.limits.maxPositionPct;

    // Scale down based on model confidence (0.5 = 50% of max, 1.0 = 100% of max)
    const confidenceScalar = Math.max(0.5, Math.min(1.0, modelConfidence));
    positionPct *= confidenceScalar;

    // Scale down for high volatility (inverse relationship)
    const volScalar = Math.max(0.5, 1 - (currentVolatility24h / this.limits.maxVolatility24h));
    positionPct *= volScalar;

    // Apply limits
    positionPct = Math.max(this.limits.minPositionPct, Math.min(this.limits.maxPositionPct, positionPct));

    const positionUsd = (positionPct / 100) * portfolioValueUsd;

    return {
      positionPct: Math.round(positionPct * 10) / 10,
      positionUsd: Math.round(positionUsd),
      rationale: `Half-Kelly=${this.limits.maxPositionPct}% × confidence=${confidenceScalar.toFixed(2)} × vol_adj=${volScalar.toFixed(2)}`,
    };
  }

  /**
   * Calculate regime-aware position size
   *
   * Adjusts position size based on current market regime and model's
   * training data distribution. More conservative in regimes where
   * model has less training data or historically performed worse.
   */
  calculateRegimeAwarePositionSize(params: {
    modelConfidence: number;
    currentVolatility24h: number;
    portfolioValueUsd: number;
    currentRegime: MarketRegime;
    modelTrainingRegimes?: { BULL: number; BEAR: number; SIDEWAYS: number };
    alamsRegimeIndex?: number | null;
  }): {
    positionPct: number;
    positionUsd: number;
    rationale: string;
    regimeAdjustment: number;
    warnings: string[];
  } {
    const {
      modelConfidence,
      currentVolatility24h,
      portfolioValueUsd,
      currentRegime,
      modelTrainingRegimes,
      alamsRegimeIndex,
    } = params;

    const warnings: string[] = [];

    // Get base position calculation
    const baseResult = this.calculatePositionSize({
      modelConfidence,
      currentVolatility24h,
      portfolioValueUsd,
    });

    let positionPct = baseResult.positionPct;

    // Use A-LAMS fine-grained scaling when available, fall back to TS regime scaling
    let regimeScaling: number;
    let regimeSource: string;
    if (alamsRegimeIndex != null && alamsRegimeIndex in ALAMS_REGIME_POSITION_SCALE) {
      regimeScaling = ALAMS_REGIME_POSITION_SCALE[alamsRegimeIndex];
      regimeSource = `alams-regime-${alamsRegimeIndex}`;
      if (alamsRegimeIndex >= 4) {
        warnings.push(`Crisis regime (A-LAMS ${alamsRegimeIndex}) — position scaled to ${regimeScaling * 100}%`);
      }
    } else {
      regimeScaling = this.limits.regimePositionScaling[currentRegime] || 0.5;
      regimeSource = `ts-${currentRegime}`;
    }
    positionPct *= regimeScaling;

    // Additional scaling if model has little training data in current regime
    let trainingAdjustment = 1.0;
    if (modelTrainingRegimes) {
      const total = modelTrainingRegimes.BULL + modelTrainingRegimes.BEAR + modelTrainingRegimes.SIDEWAYS;
      if (total > 0) {
        const regimeCount = modelTrainingRegimes[currentRegime as keyof typeof modelTrainingRegimes] || 0;
        const regimePercentage = (regimeCount / total) * 100;

        // If less than 10% of training data was in this regime, scale down further
        if (regimePercentage < 10) {
          trainingAdjustment = 0.5;  // Cut position in half for untested regimes
          warnings.push(`Model has only ${regimePercentage.toFixed(1)}% training data in ${currentRegime} regime`);
        } else if (regimePercentage < 20) {
          trainingAdjustment = 0.75;
          warnings.push(`Limited training data (${regimePercentage.toFixed(1)}%) in ${currentRegime} regime`);
        }
      }
    }

    positionPct *= trainingAdjustment;

    // Apply limits
    positionPct = Math.max(this.limits.minPositionPct, Math.min(this.limits.maxPositionPct, positionPct));

    const positionUsd = (positionPct / 100) * portfolioValueUsd;
    const regimeAdjustment = regimeScaling * trainingAdjustment;

    // Log regime adjustment
    if (regimeAdjustment < 1.0) {
      logger.info('Regime-adjusted position size', {
        regime: currentRegime,
        regimeScaling,
        trainingAdjustment,
        originalPct: baseResult.positionPct,
        adjustedPct: positionPct,
      });
    }

    return {
      positionPct: Math.round(positionPct * 10) / 10,
      positionUsd: Math.round(positionUsd),
      rationale: `${baseResult.rationale} × regime_adj=${regimeAdjustment.toFixed(2)} (${regimeSource})`,
      regimeAdjustment,
      warnings,
    };
  }

  /**
   * Get current risk state
   */
  getState(): TradeState {
    return { ...this.state };
  }

  /**
   * Get configured limits
   */
  getLimits(): RiskLimits {
    return { ...this.limits };
  }

  /**
   * Update limits (e.g., after new backtest)
   */
  updateLimits(newLimits: Partial<RiskLimits>): void {
    this.limits = { ...this.limits, ...newLimits };
    logger.info('Risk limits updated', newLimits);
  }
}

// Singleton instance for global use
let riskManagerInstance: RiskManager | null = null;

export function getRiskManager(limits?: Partial<RiskLimits>): RiskManager {
  if (!riskManagerInstance) {
    riskManagerInstance = new RiskManager(limits);
  }
  return riskManagerInstance;
}

export function resetRiskManager(): void {
  riskManagerInstance = null;
}

