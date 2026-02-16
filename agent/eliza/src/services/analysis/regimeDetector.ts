/**
 * Market Regime Detector Service
 * 
 * Real-time detection of market regimes (BULL, BEAR, SIDEWAYS)
 * for regime-aware trading decisions and position sizing.
 * 
 * Regime Classification Logic:
 * - BULL: >15% return over window + strong uptrend (>60% consistency)
 * - BEAR: <-15% return over window + strong downtrend (>60% consistency)
 * - SIDEWAYS: Returns within ±15% OR weak trend (<60% consistency)
 */
import { logger } from '../logger.js';

export enum MarketRegime {
  BULL = 'BULL',
  BEAR = 'BEAR',
  SIDEWAYS = 'SIDEWAYS',
  UNKNOWN = 'UNKNOWN',
}

/**
 * Map A-LAMS 5-regime index to TS MarketRegime.
 * A-LAMS: 0=very-low-vol, 1=low-vol, 2=normal, 3=high-vol, 4=crisis
 */
export function mapAlamsRegimeToMarketRegime(alamsRegime: number): MarketRegime {
  if (alamsRegime <= 1) return MarketRegime.BULL;
  if (alamsRegime === 2) return MarketRegime.SIDEWAYS;
  // regimes 3 and 4 both map to BEAR (crisis is a severe BEAR)
  return MarketRegime.BEAR;
}

/**
 * Position scaling factors per A-LAMS regime.
 * Crisis (4) is stricter than regular BEAR (3).
 */
export const ALAMS_REGIME_POSITION_SCALE: Record<number, number> = {
  0: 1.0,    // very-low-vol → full
  1: 1.0,    // low-vol → full
  2: 0.75,   // normal → 75%
  3: 0.5,    // high-vol → 50%
  4: 0.25,   // crisis → 25%
};

export interface RegimeConfig {
  returnWindow: number;           // Periods for return calculation (default: 30)
  volatilityWindow: number;       // Periods for volatility calculation (default: 20)
  bullReturnThreshold: number;    // Return threshold for BULL (default: 0.15)
  bearReturnThreshold: number;    // Return threshold for BEAR (default: -0.15)
  minTrendStrength: number;       // Min trend consistency for directional regime (default: 0.6)
}

export interface RegimeResult {
  regime: MarketRegime;
  confidence: number;             // 0-1 scale
  returns: number;                // Period return as decimal
  trendStrength: number;          // 0-1 trend consistency
  volatility: number;             // Annualized volatility
  timestamp: Date;
}

export interface RegimeChangeEvent {
  previousRegime: MarketRegime;
  newRegime: MarketRegime;
  timestamp: Date;
  confidence: number;
}

export const DEFAULT_REGIME_CONFIG: RegimeConfig = {
  returnWindow: 30,
  volatilityWindow: 20,
  bullReturnThreshold: 0.15,
  bearReturnThreshold: -0.15,
  minTrendStrength: 0.6,
};

/**
 * Market Regime Detector
 * 
 * Detects current market regime from price data for:
 * - Regime-aware position sizing
 * - Confidence threshold adjustments
 * - Flagging untested regime conditions
 */
export class RegimeDetector {
  private config: RegimeConfig;
  private priceHistory: number[] = [];
  private currentRegime: MarketRegime = MarketRegime.UNKNOWN;
  private lastRegimeChange: Date | null = null;
  private onRegimeChange?: (event: RegimeChangeEvent) => void;

  // A-LAMS override: when set, getCurrentRegime() returns this instead of TS-detected regime
  private regimeOverride: MarketRegime | null = null;
  private regimeOverrideAlamsIndex: number | null = null;
  private regimeOverrideAt: Date | null = null;

  constructor(config: Partial<RegimeConfig> = {}, onRegimeChange?: (event: RegimeChangeEvent) => void) {
    this.config = { ...DEFAULT_REGIME_CONFIG, ...config };
    this.onRegimeChange = onRegimeChange;

    logger.info('RegimeDetector initialized', {
      returnWindow: this.config.returnWindow,
      bullThreshold: `>${this.config.bullReturnThreshold * 100}%`,
      bearThreshold: `<${this.config.bearReturnThreshold * 100}%`,
    });
  }

  /**
   * Set regime from A-LAMS model (overrides TS-based detection).
   * When set, getCurrentRegime() returns the A-LAMS-derived regime.
   */
  setRegimeOverride(alamsRegime: number): void {
    const mapped = mapAlamsRegimeToMarketRegime(alamsRegime);
    const tsRegime = this.currentRegime;

    if (mapped !== tsRegime && tsRegime !== MarketRegime.UNKNOWN) {
      logger.warn('A-LAMS regime differs from TS regime detector', {
        alamsRegime,
        alamsRegimeMapped: mapped,
        tsRegime,
      });
    }

    const previous = this.regimeOverride;
    this.regimeOverride = mapped;
    this.regimeOverrideAlamsIndex = alamsRegime;
    this.regimeOverrideAt = new Date();

    if (previous !== null && previous !== mapped) {
      logger.warn('A-LAMS regime override changed', {
        previous,
        new: mapped,
        alamsRegime,
      });

      if (this.onRegimeChange) {
        this.onRegimeChange({
          previousRegime: previous,
          newRegime: mapped,
          timestamp: this.regimeOverrideAt,
          confidence: 1.0,
        });
      }
    }
  }

  clearRegimeOverride(): void {
    if (this.regimeOverride !== null) {
      logger.info('A-LAMS regime override cleared, falling back to TS detection');
    }
    this.regimeOverride = null;
    this.regimeOverrideAlamsIndex = null;
    this.regimeOverrideAt = null;
  }

  /** Returns true if A-LAMS regime override is active. */
  hasRegimeOverride(): boolean {
    return this.regimeOverride !== null;
  }

  /** Returns the raw A-LAMS regime index (0-4) if override is active, else null. */
  getAlamsRegimeIndex(): number | null {
    return this.regimeOverrideAlamsIndex;
  }
  
  /**
   * Add a new price observation
   */
  addPrice(price: number, timestamp?: Date): void {
    this.priceHistory.push(price);
    
    // Keep only needed history
    const maxHistory = Math.max(this.config.returnWindow, this.config.volatilityWindow) + 10;
    if (this.priceHistory.length > maxHistory) {
      this.priceHistory = this.priceHistory.slice(-maxHistory);
    }
  }
  
  /**
   * Detect current market regime
   */
  detect(): RegimeResult {
    const now = new Date();
    
    if (this.priceHistory.length < this.config.returnWindow) {
      return {
        regime: MarketRegime.UNKNOWN,
        confidence: 0,
        returns: 0,
        trendStrength: 0,
        volatility: 0,
        timestamp: now,
      };
    }
    
    // Calculate metrics
    const returns = this.calculateReturns();
    const trendStrength = this.calculateTrendStrength();
    const volatility = this.calculateVolatility();
    
    // Classify regime
    const { regime, confidence } = this.classify(returns, trendStrength, volatility);
    
    // Check for regime change
    if (regime !== this.currentRegime && regime !== MarketRegime.UNKNOWN) {
      const previousRegime = this.currentRegime;
      this.currentRegime = regime;
      this.lastRegimeChange = now;
      
      logger.warn('Market regime changed', {
        previous: previousRegime,
        new: regime,
        confidence: confidence.toFixed(2),
      });
      
      if (this.onRegimeChange && previousRegime !== MarketRegime.UNKNOWN) {
        this.onRegimeChange({
          previousRegime,
          newRegime: regime,
          timestamp: now,
          confidence,
        });
      }
    }
    
    return {
      regime,
      confidence,
      returns,
      trendStrength,
      volatility,
      timestamp: now,
    };
  }
  
  /**
   * Get current regime. Returns A-LAMS override if active, otherwise TS-detected regime.
   */
  getCurrentRegime(): MarketRegime {
    return this.regimeOverride ?? this.currentRegime;
  }

  /**
   * Get TS-detected regime (ignores A-LAMS override).
   */
  getTSDetectedRegime(): MarketRegime {
    return this.currentRegime;
  }
  
  /**
   * Calculate period return
   */
  private calculateReturns(): number {
    const window = this.config.returnWindow;
    if (this.priceHistory.length < window) return 0;

    const currentPrice = this.priceHistory[this.priceHistory.length - 1];
    const startPrice = this.priceHistory[this.priceHistory.length - window];

    if (startPrice === 0) return 0;
    return (currentPrice - startPrice) / startPrice;
  }

  /**
   * Calculate trend strength (0-1)
   * Measures consistency of price vs moving average
   */
  private calculateTrendStrength(): number {
    const window = this.config.returnWindow;
    if (this.priceHistory.length < window) return 0;

    const recentPrices = this.priceHistory.slice(-window);
    const ma = recentPrices.reduce((sum, p) => sum + p, 0) / window;

    let aboveCount = 0;
    let belowCount = 0;

    for (const price of recentPrices) {
      if (price > ma) aboveCount++;
      else if (price < ma) belowCount++;
    }

    return Math.max(aboveCount, belowCount) / window;
  }

  /**
   * Calculate annualized volatility
   */
  private calculateVolatility(): number {
    const window = this.config.volatilityWindow;
    if (this.priceHistory.length < window + 1) return 0;

    const recentPrices = this.priceHistory.slice(-(window + 1));
    const returns: number[] = [];

    for (let i = 1; i < recentPrices.length; i++) {
      if (recentPrices[i - 1] !== 0) {
        returns.push((recentPrices[i] - recentPrices[i - 1]) / recentPrices[i - 1]);
      }
    }

    if (returns.length === 0) return 0;

    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    const stdDev = Math.sqrt(variance);

    return stdDev * Math.sqrt(365);
  }

  /**
   * Classify regime based on metrics
   */
  private classify(
    returns: number,
    trendStrength: number,
    volatility: number
  ): { regime: MarketRegime; confidence: number } {
    const { bullReturnThreshold, bearReturnThreshold, minTrendStrength } = this.config;

    // Strong uptrend = BULL
    if (returns > bullReturnThreshold && trendStrength >= minTrendStrength) {
      const returnConf = Math.min(1, returns / bullReturnThreshold);
      const confidence = (returnConf + trendStrength) / 2;
      return { regime: MarketRegime.BULL, confidence };
    }

    // Strong downtrend = BEAR
    if (returns < bearReturnThreshold && trendStrength >= minTrendStrength) {
      const returnConf = Math.min(1, Math.abs(returns) / Math.abs(bearReturnThreshold));
      const confidence = (returnConf + trendStrength) / 2;
      return { regime: MarketRegime.BEAR, confidence };
    }

    // Otherwise = SIDEWAYS
    const absReturn = Math.abs(returns);
    const maxThresh = Math.max(Math.abs(bullReturnThreshold), Math.abs(bearReturnThreshold));
    const returnConf = 1 - Math.min(1, absReturn / maxThresh);
    const confidence = (returnConf + (1 - trendStrength)) / 2;

    return { regime: MarketRegime.SIDEWAYS, confidence };
  }

  /**
   * Check if model is entering an untested regime
   */
  checkUntestedRegime(modelTrainingRegimes: { BULL: number; BEAR: number; SIDEWAYS: number }): {
    isUntested: boolean;
    warning: string | null;
    trainingPercentage: number;
  } {
    const currentRegime = this.currentRegime;

    if (currentRegime === MarketRegime.UNKNOWN) {
      return { isUntested: false, warning: null, trainingPercentage: 0 };
    }

    const total = modelTrainingRegimes.BULL + modelTrainingRegimes.BEAR + modelTrainingRegimes.SIDEWAYS;
    if (total === 0) {
      return { isUntested: true, warning: 'No training regime data available', trainingPercentage: 0 };
    }

    const regimeCount = modelTrainingRegimes[currentRegime] || 0;
    const percentage = (regimeCount / total) * 100;

    if (percentage < 10) {
      return {
        isUntested: true,
        warning: `Model has only ${percentage.toFixed(1)}% training data in ${currentRegime} regime`,
        trainingPercentage: percentage,
      };
    }

    return { isUntested: false, warning: null, trainingPercentage: percentage };
  }

  /**
   * Get regime-adjusted confidence threshold
   */
  getAdjustedConfidenceThreshold(
    baseThreshold: number,
    modelTrainingRegimes: { BULL: number; BEAR: number; SIDEWAYS: number }
  ): number {
    const { isUntested, trainingPercentage } = this.checkUntestedRegime(modelTrainingRegimes);

    if (isUntested) {
      const adjustment = 0.20 * (1 - trainingPercentage / 10);
      return Math.min(0.95, baseThreshold + adjustment);
    }

    return baseThreshold;
  }
}
