/**
 * ML-Powered Perps Trading Agent
 *
 * Autonomous trading agent that:
 * 1. Fetches funding rate data every 60 seconds
 * 2. Extracts features and runs ML inference
 * 3. Executes trades when model predicts profitable opportunities
 *
 * Uses the trained XGBoost model to identify funding rate arbitrage opportunities
 * with >0.25% funding rate threshold for profitability.
 *
 * INTEGRATED: GlobalRiskManager for cross-strategy risk checks
 */
import { logger } from '../../logger.js';
import { getPerpsModelLoader, type PredictionResult, type ModelConfig } from './modelLoader.js';
import { PerpsFeatureExtractor, createFeatureExtractor, type FundingDataPoint } from './featureExtractor.js';
import { perpsFileLogger } from './fileLogger.js';
import { pnlTracker } from './pnlTracker.js';
import { getGlobalRiskManager } from '../../risk/index.js';
import { getPortfolioManager } from '../../portfolioManager.js';
import { getSentimentIntegration, type SentimentAdjustedScore } from '../../sentiment/sentimentIntegration.js';
import type { PerpsService } from '../perpsService.js';
import type { FundingRate, PerpsVenue, PositionSide, PerpsTradeResult } from '../../../types/perps.js';

// ============= TYPES =============

export interface TradingAgentConfig {
  /** Default polling interval in milliseconds (default: 10000 = 10 seconds) */
  pollingIntervalMs: number;
  /** Jupiter-specific polling interval (default: 5000 = 5 seconds) - no native SL/TP */
  jupiterPollingIntervalMs: number;
  /** Drift-specific polling interval (default: 30000 = 30 seconds) - has on-chain SL/TP */
  driftPollingIntervalMs: number;
  /** Adrena-specific polling interval (default: 30000 = 30 seconds) - has on-chain SL/TP */
  adrenaPollingIntervalMs: number;
  /** Flash-specific polling interval (default: 60000 = 60 seconds) - simulated */
  flashPollingIntervalMs: number;
  /** Minimum confidence for trading (default: 0.6) */
  minConfidence: number;
  /** Minimum funding rate threshold (default: 0.0025 = 0.25%) */
  fundingThreshold: number;
  /** Position size in USD (default: 1000) */
  positionSizeUsd: number;
  /** Default leverage (default: 1) */
  leverage: number;
  /** Markets to monitor (default: ['SOL-PERP']) */
  markets: string[];
  /** Preferred venue (default: 'drift') */
  preferredVenue: PerpsVenue;
  /** Enable dry run mode (no actual trades) */
  dryRun: boolean;
  /** Maximum concurrent positions */
  maxPositions: number;
  /** Maximum retries for failed position closes */
  maxCloseRetries: number;
  /** Default stop loss percentage (default: 0.05 = 5%) - used when not specified per position */
  defaultStopLossPercent: number;
  /** Default take profit percentage (default: 0.10 = 10%) - used when not specified per position */
  defaultTakeProfitPercent: number;
}

export interface TradeSignal {
  market: string;
  prediction: PredictionResult;
  fundingRate: number;
  timestamp: Date;
  executed: boolean;
  tradeResult?: PerpsTradeResult;
  /** Sentiment-adjusted score (if available) */
  sentimentAdjustment?: SentimentAdjustedScore;
}

/** Tracked position with entry context for close logic */
export interface TrackedPosition {
  id: string;
  market: string;
  side: PositionSide;
  entryPrice: number;
  entryTime: Date;
  entryFundingRate: number;
  size: number;
  orderId?: string;
  portfolioPositionId?: string; // PortfolioManager position ID for tracking
  venue: PerpsVenue; // Venue for venue-specific monitoring
  /** Stop loss price (actual price, not percentage) */
  stopLossPrice?: number;
  /** Take profit price (actual price, not percentage) */
  takeProfitPrice?: number;
  /** Leverage for priority calculation */
  leverage: number;
}

/** Position close reason */
export type CloseReason = 'funding_reversal' | 'max_hold_time' | 'stop_loss' | 'take_profit' | 'manual';

export interface AgentState {
  running: boolean;
  lastScan: Date | null;
  scansCompleted: number;
  tradesExecuted: number;
  positionsClosed: number;
  signalsGenerated: TradeSignal[];
  activePositions: number;
  trackedPositions: TrackedPosition[];
  totalPnL: number;
}

// ============= DEFAULT CONFIG =============

// Position management constants
const MAX_HOLD_HOURS = 8;           // Max hours to hold position (1 funding cycle)

const DEFAULT_CONFIG: TradingAgentConfig = {
  pollingIntervalMs: 10000,         // 10 seconds (was 60s) - faster monitoring
  jupiterPollingIntervalMs: 5000,   // 5 seconds - Jupiter has no native SL/TP
  driftPollingIntervalMs: 30000,    // 30 seconds - Drift has on-chain SL/TP
  adrenaPollingIntervalMs: 30000,   // 30 seconds - Adrena has on-chain SL/TP
  flashPollingIntervalMs: 60000,    // 60 seconds - Flash is simulated
  minConfidence: 0.6,
  fundingThreshold: 0.0025,        // 0.25%
  positionSizeUsd: 1000,
  leverage: 1,
  markets: ['SOL-PERP'],
  preferredVenue: 'drift',
  dryRun: true,                    // Default to dry run for safety
  maxPositions: 3,
  maxCloseRetries: 3,              // Retry failed closes up to 3 times
  defaultStopLossPercent: 0.05,    // -5% stop loss
  defaultTakeProfitPercent: 0.10,  // +10% take profit
};

// ============= TRADING AGENT =============

export class PerpsTradingAgent {
  private config: TradingAgentConfig;
  private perpsService: PerpsService | null = null;
  private featureExtractors: Map<string, PerpsFeatureExtractor> = new Map();
  private state: AgentState;
  private intervalId: NodeJS.Timeout | null = null;
  private initialized = false;
  /** Tracks last check time per position for venue-specific intervals */
  private positionLastCheckTime: Map<string, number> = new Map();

  constructor(config: Partial<TradingAgentConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.state = {
      running: false,
      lastScan: null,
      scansCompleted: 0,
      tradesExecuted: 0,
      positionsClosed: 0,
      signalsGenerated: [],
      activePositions: 0,
      trackedPositions: [],
      totalPnL: 0,
    };

    // Initialize feature extractors for each market
    for (const market of this.config.markets) {
      this.featureExtractors.set(market, createFeatureExtractor());
    }

    logger.info('PerpsTradingAgent created', { config: this.config });
  }

  /** Set the perps service for trade execution */
  setPerpsService(service: PerpsService): void {
    this.perpsService = service;
  }

  /** Initialize the agent (load model) */
  async initialize(): Promise<boolean> {
    try {
      const modelConfig: Partial<ModelConfig> = {
        minConfidence: this.config.minConfidence,
        fundingThreshold: this.config.fundingThreshold,
      };

      const modelLoader = getPerpsModelLoader(modelConfig);
      const success = await modelLoader.initialize();

      if (!success) {
        logger.error('Failed to initialize ML model');
        return false;
      }

      this.initialized = true;
      logger.info('PerpsTradingAgent initialized');
      return true;

    } catch (error) {
      logger.error('Failed to initialize trading agent', { error });
      return false;
    }
  }

  /** Start the trading loop */
  async start(): Promise<void> {
    if (!this.initialized) {
      throw new Error('Agent not initialized. Call initialize() first.');
    }

    if (this.state.running) {
      logger.warn('Trading agent already running');
      return;
    }

    this.state.running = true;

    // Log venue-specific intervals for visibility
    logger.info('Starting trading agent with venue-specific monitoring', {
      baseIntervalMs: this.config.pollingIntervalMs,
      venueIntervals: {
        jupiter: this.config.jupiterPollingIntervalMs,
        drift: this.config.driftPollingIntervalMs,
        adrena: this.config.adrenaPollingIntervalMs,
        flash: this.config.flashPollingIntervalMs,
      },
      markets: this.config.markets,
      dryRun: this.config.dryRun,
      maxCloseRetries: this.config.maxCloseRetries,
      defaultStopLossPercent: `${(this.config.defaultStopLossPercent * 100).toFixed(1)}%`,
      defaultTakeProfitPercent: `${(this.config.defaultTakeProfitPercent * 100).toFixed(1)}%`,
    });

    // Run immediately
    await this.runScanCycle();

    // Set up polling - use base interval, venue-specific logic in managePositions
    // Base interval determines how often we check, venue-specific intervals determine
    // which positions get checked each cycle
    this.intervalId = setInterval(async () => {
      await this.runScanCycle();
    }, this.config.pollingIntervalMs);
  }

  /** Stop the trading loop */
  stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    this.state.running = false;
    logger.info('Trading agent stopped');
  }

  /** Get current agent state */
  getState(): AgentState {
    return { ...this.state };
  }

  /** Run a single scan cycle */
  private async runScanCycle(): Promise<void> {
    try {
      logger.debug('Running scan cycle...');
      this.state.lastScan = new Date();

      // First, check existing positions for close conditions
      await this.managePositions();

      // Then scan for new trading opportunities
      for (const market of this.config.markets) {
        await this.scanMarket(market);
      }

      this.state.scansCompleted++;

    } catch (error) {
      logger.error('Scan cycle failed', { error });
    }
  }

  /** Scan a single market for trading opportunities */
  private async scanMarket(market: string): Promise<void> {
    const extractor = this.featureExtractors.get(market);
    if (!extractor) {
      logger.warn('No feature extractor for market', { market });
      return;
    }

    // Fetch latest funding rate
    const fundingRate = await this.fetchFundingRate(market);
    if (fundingRate === null) {
      return;
    }

    // Add to buffer
    const dataPoint: FundingDataPoint = {
      timestamp: new Date(),
      fundingRate: fundingRate.rate,
      fundingRateRaw: fundingRate.rate,
      // Note: FundingRate type doesn't include prices, would need to fetch separately
    };
    extractor.addDataPoint(dataPoint);

    // Check if we have enough history
    if (!extractor.hasEnoughHistory()) {
      logger.debug('Not enough history for prediction', {
        market,
        bufferSize: extractor.getBufferSize(),
      });
      return;
    }

    // Extract features and predict
    const features = extractor.extractFeatures();
    const modelLoader = getPerpsModelLoader();
    const prediction = await modelLoader.predict(features, fundingRate.rate);

    // ========== SENTIMENT INTEGRATION (25% weight for perps) ==========
    // Extract base symbol from market (e.g., 'SOL-PERP' -> 'SOL')
    const baseSymbol = market.split('-')[0] || market;
    const sentimentIntegration = getSentimentIntegration();
    const sentimentAdjustment = await sentimentIntegration.getAdjustedScore(
      baseSymbol,
      prediction.confidence,
      'perps'
    );

    // Create adjusted prediction with sentiment-modified confidence
    const adjustedPrediction: PredictionResult = {
      ...prediction,
      confidence: sentimentAdjustment.finalScore,
      // Re-evaluate shouldTrade with adjusted confidence
      shouldTrade: prediction.shouldTrade && sentimentAdjustment.finalScore >= this.config.minConfidence,
    };

    logger.debug('ML prediction with sentiment', {
      market,
      fundingRate: fundingRate.rate,
      prediction: adjustedPrediction.prediction,
      probability: adjustedPrediction.probability,
      mlConfidence: prediction.confidence,
      sentimentSignal: sentimentAdjustment.signal,
      adjustedConfidence: adjustedPrediction.confidence,
      shouldTrade: adjustedPrediction.shouldTrade,
      sentimentReasoning: sentimentAdjustment.reasoning,
    });
    // ==================================================================

    // Log prediction to file (with sentiment data)
    perpsFileLogger.logPrediction({
      timestamp: new Date().toISOString(),
      market,
      fundingRate: fundingRate.rate,
      prediction: adjustedPrediction.prediction,
      probability: adjustedPrediction.probability,
      direction: adjustedPrediction.direction || 'short',
      shouldTrade: adjustedPrediction.shouldTrade,
      confidence: adjustedPrediction.confidence,
      featureCount: features.length,
    });

    // Record signal with sentiment adjustment
    const signal: TradeSignal = {
      market,
      prediction: adjustedPrediction,
      fundingRate: fundingRate.rate,
      timestamp: new Date(),
      executed: false,
      sentimentAdjustment,
    };

    if (adjustedPrediction.shouldTrade) {
      this.state.signalsGenerated.push(signal);

      // Execute trade if conditions are met
      if (this.shouldExecuteTrade(adjustedPrediction)) {
        await this.executeTrade(market, adjustedPrediction, signal);
      }
    }
  }

  /** Fetch funding rate for a market */
  private async fetchFundingRate(market: string): Promise<FundingRate | null> {
    if (!this.perpsService) {
      logger.warn('PerpsService not set, cannot fetch funding rates');
      return null;
    }

    try {
      const rates = await this.perpsService.getFundingRates();
      // Filter by preferred venue and market
      const rate = rates.find(r =>
        r.market === market && r.venue === this.config.preferredVenue
      );
      return rate || null;
    } catch (error) {
      logger.error('Failed to fetch funding rate', { market, error });
      return null;
    }
  }

  /** Check if we should execute a trade */
  private shouldExecuteTrade(prediction: PredictionResult): boolean {
    // Check position limits
    if (this.state.activePositions >= this.config.maxPositions) {
      logger.debug('Max positions reached, skipping trade');
      return false;
    }

    // Check dry run mode
    if (this.config.dryRun) {
      logger.info('DRY RUN: Would execute trade', { prediction });
      return false;
    }

    return prediction.shouldTrade;
  }

  /** Execute a trade based on prediction */
  private async executeTrade(
    market: string,
    prediction: PredictionResult,
    signal: TradeSignal
  ): Promise<void> {
    if (!this.perpsService) {
      logger.error('Cannot execute trade: PerpsService not set');
      return;
    }

    const side: PositionSide = prediction.direction === 'long' ? 'long' : 'short';

    // ========== GLOBAL RISK CHECK ==========
    // Extract base symbol from market (e.g., 'SOL-PERP' -> 'SOL')
    const baseSymbol = market.split('-')[0] || market;
    const riskManager = getGlobalRiskManager();
    // Get actual wallet balance from perps service for accurate risk checks
    const walletBalanceSol = this.perpsService
      ? await this.perpsService.getWalletBalance()
      : 1;
    const riskCheck = await riskManager.performGlobalRiskCheck({
      symbol: baseSymbol,
      protocol: this.config.preferredVenue,
      sizeUsd: this.config.positionSizeUsd,
      walletBalanceSol,
      strategyType: 'perps',
    });

    if (!riskCheck.canTrade) {
      logger.warn('Perps trade blocked by risk manager', {
        market,
        side,
        reasons: riskCheck.blockReasons,
      });
      perpsFileLogger.logError({
        timestamp: new Date().toISOString(),
        category: 'trade',
        message: 'Trade blocked by GlobalRiskManager',
        error: riskCheck.blockReasons.join('; '),
        context: { market, side, circuitBreakerState: riskCheck.circuitBreakerState },
      });
      return;
    }
    // ========================================

    // Apply A-LAMS regime-based position scaling
    const regimeScale = riskCheck.alamsVar?.regimePositionScale ?? 1.0;
    const scaledSize = this.config.positionSizeUsd * regimeScale;

    logger.info('Executing ML-predicted trade', {
      market,
      side,
      size: scaledSize,
      originalSize: this.config.positionSizeUsd,
      regimeScale,
      confidence: prediction.confidence,
      fundingRate: signal.fundingRate,
      riskStatus: riskCheck.circuitBreakerState,
    });

    try {
      const result = await this.perpsService.openPosition({
        venue: this.config.preferredVenue,
        market,
        side,
        size: scaledSize,
        leverage: this.config.leverage,
      });

      signal.executed = result.success;
      signal.tradeResult = result;

      if (result.success) {
        this.state.tradesExecuted++;
        this.state.activePositions++;

        // ========== TRACK POSITION IN PORTFOLIO ==========
        const portfolioManager = getPortfolioManager();
        const portfolioPositionId = portfolioManager.openPerpsPosition({
          venue: this.config.preferredVenue,
          market,
          side,
          leverage: this.config.leverage,
          sizeUsd: this.config.positionSizeUsd,
          collateralUsd: this.config.positionSizeUsd / this.config.leverage,
          entryPrice: result.entryPrice || 0,
        });
        // ================================================

        // Track the position for management
        const entryPrice = result.entryPrice || 0;

        // Calculate SL/TP prices based on entry price and configured percentages
        const { stopLossPrice, takeProfitPrice } = this.calculateSlTpPrices(
          entryPrice,
          side,
          this.config.defaultStopLossPercent,
          this.config.defaultTakeProfitPercent
        );

        const trackedPosition: TrackedPosition = {
          id: result.orderId || `${market}-${Date.now()}`,
          market,
          side,
          entryPrice,
          entryTime: new Date(),
          entryFundingRate: signal.fundingRate,
          size: this.config.positionSizeUsd,
          orderId: result.orderId,
          portfolioPositionId, // Store portfolio position ID
          venue: this.config.preferredVenue,
          stopLossPrice,
          takeProfitPrice,
          leverage: this.config.leverage,
        };
        this.state.trackedPositions.push(trackedPosition);

        logger.debug('Position SL/TP prices calculated', {
          positionId: trackedPosition.id,
          entryPrice,
          side,
          stopLossPrice,
          takeProfitPrice,
        });

        // Log trade to file
        perpsFileLogger.logTrade({
          timestamp: new Date().toISOString(),
          type: 'open',
          market,
          side,
          sizeUsd: this.config.positionSizeUsd,
          entryPrice: result.entryPrice,
          fundingRate: signal.fundingRate,
          confidence: prediction.confidence,
          orderId: result.orderId,
          venue: this.config.preferredVenue,
          dryRun: this.config.dryRun,
        });

        logger.info('Trade executed successfully', {
          orderId: result.orderId,
          entryPrice: result.entryPrice,
          trackedPositionId: trackedPosition.id,
          portfolioPositionId,
        });
      } else {
        logger.error('Trade execution failed', { error: result.error });
        perpsFileLogger.logError({
          timestamp: new Date().toISOString(),
          category: 'trade',
          message: 'Trade execution failed',
          error: result.error,
          context: { market, side, size: this.config.positionSizeUsd },
        });
      }

    } catch (error) {
      logger.error('Trade execution threw error', { error });
      perpsFileLogger.logError({
        timestamp: new Date().toISOString(),
        category: 'trade',
        message: 'Trade execution threw exception',
        error: String(error),
        stack: error instanceof Error ? error.stack : undefined,
        context: { market },
      });
    }
  }

  /** Load historical funding rate data for warm-up */
  async loadHistoricalData(market: string, data: FundingDataPoint[]): Promise<void> {
    const extractor = this.featureExtractors.get(market);
    if (extractor) {
      extractor.loadHistory(data);
      logger.info('Loaded historical data', { market, dataPoints: data.length });
    }
  }

  /** Get recent signals for a market */
  getRecentSignals(market?: string, limit = 10): TradeSignal[] {
    let signals = this.state.signalsGenerated;
    if (market) {
      signals = signals.filter(s => s.market === market);
    }
    return signals.slice(-limit);
  }

  // ============= POSITION MANAGEMENT =============

  /** Manage existing positions - check for close conditions using venue-specific intervals */
  private async managePositions(): Promise<void> {
    if (this.state.trackedPositions.length === 0) return;

    const now = Date.now();
    const positionsToClose: Array<{ position: TrackedPosition; reason: CloseReason; pnlPercent: number }> = [];
    const positionsToCheck: TrackedPosition[] = [];

    // Determine which positions need checking based on venue-specific intervals
    for (const position of this.state.trackedPositions) {
      const lastCheckTime = this.positionLastCheckTime.get(position.id) || 0;
      const venueInterval = this.getVenuePollingInterval(position.venue);
      const timeSinceLastCheck = now - lastCheckTime;

      // Check if enough time has passed for this venue's interval
      if (timeSinceLastCheck >= venueInterval) {
        positionsToCheck.push(position);
      }
    }

    // Log venue-specific monitoring status
    if (positionsToCheck.length > 0) {
      const venueBreakdown = positionsToCheck.reduce((acc, p) => {
        acc[p.venue] = (acc[p.venue] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);

      logger.debug('Checking positions with venue-specific intervals', {
        totalPositions: this.state.trackedPositions.length,
        checkingNow: positionsToCheck.length,
        venueBreakdown,
      });
    }

    // Check positions that are due for review
    for (const position of positionsToCheck) {
      // Update last check time
      this.positionLastCheckTime.set(position.id, now);

      const closeInfo = await this.shouldClosePosition(position);
      if (closeInfo) {
        positionsToClose.push({ position, ...closeInfo });
      }
    }

    // Close positions
    for (const { position, reason, pnlPercent } of positionsToClose) {
      await this.closePosition(position, reason, pnlPercent);
      // Clean up last check time for closed positions
      this.positionLastCheckTime.delete(position.id);
    }
  }

  /** Check if a position should be closed */
  private async shouldClosePosition(position: TrackedPosition): Promise<{ reason: CloseReason; pnlPercent: number } | null> {
    const now = new Date();
    const holdHours = (now.getTime() - position.entryTime.getTime()) / (1000 * 60 * 60);

    // Get current funding rate and price
    const fundingRate = await this.fetchFundingRate(position.market);
    const currentFunding = fundingRate?.rate ?? 0;

    // Get current price from positions
    let currentPrice = position.entryPrice;
    let pnlPercent = 0;

    if (this.perpsService) {
      try {
        const positions = await this.perpsService.getAllPositions();
        const livePosition = positions.find(p =>
          p.market === position.market && p.side === position.side
        );
        if (livePosition) {
          currentPrice = livePosition.markPrice;
          pnlPercent = livePosition.unrealizedPnlPct;
        } else {
          // Position might have been liquidated or closed externally
          logger.warn('Tracked position not found on-chain, removing', { position });
          return { reason: 'manual', pnlPercent: 0 };
        }
      } catch (e) {
        // Use estimate
        if (position.entryPrice > 0) {
          const priceChange = (currentPrice - position.entryPrice) / position.entryPrice;
          pnlPercent = position.side === 'long' ? priceChange : -priceChange;
        }
      }
    }

    // Check close conditions in priority order

    // 1. Stop loss - check against position-specific price if available
    if (position.stopLossPrice !== undefined) {
      // Price-based stop loss check
      const stopLossHit = position.side === 'long'
        ? currentPrice <= position.stopLossPrice
        : currentPrice >= position.stopLossPrice;

      if (stopLossHit) {
        logger.info('Stop loss triggered (price-based)', {
          position: position.id,
          currentPrice,
          stopLossPrice: position.stopLossPrice,
          pnlPercent,
        });
        return { reason: 'stop_loss', pnlPercent };
      }
    } else {
      // Fallback to percentage-based check using config default
      if (pnlPercent <= -this.config.defaultStopLossPercent) {
        logger.info('Stop loss triggered (percentage-based)', { position: position.id, pnlPercent });
        return { reason: 'stop_loss', pnlPercent };
      }
    }

    // 2. Take profit - check against position-specific price if available
    if (position.takeProfitPrice !== undefined) {
      // Price-based take profit check
      const takeProfitHit = position.side === 'long'
        ? currentPrice >= position.takeProfitPrice
        : currentPrice <= position.takeProfitPrice;

      if (takeProfitHit) {
        logger.info('Take profit triggered (price-based)', {
          position: position.id,
          currentPrice,
          takeProfitPrice: position.takeProfitPrice,
          pnlPercent,
        });
        return { reason: 'take_profit', pnlPercent };
      }
    } else {
      // Fallback to percentage-based check using config default
      if (pnlPercent >= this.config.defaultTakeProfitPercent) {
        logger.info('Take profit triggered (percentage-based)', { position: position.id, pnlPercent });
        return { reason: 'take_profit', pnlPercent };
      }
    }

    // 3. Funding rate reversal
    const fundingReversed = this.checkFundingReversal(position, currentFunding);
    if (fundingReversed) {
      logger.info('Funding rate reversal detected', {
        position: position.id,
        entryFunding: position.entryFundingRate,
        currentFunding,
      });
      return { reason: 'funding_reversal', pnlPercent };
    }

    // 4. Max hold time (8 hours = 1 funding cycle)
    if (holdHours >= MAX_HOLD_HOURS) {
      logger.info('Max hold time reached', { position: position.id, holdHours });
      return { reason: 'max_hold_time', pnlPercent };
    }

    return null;
  }

  /** Check if funding rate has reversed against position */
  private checkFundingReversal(position: TrackedPosition, currentFunding: number): boolean {
    // For SHORT positions (we entered because funding was positive/high)
    // Close if funding goes negative (longs are now paying us, no edge)
    if (position.side === 'short') {
      return position.entryFundingRate > 0 && currentFunding < 0;
    }

    // For LONG positions (we entered because funding was negative/high)
    // Close if funding goes positive (shorts are now paying, we'd pay)
    if (position.side === 'long') {
      return position.entryFundingRate < 0 && currentFunding > 0;
    }

    return false;
  }

  /** Calculate stop loss and take profit prices based on entry price and percentages */
  private calculateSlTpPrices(
    entryPrice: number,
    side: PositionSide,
    stopLossPercent: number,
    takeProfitPercent: number
  ): { stopLossPrice: number; takeProfitPrice: number } {
    if (side === 'long') {
      // For long: SL is below entry, TP is above entry
      return {
        stopLossPrice: entryPrice * (1 - stopLossPercent),
        takeProfitPrice: entryPrice * (1 + takeProfitPercent),
      };
    } else {
      // For short: SL is above entry, TP is below entry
      return {
        stopLossPrice: entryPrice * (1 + stopLossPercent),
        takeProfitPrice: entryPrice * (1 - takeProfitPercent),
      };
    }
  }

  /** Get venue-specific monitoring interval */
  private getVenuePollingInterval(venue: PerpsVenue): number {
    switch (venue) {
      case 'jupiter':
        return this.config.jupiterPollingIntervalMs;
      case 'drift':
        return this.config.driftPollingIntervalMs;
      case 'adrena':
        return this.config.adrenaPollingIntervalMs;
      case 'flash':
        return this.config.flashPollingIntervalMs;
      default:
        return this.config.pollingIntervalMs;
    }
  }

  /** Close a position with retry logic */
  private async closePosition(position: TrackedPosition, reason: CloseReason, pnlPercent: number): Promise<void> {
    if (!this.perpsService) {
      logger.error('Cannot close position: PerpsService not set');
      return;
    }

    const holdTimeHours = (Date.now() - position.entryTime.getTime()) / (1000 * 60 * 60);

    logger.info('Closing position', {
      positionId: position.id,
      market: position.market,
      side: position.side,
      reason,
      pnlPercent: (pnlPercent * 100).toFixed(2) + '%',
      holdTime: holdTimeHours.toFixed(1) + 'h',
    });

    // Get current price for logging
    const fundingRate = await this.fetchFundingRate(position.market);
    const currentFunding = fundingRate?.rate ?? 0;

    if (this.config.dryRun) {
      logger.info('DRY RUN: Would close position', { position, reason });

      // Log trade close
      perpsFileLogger.logTrade({
        timestamp: new Date().toISOString(),
        type: 'close',
        market: position.market,
        side: position.side,
        sizeUsd: position.size,
        entryPrice: position.entryPrice,
        exitPrice: position.entryPrice * (1 + pnlPercent), // Estimate
        fundingRate: currentFunding,
        confidence: 0,
        closeReason: reason,
        pnlUsd: position.size * pnlPercent,
        pnlPercent,
        holdTimeHours,
        orderId: position.orderId,
        venue: this.config.preferredVenue,
        dryRun: true,
      });

      // Record in P&L tracker
      pnlTracker.recordTrade(
        position,
        position.entryPrice * (1 + pnlPercent),
        pnlPercent,
        reason,
        0,
        this.config.preferredVenue
      );

      this.removeTrackedPosition(position.id, pnlPercent);
      return;
    }

    // Retry loop with exponential backoff
    const maxRetries = this.config.maxCloseRetries;
    let lastError: unknown = null;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const result = await this.perpsService.closePosition({
          venue: position.venue,
          market: position.market,
        });

        if (result.success) {
          const exitPrice = result.entryPrice || position.entryPrice * (1 + pnlPercent);

          // ========== CLOSE POSITION IN PORTFOLIO ==========
          if (position.portfolioPositionId) {
            const portfolioManager = getPortfolioManager();
            const portfolioPnl = portfolioManager.closePerpsPosition(position.portfolioPositionId, exitPrice);
            logger.info('Position closed in portfolio', {
              portfolioPositionId: position.portfolioPositionId,
              exitPrice,
              portfolioPnl,
            });
          }
          // ================================================

          // Log trade close
          perpsFileLogger.logTrade({
            timestamp: new Date().toISOString(),
            type: 'close',
            market: position.market,
            side: position.side,
            sizeUsd: position.size,
            entryPrice: position.entryPrice,
            exitPrice,
            fundingRate: currentFunding,
            confidence: 0,
            closeReason: reason,
            pnlUsd: position.size * pnlPercent,
            pnlPercent,
            holdTimeHours,
            orderId: result.orderId,
            venue: position.venue,
            dryRun: false,
          });

          // Record in P&L tracker
          pnlTracker.recordTrade(position, exitPrice, pnlPercent, reason, 0, position.venue);

          this.removeTrackedPosition(position.id, pnlPercent);
          logger.info('Position closed successfully', {
            positionId: position.id,
            reason,
            estimatedPnL: (position.size * pnlPercent).toFixed(2),
            attempt,
          });
          return; // Success - exit retry loop
        } else {
          lastError = result.error;
          logger.warn(`Failed to close position (attempt ${attempt}/${maxRetries})`, {
            error: result.error,
            positionId: position.id,
          });
        }
      } catch (error) {
        lastError = error;
        logger.warn(`Error closing position (attempt ${attempt}/${maxRetries})`, {
          error: String(error),
          positionId: position.id,
        });
      }

      // Wait before retry with exponential backoff (2s, 4s, 8s, ...)
      if (attempt < maxRetries) {
        const delayMs = Math.pow(2, attempt) * 1000;
        logger.info(`Retrying position close in ${delayMs}ms`, {
          positionId: position.id,
          attempt,
          maxRetries,
        });
        await new Promise(resolve => setTimeout(resolve, delayMs));
      }
    }

    // All retries failed - log critical error
    logger.error('CRITICAL: Failed to close position after max retries', {
      positionId: position.id,
      market: position.market,
      reason,
      maxRetries,
      lastError: String(lastError),
    });
    perpsFileLogger.logError({
      timestamp: new Date().toISOString(),
      category: 'trade',
      message: 'CRITICAL: Failed to close position after max retries',
      error: String(lastError),
      stack: lastError instanceof Error ? lastError.stack : undefined,
      context: {
        positionId: position.id,
        market: position.market,
        reason,
        maxRetries,
        venue: position.venue,
      },
    });
  }

  /** Remove a tracked position and update stats */
  private removeTrackedPosition(positionId: string, pnlPercent: number): void {
    const idx = this.state.trackedPositions.findIndex(p => p.id === positionId);
    if (idx >= 0) {
      const position = this.state.trackedPositions[idx];
      const pnl = position.size * pnlPercent;

      this.state.trackedPositions.splice(idx, 1);
      this.state.activePositions = Math.max(0, this.state.activePositions - 1);
      this.state.positionsClosed++;
      this.state.totalPnL += pnl;

      // Clean up venue-specific interval tracking
      this.positionLastCheckTime.delete(positionId);

      logger.info('Position removed from tracking', {
        positionId,
        pnl: pnl.toFixed(2),
        totalPnL: this.state.totalPnL.toFixed(2),
        remainingPositions: this.state.trackedPositions.length,
      });
    }
  }

  /** Get P&L statistics */
  getPnLStats() {
    return pnlTracker.getStats();
  }

  /** Get recent trades from file logger */
  getRecentTradesFromLog(limit = 5) {
    return perpsFileLogger.getRecentTrades(limit);
  }

  /** Get recent predictions from file logger */
  getRecentPredictionsFromLog(limit = 5) {
    return perpsFileLogger.getRecentPredictions(limit);
  }
}

// Export factory function
export function createTradingAgent(config?: Partial<TradingAgentConfig>): PerpsTradingAgent {
  return new PerpsTradingAgent(config);
}
