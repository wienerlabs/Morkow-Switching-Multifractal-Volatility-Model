/**
 * Global Risk Manager
 * 
 * Cross-strategy risk management with 6 critical controls:
 * 1. Drawdown Circuit Breakers (Daily 5%, Weekly 10%, Monthly 15%)
 * 2. Correlation Risk Tracking (Asset/Protocol exposure limits)
 * 3. Dynamic Stop Loss (ATR-based or asset class)
 * 4. Protocol Concentration Limit (Max 50% per protocol)
 * 5. Oracle Staleness Protection (30s reject, 60s emergency)
 * 6. Emergency Gas Budget (Real-time gas monitoring)
 * 
 * ALL DATA FROM REAL ON-CHAIN SOURCES - NO MOCK/PLACEHOLDER DATA
 */
import { Connection } from '@solana/web3.js';
import { logger } from '../logger.js';
import { getDb } from '../db/index.js';
import { getSolanaConnection } from '../solana/connection.js';
import { getPortfolioManager, type PortfolioManager } from '../portfolioManager.js';
import { OracleService, DEFAULT_ORACLE_CONFIG } from './oracleService.js';
import { GasService, DEFAULT_GAS_CONFIG, COMPUTE_UNITS } from './gasService.js';
import type Database from 'better-sqlite3';
import type {
  GlobalRiskConfig,
  GlobalRiskStatus,
  CircuitBreakerState,
  DrawdownStatus,
  DrawdownLimits,
  ExposureLimits,
  CorrelationRiskStatus,
  AssetExposure,
  ProtocolExposure,
  DynamicStopLossConfig,
  AssetVolatilityData,
  AssetClass,
  TrackedPosition,
  RiskAlert,
  OracleStatus,
  GasBudgetStatus,
} from './types.js';

// ============= DEFAULT CONFIGURATION =============

export const DEFAULT_DRAWDOWN_LIMITS: DrawdownLimits = {
  daily: 5.0,       // 5% daily → pause all strategies
  weekly: 10.0,     // 10% weekly → full stop
  monthly: 15.0,    // 15% monthly → system lockdown
};

export const DEFAULT_EXPOSURE_LIMITS: ExposureLimits = {
  maxBaseAssetPct: 40,     // Max 40% in single base asset
  maxQuoteAssetPct: 60,    // Max 60% in single quote asset
  maxProtocolPct: 50,      // Max 50% in single protocol
};

export const DEFAULT_STOP_LOSS_CONFIG: DynamicStopLossConfig = {
  majorStopPct: 3.0,       // BTC, ETH, SOL
  midcapStopPct: 5.0,      // Mid-cap tokens
  altStopPct: 7.0,         // Small-cap alts
  useATR: true,            // Prefer ATR-based stops
  atrMultiplier: 2.0,      // 2x ATR for stop distance
};

export const DEFAULT_GLOBAL_RISK_CONFIG: GlobalRiskConfig = {
  drawdownLimits: DEFAULT_DRAWDOWN_LIMITS,
  exposureLimits: DEFAULT_EXPOSURE_LIMITS,
  stopLossConfig: DEFAULT_STOP_LOSS_CONFIG,
  oracleConfig: DEFAULT_ORACLE_CONFIG,
  gasBudgetConfig: DEFAULT_GAS_CONFIG,
};

// Major assets for classification
const MAJOR_ASSETS = ['BTC', 'ETH', 'SOL', 'USDC', 'USDT'];
const MIDCAP_ASSETS = ['JUP', 'RAY', 'ORCA', 'mSOL', 'stSOL', 'jitoSOL', 'PYTH', 'JTO'];

// ============= GLOBAL RISK MANAGER CLASS =============

export class GlobalRiskManager {
  private config: GlobalRiskConfig;
  private portfolioManager: PortfolioManager;
  private oracleService: OracleService;
  private gasService: GasService;
  private connection: Connection;
  private db: Database.Database;

  // Drawdown tracking
  private peakValueUsd: number = 0;
  private dayStartValueUsd: number = 0;
  private weekStartValueUsd: number = 0;
  private monthStartValueUsd: number = 0;
  private dayStartTimestamp: number = 0;
  private weekStartTimestamp: number = 0;
  private monthStartTimestamp: number = 0;

  // Circuit breaker state
  private circuitBreakerState: CircuitBreakerState = 'ACTIVE';
  private lastCircuitBreakerTrigger?: Date;
  private circuitBreakerReason?: string;

  // Position tracking for exposure calculation
  private trackedPositions: Map<string, TrackedPosition> = new Map();

  // Alerts
  private alerts: RiskAlert[] = [];

  // Price cache for exposure calculations
  private priceCache: Map<string, number> = new Map();

  // Dry-run mode flag (allows simulation without wallet)
  private isDryRun: boolean = false;

  constructor(
    config: Partial<GlobalRiskConfig> = {},
    rpcUrl?: string,
    dryRun: boolean = false
  ) {
    this.config = { ...DEFAULT_GLOBAL_RISK_CONFIG, ...config };
    this.connection = rpcUrl
      ? new Connection(rpcUrl, 'confirmed')
      : getSolanaConnection();

    this.isDryRun = dryRun;
    this.db = getDb();
    this.portfolioManager = getPortfolioManager();
    this.oracleService = new OracleService(rpcUrl, this.config.oracleConfig);
    this.gasService = new GasService(rpcUrl, this.config.gasBudgetConfig);

    // Try to load persisted risk state first
    const loaded = this.loadRiskState();
    if (!loaded) {
      // No persisted state — initialize fresh period tracking
      this.initializePeriodTracking();
    }

    // Load persisted alerts
    this.loadAlerts();

    logger.info('GlobalRiskManager initialized', {
      drawdownLimits: this.config.drawdownLimits,
      exposureLimits: this.config.exposureLimits,
      dryRun: this.isDryRun,
      circuitBreaker: this.circuitBreakerState,
      restoredFromDb: loaded,
    });
  }

  // ============= RISK STATE PERSISTENCE (SQLite) =============

  private loadRiskState(): boolean {
    try {
      const row = this.db.prepare('SELECT * FROM risk_state WHERE id = 1').get() as any;
      if (!row) return false;

      this.circuitBreakerState = row.circuit_breaker_state as CircuitBreakerState;
      this.circuitBreakerReason = row.circuit_breaker_reason || undefined;
      this.lastCircuitBreakerTrigger = row.last_circuit_breaker_trigger
        ? new Date(row.last_circuit_breaker_trigger)
        : undefined;
      this.peakValueUsd = row.peak_value_usd;
      this.dayStartValueUsd = row.day_start_value_usd;
      this.weekStartValueUsd = row.week_start_value_usd;
      this.monthStartValueUsd = row.month_start_value_usd;
      this.dayStartTimestamp = row.day_start_timestamp;
      this.weekStartTimestamp = row.week_start_timestamp;
      this.monthStartTimestamp = row.month_start_timestamp;

      logger.info('Risk state restored from DB', {
        circuitBreaker: this.circuitBreakerState,
        peakValue: this.peakValueUsd,
      });
      return true;
    } catch (error) {
      logger.error('Failed to load risk state from DB', { error });
      return false;
    }
  }

  private persistRiskState(): void {
    try {
      this.db.prepare(`
        INSERT OR REPLACE INTO risk_state
        (id, circuit_breaker_state, circuit_breaker_reason, last_circuit_breaker_trigger,
         peak_value_usd, day_start_value_usd, week_start_value_usd, month_start_value_usd,
         day_start_timestamp, week_start_timestamp, month_start_timestamp)
        VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        this.circuitBreakerState,
        this.circuitBreakerReason || null,
        this.lastCircuitBreakerTrigger ? this.lastCircuitBreakerTrigger.getTime() : null,
        this.peakValueUsd,
        this.dayStartValueUsd,
        this.weekStartValueUsd,
        this.monthStartValueUsd,
        this.dayStartTimestamp,
        this.weekStartTimestamp,
        this.monthStartTimestamp,
      );
    } catch (error) {
      logger.error('Failed to persist risk state', { error });
    }
  }

  private persistAlert(alert: RiskAlert): void {
    try {
      this.db.prepare(`
        INSERT OR REPLACE INTO risk_alerts (id, timestamp, severity, category, message, data)
        VALUES (?, ?, ?, ?, ?, ?)
      `).run(
        alert.id,
        alert.timestamp.getTime(),
        alert.severity,
        alert.category,
        alert.message,
        alert.data ? JSON.stringify(alert.data) : null,
      );
    } catch (error) {
      logger.error('Failed to persist risk alert', { error });
    }
  }

  private loadAlerts(): void {
    try {
      const rows = this.db.prepare(
        'SELECT * FROM risk_alerts ORDER BY timestamp DESC LIMIT 100'
      ).all() as any[];
      this.alerts = rows.map(r => ({
        id: r.id,
        timestamp: new Date(r.timestamp),
        severity: r.severity,
        category: r.category,
        message: r.message,
        data: r.data ? JSON.parse(r.data) : undefined,
      })).reverse();
    } catch (error) {
      logger.error('Failed to load risk alerts from DB', { error });
    }
  }

  /**
   * Initialize period tracking for drawdown calculations
   */
  private initializePeriodTracking(): void {
    const now = Date.now();
    const currentValue = this.portfolioManager.getTotalValueUsd();

    // Initialize peak
    this.peakValueUsd = currentValue;

    // Initialize day start
    const dayStart = new Date(now);
    dayStart.setUTCHours(0, 0, 0, 0);
    this.dayStartTimestamp = dayStart.getTime();
    this.dayStartValueUsd = currentValue;

    // Initialize week start (Monday)
    const weekStart = new Date(now);
    weekStart.setUTCHours(0, 0, 0, 0);
    weekStart.setUTCDate(weekStart.getUTCDate() - weekStart.getUTCDay() + 1);
    this.weekStartTimestamp = weekStart.getTime();
    this.weekStartValueUsd = currentValue;

    // Initialize month start
    const monthStart = new Date(now);
    monthStart.setUTCHours(0, 0, 0, 0);
    monthStart.setUTCDate(1);
    this.monthStartTimestamp = monthStart.getTime();
    this.monthStartValueUsd = currentValue;

    this.persistRiskState();
  }

  /**
   * Check and reset period boundaries
   */
  private checkPeriodReset(): void {
    const now = Date.now();
    const currentValue = this.portfolioManager.getTotalValueUsd();

    let changed = false;

    // Check day reset
    const todayStart = new Date(now);
    todayStart.setUTCHours(0, 0, 0, 0);
    if (todayStart.getTime() > this.dayStartTimestamp) {
      this.dayStartTimestamp = todayStart.getTime();
      this.dayStartValueUsd = currentValue;
      changed = true;
      logger.info('Day reset - drawdown tracking reset');
    }

    // Check week reset (Monday)
    const thisWeekStart = new Date(now);
    thisWeekStart.setUTCHours(0, 0, 0, 0);
    thisWeekStart.setUTCDate(thisWeekStart.getUTCDate() - thisWeekStart.getUTCDay() + 1);
    if (thisWeekStart.getTime() > this.weekStartTimestamp) {
      this.weekStartTimestamp = thisWeekStart.getTime();
      this.weekStartValueUsd = currentValue;
      changed = true;
      logger.info('Week reset - drawdown tracking reset');
    }

    // Check month reset
    const thisMonthStart = new Date(now);
    thisMonthStart.setUTCHours(0, 0, 0, 0);
    thisMonthStart.setUTCDate(1);
    if (thisMonthStart.getTime() > this.monthStartTimestamp) {
      this.monthStartTimestamp = thisMonthStart.getTime();
      this.monthStartValueUsd = currentValue;
      changed = true;
      // Also reset circuit breaker on new month if in LOCKDOWN
      if (this.circuitBreakerState === 'LOCKDOWN') {
        this.circuitBreakerState = 'ACTIVE';
        logger.info('Month reset - circuit breaker reset from LOCKDOWN');
      }
    }

    if (changed) {
      this.persistRiskState();
    }
  }

  // ============= 1. DRAWDOWN CIRCUIT BREAKERS =============

  /**
   * Calculate current drawdown from REAL executed trades
   * Uses PortfolioManager for actual P&L data
   */
  calculateDrawdownStatus(): DrawdownStatus {
    this.checkPeriodReset();

    const currentValue = this.portfolioManager.getTotalValueUsd();

    // Update peak if we hit new high
    if (currentValue > this.peakValueUsd) {
      this.peakValueUsd = currentValue;
    }

    // Calculate drawdowns from period starts (from real executed trades)
    const dailyDrawdownPct = this.dayStartValueUsd > 0
      ? ((this.dayStartValueUsd - currentValue) / this.dayStartValueUsd) * 100
      : 0;

    const weeklyDrawdownPct = this.weekStartValueUsd > 0
      ? ((this.weekStartValueUsd - currentValue) / this.weekStartValueUsd) * 100
      : 0;

    const monthlyDrawdownPct = this.monthStartValueUsd > 0
      ? ((this.monthStartValueUsd - currentValue) / this.monthStartValueUsd) * 100
      : 0;

    // Determine circuit breaker state based on drawdowns
    let newState: CircuitBreakerState = 'ACTIVE';
    let triggerReason: string | undefined;

    if (monthlyDrawdownPct >= this.config.drawdownLimits.monthly) {
      newState = 'LOCKDOWN';
      triggerReason = `Monthly drawdown ${monthlyDrawdownPct.toFixed(2)}% >= ${this.config.drawdownLimits.monthly}%`;
    } else if (weeklyDrawdownPct >= this.config.drawdownLimits.weekly) {
      newState = 'STOPPED';
      triggerReason = `Weekly drawdown ${weeklyDrawdownPct.toFixed(2)}% >= ${this.config.drawdownLimits.weekly}%`;
    } else if (dailyDrawdownPct >= this.config.drawdownLimits.daily) {
      newState = 'PAUSED';
      triggerReason = `Daily drawdown ${dailyDrawdownPct.toFixed(2)}% >= ${this.config.drawdownLimits.daily}%`;
    }

    // Update peak value in DB
    if (currentValue > this.peakValueUsd) {
      this.persistRiskState();
    }

    // Update state if changed
    if (newState !== this.circuitBreakerState && newState !== 'ACTIVE') {
      this.circuitBreakerState = newState;
      this.lastCircuitBreakerTrigger = new Date();
      this.circuitBreakerReason = triggerReason;

      this.persistRiskState();

      this.addAlert({
        severity: newState === 'LOCKDOWN' ? 'emergency' : 'critical',
        category: 'drawdown',
        message: triggerReason || 'Circuit breaker triggered',
        data: { dailyDrawdownPct, weeklyDrawdownPct, monthlyDrawdownPct },
      });

      logger.warn('Circuit breaker triggered', {
        state: newState,
        reason: triggerReason,
        daily: dailyDrawdownPct.toFixed(2),
        weekly: weeklyDrawdownPct.toFixed(2),
        monthly: monthlyDrawdownPct.toFixed(2),
      });
    }

    return {
      dailyDrawdownPct,
      weeklyDrawdownPct,
      monthlyDrawdownPct,
      peakValueUsd: this.peakValueUsd,
      currentValueUsd: currentValue,
      circuitBreakerState: this.circuitBreakerState,
      lastTriggered: this.lastCircuitBreakerTrigger,
      triggerReason: this.circuitBreakerReason,
    };
  }

  /**
   * Check if trading is allowed based on circuit breaker state
   */
  isCircuitBreakerActive(): boolean {
    return this.circuitBreakerState !== 'ACTIVE';
  }

  /**
   * Get current circuit breaker state
   */
  getCircuitBreakerState(): CircuitBreakerState {
    return this.circuitBreakerState;
  }

  /**
   * Manually reset circuit breaker (requires confirmation)
   */
  resetCircuitBreaker(confirmPhrase: string): boolean {
    if (confirmPhrase !== 'CONFIRM_RESET') {
      logger.warn('Circuit breaker reset rejected - invalid confirmation');
      return false;
    }

    this.circuitBreakerState = 'ACTIVE';
    this.circuitBreakerReason = undefined;
    this.persistRiskState();
    logger.info('Circuit breaker manually reset');
    return true;
  }

  // ============= 2. CORRELATION RISK TRACKING =============

  /**
   * Update tracked positions from portfolio manager
   */
  syncPositionsFromPortfolio(): void {
    const portfolio = this.portfolioManager.getState();

    // Sync LP positions
    for (const [id, pos] of portfolio.lpPositions) {
      if (!pos.exitTime) {
        const tokens = pos.poolName.split('/');
        this.trackedPositions.set(id, {
          id,
          type: 'lp',
          protocol: pos.dex.toLowerCase(),
          baseAsset: tokens[0]?.toUpperCase() || 'UNKNOWN',
          quoteAsset: tokens[1]?.toUpperCase() || 'USDC',
          sizeUsd: pos.currentValueUsd,
          entryPrice: pos.entryPriceUsd,
          unrealizedPnlUsd: pos.currentValueUsd - pos.capitalUsd,
          entryTime: new Date(pos.entryTime),
        });
      } else {
        this.trackedPositions.delete(id);
      }
    }

    // Sync perps positions
    for (const [id, pos] of portfolio.perpsPositions) {
      if (!pos.exitTime) {
        this.trackedPositions.set(id, {
          id,
          type: 'perps',
          protocol: pos.venue.toLowerCase(),
          baseAsset: pos.market.replace('-PERP', '').toUpperCase(),
          quoteAsset: 'USDC',
          sizeUsd: pos.sizeUsd,
          entryPrice: pos.entryPrice,
          currentPrice: pos.currentPrice,
          unrealizedPnlUsd: pos.unrealizedPnlUsd,
          entryTime: new Date(pos.entryTime),
        });
      } else {
        this.trackedPositions.delete(id);
      }
    }
  }

  /**
   * Calculate correlation risk from REAL on-chain positions
   */
  calculateCorrelationRisk(): CorrelationRiskStatus {
    this.syncPositionsFromPortfolio();

    const totalValueUsd = this.portfolioManager.getTotalValueUsd();
    const baseExposures: Map<string, AssetExposure> = new Map();
    const quoteExposures: Map<string, AssetExposure> = new Map();
    const protocolExposures: Map<string, ProtocolExposure> = new Map();
    const violations: string[] = [];

    // Calculate exposures from all tracked positions
    for (const pos of this.trackedPositions.values()) {
      // Base asset exposure (e.g., SOL in SOL/USDC)
      if (!baseExposures.has(pos.baseAsset)) {
        baseExposures.set(pos.baseAsset, {
          asset: pos.baseAsset,
          exposureUsd: 0,
          exposurePct: 0,
          positions: [],
        });
      }
      const baseExp = baseExposures.get(pos.baseAsset)!;
      baseExp.exposureUsd += pos.sizeUsd / 2; // LP has 50% in each asset
      baseExp.positions.push(pos.id);

      // Quote asset exposure (e.g., USDC in SOL/USDC)
      if (!quoteExposures.has(pos.quoteAsset)) {
        quoteExposures.set(pos.quoteAsset, {
          asset: pos.quoteAsset,
          exposureUsd: 0,
          exposurePct: 0,
          positions: [],
        });
      }
      const quoteExp = quoteExposures.get(pos.quoteAsset)!;
      quoteExp.exposureUsd += pos.sizeUsd / 2;
      quoteExp.positions.push(pos.id);

      // Protocol exposure
      if (!protocolExposures.has(pos.protocol)) {
        protocolExposures.set(pos.protocol, {
          protocol: pos.protocol,
          exposureUsd: 0,
          exposurePct: 0,
          positions: [],
        });
      }
      const protoExp = protocolExposures.get(pos.protocol)!;
      protoExp.exposureUsd += pos.sizeUsd;
      protoExp.positions.push(pos.id);
    }

    // Calculate percentages and check violations
    for (const exp of baseExposures.values()) {
      exp.exposurePct = totalValueUsd > 0 ? (exp.exposureUsd / totalValueUsd) * 100 : 0;
      if (exp.exposurePct > this.config.exposureLimits.maxBaseAssetPct) {
        violations.push(`Base asset ${exp.asset}: ${exp.exposurePct.toFixed(1)}% > ${this.config.exposureLimits.maxBaseAssetPct}% limit`);
      }
    }

    for (const exp of quoteExposures.values()) {
      exp.exposurePct = totalValueUsd > 0 ? (exp.exposureUsd / totalValueUsd) * 100 : 0;
      if (exp.exposurePct > this.config.exposureLimits.maxQuoteAssetPct) {
        violations.push(`Quote asset ${exp.asset}: ${exp.exposurePct.toFixed(1)}% > ${this.config.exposureLimits.maxQuoteAssetPct}% limit`);
      }
    }

    for (const exp of protocolExposures.values()) {
      exp.exposurePct = totalValueUsd > 0 ? (exp.exposureUsd / totalValueUsd) * 100 : 0;
      if (exp.exposurePct > this.config.exposureLimits.maxProtocolPct) {
        violations.push(`Protocol ${exp.protocol}: ${exp.exposurePct.toFixed(1)}% > ${this.config.exposureLimits.maxProtocolPct}% limit`);
      }
    }

    return {
      baseAssetExposures: Array.from(baseExposures.values()),
      quoteAssetExposures: Array.from(quoteExposures.values()),
      protocolExposures: Array.from(protocolExposures.values()),
      violations,
      isCompliant: violations.length === 0,
    };
  }

  /**
   * Check if a new position would violate exposure limits
   */
  wouldViolateExposureLimits(position: {
    baseAsset: string;
    quoteAsset: string;
    protocol: string;
    sizeUsd: number;
  }): { allowed: boolean; violations: string[] } {
    const currentRisk = this.calculateCorrelationRisk();
    const totalValue = this.portfolioManager.getTotalValueUsd();
    const newTotalValue = totalValue + position.sizeUsd;
    const violations: string[] = [];

    // Check base asset
    const currentBase = currentRisk.baseAssetExposures.find(e => e.asset === position.baseAsset);
    const newBaseExposure = (currentBase?.exposureUsd || 0) + position.sizeUsd / 2;
    const newBasePct = (newBaseExposure / newTotalValue) * 100;
    if (newBasePct > this.config.exposureLimits.maxBaseAssetPct) {
      violations.push(`Would exceed ${position.baseAsset} limit: ${newBasePct.toFixed(1)}%`);
    }

    // Check quote asset
    const currentQuote = currentRisk.quoteAssetExposures.find(e => e.asset === position.quoteAsset);
    const newQuoteExposure = (currentQuote?.exposureUsd || 0) + position.sizeUsd / 2;
    const newQuotePct = (newQuoteExposure / newTotalValue) * 100;
    if (newQuotePct > this.config.exposureLimits.maxQuoteAssetPct) {
      violations.push(`Would exceed ${position.quoteAsset} limit: ${newQuotePct.toFixed(1)}%`);
    }

    // Check protocol
    const currentProto = currentRisk.protocolExposures.find(e => e.protocol === position.protocol);
    const newProtoExposure = (currentProto?.exposureUsd || 0) + position.sizeUsd;
    const newProtoPct = (newProtoExposure / newTotalValue) * 100;
    if (newProtoPct > this.config.exposureLimits.maxProtocolPct) {
      violations.push(`Would exceed ${position.protocol} limit: ${newProtoPct.toFixed(1)}%`);
    }

    return { allowed: violations.length === 0, violations };
  }

  // ============= HELPER: ADD ALERT =============

  private addAlert(params: Omit<RiskAlert, 'id' | 'timestamp'>): void {
    const alert: RiskAlert = {
      id: `alert_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
      timestamp: new Date(),
      ...params,
    };

    this.alerts.push(alert);
    this.persistAlert(alert);

    // Keep only last 100 alerts in memory
    if (this.alerts.length > 100) {
      this.alerts = this.alerts.slice(-100);
    }

    logger.warn('Risk alert', {
      severity: alert.severity,
      category: alert.category,
      message: alert.message
    });
  }

  /**
   * Get recent alerts
   */
  getAlerts(limit: number = 10): RiskAlert[] {
    return this.alerts.slice(-limit).reverse();
  }

  // ============= 3. DYNAMIC STOP LOSS =============

  /**
   * Classify asset by market cap/liquidity
   */
  classifyAsset(symbol: string): AssetClass {
    const upper = symbol.toUpperCase();
    if (MAJOR_ASSETS.includes(upper)) return 'major';
    if (MIDCAP_ASSETS.includes(upper)) return 'midcap';
    return 'alt';
  }

  /**
   * Get recommended stop loss for an asset
   * Uses asset classification and optional ATR data
   */
  getRecommendedStopLoss(symbol: string, atr24h?: number, currentPrice?: number): AssetVolatilityData {
    const classification = this.classifyAsset(symbol);

    // Base stop loss by asset class
    let baseStopPct: number;
    switch (classification) {
      case 'major':
        baseStopPct = this.config.stopLossConfig.majorStopPct;
        break;
      case 'midcap':
        baseStopPct = this.config.stopLossConfig.midcapStopPct;
        break;
      case 'alt':
        baseStopPct = this.config.stopLossConfig.altStopPct;
        break;
    }

    // If ATR is available and enabled, use ATR-based stop
    let recommendedStopPct = baseStopPct;
    if (this.config.stopLossConfig.useATR && atr24h && currentPrice && currentPrice > 0) {
      const atrPct = (atr24h / currentPrice) * 100;
      const atrBasedStop = atrPct * this.config.stopLossConfig.atrMultiplier;
      // Use the larger of ATR-based or class-based stop
      recommendedStopPct = Math.max(baseStopPct, atrBasedStop);
    }

    return {
      asset: symbol,
      atr24h,
      classification,
      recommendedStopPct,
    };
  }

  /**
   * Check if a position should be stopped out
   */
  shouldStopOut(position: TrackedPosition): { shouldStop: boolean; reason?: string } {
    if (!position.currentPrice || !position.entryPrice) {
      return { shouldStop: false };
    }

    const pnlPct = ((position.currentPrice - position.entryPrice) / position.entryPrice) * 100;
    const stopLossData = this.getRecommendedStopLoss(position.baseAsset);
    const stopThreshold = position.stopLossPct || stopLossData.recommendedStopPct;

    if (pnlPct <= -stopThreshold) {
      return {
        shouldStop: true,
        reason: `Position ${position.id} hit stop loss: ${pnlPct.toFixed(2)}% <= -${stopThreshold}%`,
      };
    }

    return { shouldStop: false };
  }

  // ============= 4. ORACLE STALENESS PROTECTION =============

  /**
   * Check oracle price with staleness validation
   */
  async checkOraclePrice(symbol: string): Promise<OracleStatus> {
    try {
      const status = await this.oracleService.getAggregatedPrice(symbol);

      // Cache the price for exposure calculations
      this.priceCache.set(symbol, status.price);

      // Update gas service with SOL price
      if (symbol.toUpperCase() === 'SOL') {
        this.gasService.setSolPrice(status.price);
      }

      // Alert on staleness
      if (status.isEmergency) {
        this.addAlert({
          severity: 'emergency',
          category: 'oracle',
          message: `Oracle emergency: ${symbol} price ${status.stalenessSeconds.toFixed(0)}s stale`,
          data: { symbol, stalenessSeconds: status.stalenessSeconds },
        });
      } else if (status.isStale) {
        this.addAlert({
          severity: 'warning',
          category: 'oracle',
          message: `Oracle stale: ${symbol} price ${status.stalenessSeconds.toFixed(0)}s old`,
          data: { symbol, stalenessSeconds: status.stalenessSeconds },
        });
      }

      return status;
    } catch (error) {
      this.addAlert({
        severity: 'critical',
        category: 'oracle',
        message: `Oracle failed for ${symbol}: ${error instanceof Error ? error.message : 'Unknown error'}`,
      });
      throw error;
    }
  }

  /**
   * Validate oracle before trade execution
   */
  async validateOracleForTrade(symbol: string): Promise<{ valid: boolean; reason?: string; price?: number }> {
    try {
      const status = await this.checkOraclePrice(symbol);

      if (status.isEmergency) {
        return {
          valid: false,
          reason: `Oracle emergency: ${symbol} price is ${status.stalenessSeconds.toFixed(0)}s stale (>${this.config.oracleConfig.emergencyExitSeconds}s)`,
        };
      }

      if (status.isStale) {
        return {
          valid: false,
          reason: `Oracle stale: ${symbol} price is ${status.stalenessSeconds.toFixed(0)}s old (>${this.config.oracleConfig.maxStalenessSeconds}s)`,
        };
      }

      return { valid: true, price: status.price };
    } catch (error) {
      return {
        valid: false,
        reason: `Oracle unavailable for ${symbol}`,
      };
    }
  }

  // ============= 5. EMERGENCY GAS BUDGET =============

  /**
   * Get current gas budget status
   */
  async getGasBudgetStatus(walletBalanceSol: number): Promise<{
    canTrade: boolean;
    canEmergencyExit: boolean;
    recommendedPriorityFee: number;
    reason?: string;
  }> {
    const status = await this.gasService.getGasBudgetStatus(walletBalanceSol);

    if (!status.canAffordEmergencyExit) {
      this.addAlert({
        severity: 'critical',
        category: 'gas',
        message: `Insufficient gas reserve for emergency exit. Balance: $${status.reserveBalanceUsd.toFixed(2)}`,
        data: { reserveBalanceUsd: status.reserveBalanceUsd },
      });

      return {
        canTrade: false,
        canEmergencyExit: false,
        recommendedPriorityFee: status.recommendedPriorityFee,
        reason: 'Insufficient gas reserve for emergency exit',
      };
    }

    return {
      canTrade: true,
      canEmergencyExit: true,
      recommendedPriorityFee: status.recommendedPriorityFee,
    };
  }

  // ============= 6. GLOBAL RISK CHECK =============

  /**
   * Comprehensive risk check before any trade
   * Returns whether trading is allowed and all block reasons
   */
  async performGlobalRiskCheck(params: {
    symbol: string;
    protocol: string;
    sizeUsd: number;
    walletBalanceSol: number;
  }): Promise<GlobalRiskStatus> {
    const blockReasons: string[] = [];
    const oracleStatuses = new Map<string, OracleStatus>();

    // 1. Check circuit breaker
    const drawdown = this.calculateDrawdownStatus();
    if (this.isCircuitBreakerActive()) {
      blockReasons.push(`Circuit breaker ${this.circuitBreakerState}: ${this.circuitBreakerReason}`);
    }

    // 2. Check correlation risk
    const correlationRisk = this.calculateCorrelationRisk();
    if (!correlationRisk.isCompliant) {
      blockReasons.push(...correlationRisk.violations);
    }

    // 3. Check if new position would violate limits
    const exposureCheck = this.wouldViolateExposureLimits({
      baseAsset: params.symbol,
      quoteAsset: 'USDC',
      protocol: params.protocol,
      sizeUsd: params.sizeUsd,
    });
    if (!exposureCheck.allowed) {
      blockReasons.push(...exposureCheck.violations);
    }

    // 4. Check oracle staleness
    try {
      const oracleStatus = await this.checkOraclePrice(params.symbol);
      oracleStatuses.set(params.symbol, oracleStatus);
      if (oracleStatus.isStale || oracleStatus.isEmergency) {
        if (this.isDryRun) {
          logger.warn(`[DRY-RUN] Oracle stale for ${params.symbol}: ${oracleStatus.stalenessSeconds.toFixed(0)}s - allowing for simulation`);
        } else {
          blockReasons.push(`Oracle stale for ${params.symbol}: ${oracleStatus.stalenessSeconds.toFixed(0)}s`);
        }
      }
    } catch (error) {
      if (this.isDryRun) {
        logger.warn(`[DRY-RUN] Oracle unavailable for ${params.symbol} - allowing for simulation`);
      } else {
        blockReasons.push(`Oracle unavailable for ${params.symbol}`);
      }
    }

    // 5. Check gas budget
    const gasBudget = await this.gasService.getGasBudgetStatus(params.walletBalanceSol);
    if (!gasBudget.canAffordEmergencyExit) {
      blockReasons.push('Insufficient gas reserve for emergency exit');
    }

    const canTrade = blockReasons.length === 0;

    if (!canTrade) {
      logger.warn('Trade blocked by risk manager', {
        symbol: params.symbol,
        reasons: blockReasons
      });
    }

    return {
      timestamp: new Date(),
      circuitBreakerState: this.circuitBreakerState,
      drawdown,
      correlationRisk,
      oracleStatuses,
      gasBudget,
      canTrade,
      blockReasons,
    };
  }

  /**
   * Get full risk status summary
   */
  async getRiskSummary(walletBalanceSol: number): Promise<{
    circuitBreaker: CircuitBreakerState;
    drawdown: DrawdownStatus;
    correlationRisk: CorrelationRiskStatus;
    gasBudget: GasBudgetStatus;
    alerts: RiskAlert[];
    canTrade: boolean;
  }> {
    const drawdown = this.calculateDrawdownStatus();
    const correlationRisk = this.calculateCorrelationRisk();
    const gasBudget = await this.gasService.getGasBudgetStatus(walletBalanceSol);

    const canTrade =
      !this.isCircuitBreakerActive() &&
      correlationRisk.isCompliant &&
      gasBudget.canAffordEmergencyExit;

    return {
      circuitBreaker: this.circuitBreakerState,
      drawdown,
      correlationRisk,
      gasBudget,
      alerts: this.getAlerts(10),
      canTrade,
    };
  }
}

// ============= SINGLETON INSTANCE =============

let globalRiskManagerInstance: GlobalRiskManager | null = null;

export function getGlobalRiskManager(
  config?: Partial<GlobalRiskConfig>,
  rpcUrl?: string,
  dryRun?: boolean
): GlobalRiskManager {
  if (!globalRiskManagerInstance) {
    globalRiskManagerInstance = new GlobalRiskManager(config, rpcUrl, dryRun);
  }
  return globalRiskManagerInstance;
}

export function resetGlobalRiskManager(): void {
  globalRiskManagerInstance = null;
}

