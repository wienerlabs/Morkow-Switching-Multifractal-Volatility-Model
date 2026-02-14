/**
 * Portfolio Manager
 *
 * Tracks wallet balances, positions, trade history, and PnL.
 * Persists state to JSON file for crash recovery.
 *
 * Features:
 * - Multi-asset balance tracking (SOL, USDC, tokens)
 * - LP position management with entry price/time
 * - Trade history with full execution details
 * - Realized + unrealized PnL calculation
 * - JSON file persistence (no DB dependency)
 *
 * Created: 2026-01-07
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { logger } from './logger.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Default portfolio state file location
const DEFAULT_STATE_FILE = path.join(__dirname, '../../data/portfolio_state.json');

// ============= TYPES =============

export interface TokenBalance {
  symbol: string;
  mint: string;
  balance: number;
  valueUsd: number;
  lastUpdated: number;
}

export interface LPPosition {
  id: string;
  poolAddress: string;
  poolName: string;
  dex: string;
  token0: string;
  token1: string;

  // Entry details
  entryTime: number;
  entryPriceUsd: number;
  entryApy: number;
  capitalUsd: number;

  // Current state
  currentValueUsd: number;
  feesEarnedUsd: number;
  impermanentLossUsd: number;

  // Exit details (null if still open)
  exitTime?: number;
  exitPriceUsd?: number;
  realizedPnlUsd?: number;
}

export interface PerpsPositionRecord {
  id: string;
  venue: string;
  market: string;
  side: 'long' | 'short';
  leverage: number;

  // Entry details
  entryTime: number;
  entryPrice: number;
  sizeUsd: number;
  collateralUsd: number;

  // Current state
  currentPrice?: number;
  unrealizedPnlUsd: number;
  fundingPaidUsd: number;

  // Exit details (null if still open)
  exitTime?: number;
  exitPrice?: number;
  realizedPnlUsd?: number;
}

export interface TradeRecord {
  id: string;
  type: 'arbitrage' | 'lp_entry' | 'lp_exit' | 'perps_open' | 'perps_close' | 'swap';
  timestamp: number;

  // What was traded
  asset: string;
  side: 'buy' | 'sell' | 'deposit' | 'withdraw';
  amountUsd: number;

  // Execution details
  venue: string;
  txSignature?: string;
  fees: number;
  slippage?: number;

  // Result
  pnlUsd: number;
  mlConfidence?: number;
  notes?: string;
}

export interface PortfolioState {
  version: number;
  lastUpdated: number;

  // Balances
  balances: Map<string, TokenBalance>;
  totalValueUsd: number;

  // Positions
  lpPositions: Map<string, LPPosition>;
  perpsPositions: Map<string, PerpsPositionRecord>;

  // History & Stats
  trades: TradeRecord[];

  // PnL
  realizedPnlUsd: number;
  unrealizedPnlUsd: number;
  totalFeesUsd: number;

  // Daily tracking
  dailyPnlUsd: number;
  dailyTradeCount: number;
  dayStartTimestamp: number;
}

// Serializable version for JSON persistence
interface PortfolioStateJSON {
  version: number;
  lastUpdated: number;
  balances: Record<string, TokenBalance>;
  totalValueUsd: number;
  lpPositions: Record<string, LPPosition>;
  perpsPositions: Record<string, PerpsPositionRecord>;
  trades: TradeRecord[];
  realizedPnlUsd: number;
  unrealizedPnlUsd: number;
  totalFeesUsd: number;
  dailyPnlUsd: number;
  dailyTradeCount: number;
  dayStartTimestamp: number;
}

// ============= PORTFOLIO MANAGER CLASS =============

export class PortfolioManager {
  private state: PortfolioState;
  private stateFilePath: string;
  private autoSave: boolean;
  private saveDebounceTimer: NodeJS.Timeout | null = null;

  constructor(options: {
    stateFilePath?: string;
    autoSave?: boolean;
    initialCapitalUsd?: number;
  } = {}) {
    this.stateFilePath = options.stateFilePath || DEFAULT_STATE_FILE;
    this.autoSave = options.autoSave ?? true;

    // Try to load existing state, or create new
    const loaded = this.loadState();

    if (loaded) {
      this.state = loaded;
      logger.info('PortfolioManager loaded existing state', {
        totalValue: this.state.totalValueUsd,
        lpPositions: this.state.lpPositions.size,
        trades: this.state.trades.length,
      });
    } else {
      this.state = this.createInitialState(options.initialCapitalUsd || 10000);
      logger.info('PortfolioManager created new state', {
        initialCapital: options.initialCapitalUsd || 10000,
      });
    }

    // Check if we need to reset daily counters
    this.checkDailyReset();
  }

  // ============= STATE PERSISTENCE =============

  private createInitialState(initialCapitalUsd: number): PortfolioState {
    const now = Date.now();
    const state: PortfolioState = {
      version: 1,
      lastUpdated: now,
      balances: new Map(),
      totalValueUsd: initialCapitalUsd,
      lpPositions: new Map(),
      perpsPositions: new Map(),
      trades: [],
      realizedPnlUsd: 0,
      unrealizedPnlUsd: 0,
      totalFeesUsd: 0,
      dailyPnlUsd: 0,
      dailyTradeCount: 0,
      dayStartTimestamp: this.getDayStart(now),
    };

    // Initialize with USDC balance
    state.balances.set('USDC', {
      symbol: 'USDC',
      mint: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
      balance: initialCapitalUsd,
      valueUsd: initialCapitalUsd,
      lastUpdated: now,
    });

    return state;
  }

  private getDayStart(timestamp: number): number {
    const date = new Date(timestamp);
    date.setUTCHours(0, 0, 0, 0);
    return date.getTime();
  }

  private checkDailyReset(): void {
    const todayStart = this.getDayStart(Date.now());
    if (this.state.dayStartTimestamp < todayStart) {
      // New day - reset daily counters
      this.state.dailyPnlUsd = 0;
      this.state.dailyTradeCount = 0;
      this.state.dayStartTimestamp = todayStart;
      logger.info('Daily counters reset for new day');
    }
  }

  private loadState(): PortfolioState | null {
    try {
      if (!fs.existsSync(this.stateFilePath)) {
        return null;
      }

      const json = fs.readFileSync(this.stateFilePath, 'utf-8');
      const data: PortfolioStateJSON = JSON.parse(json);

      // Convert from JSON to PortfolioState (Maps)
      return {
        version: data.version,
        lastUpdated: data.lastUpdated,
        balances: new Map(Object.entries(data.balances)),
        totalValueUsd: data.totalValueUsd,
        lpPositions: new Map(Object.entries(data.lpPositions)),
        perpsPositions: new Map(Object.entries(data.perpsPositions)),
        trades: data.trades,
        realizedPnlUsd: data.realizedPnlUsd,
        unrealizedPnlUsd: data.unrealizedPnlUsd,
        totalFeesUsd: data.totalFeesUsd,
        dailyPnlUsd: data.dailyPnlUsd,
        dailyTradeCount: data.dailyTradeCount,
        dayStartTimestamp: data.dayStartTimestamp,
      };
    } catch (error) {
      logger.error('Failed to load portfolio state', { error, path: this.stateFilePath });
      return null;
    }
  }

  saveState(): void {
    try {
      // Ensure directory exists
      const dir = path.dirname(this.stateFilePath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }

      // Convert Maps to objects for JSON
      const data: PortfolioStateJSON = {
        version: this.state.version,
        lastUpdated: Date.now(),
        balances: Object.fromEntries(this.state.balances),
        totalValueUsd: this.state.totalValueUsd,
        lpPositions: Object.fromEntries(this.state.lpPositions),
        perpsPositions: Object.fromEntries(this.state.perpsPositions),
        trades: this.state.trades,
        realizedPnlUsd: this.state.realizedPnlUsd,
        unrealizedPnlUsd: this.state.unrealizedPnlUsd,
        totalFeesUsd: this.state.totalFeesUsd,
        dailyPnlUsd: this.state.dailyPnlUsd,
        dailyTradeCount: this.state.dailyTradeCount,
        dayStartTimestamp: this.state.dayStartTimestamp,
      };

      const tmpPath = this.stateFilePath + '.tmp';
      fs.writeFileSync(tmpPath, JSON.stringify(data, null, 2));
      fs.renameSync(tmpPath, this.stateFilePath);
      logger.info('Portfolio state saved', { path: this.stateFilePath });
    } catch (error) {
      logger.error('Failed to save portfolio state', { error });
    }
  }

  private scheduleSave(): void {
    if (!this.autoSave) return;

    // Debounce saves to avoid excessive writes
    if (this.saveDebounceTimer) {
      clearTimeout(this.saveDebounceTimer);
    }
    this.saveDebounceTimer = setTimeout(() => {
      this.saveState();
    }, 1000);
  }

  // ============= BALANCE MANAGEMENT =============

  updateBalance(symbol: string, mint: string, balance: number, priceUsd: number): void {
    const tokenBalance: TokenBalance = {
      symbol,
      mint,
      balance,
      valueUsd: balance * priceUsd,
      lastUpdated: Date.now(),
    };
    this.state.balances.set(symbol, tokenBalance);
    this.recalculateTotalValue();
    this.scheduleSave();
  }

  getBalance(symbol: string): TokenBalance | undefined {
    return this.state.balances.get(symbol);
  }

  getAllBalances(): TokenBalance[] {
    return Array.from(this.state.balances.values());
  }

  private recalculateTotalValue(): void {
    let total = 0;
    for (const balance of this.state.balances.values()) {
      total += balance.valueUsd;
    }
    // Add LP position values
    for (const position of this.state.lpPositions.values()) {
      if (!position.exitTime) {
        total += position.currentValueUsd;
      }
    }
    // Add perps position values
    for (const position of this.state.perpsPositions.values()) {
      if (!position.exitTime) {
        total += position.collateralUsd + position.unrealizedPnlUsd;
      }
    }
    this.state.totalValueUsd = total;
  }

  // ============= LP POSITION MANAGEMENT =============

  openLPPosition(params: {
    poolAddress: string;
    poolName: string;
    dex: string;
    token0: string;
    token1: string;
    capitalUsd: number;
    entryApy: number;
  }): string {
    const id = `lp_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const position: LPPosition = {
      id,
      poolAddress: params.poolAddress,
      poolName: params.poolName,
      dex: params.dex,
      token0: params.token0,
      token1: params.token1,
      entryTime: Date.now(),
      entryPriceUsd: params.capitalUsd,
      entryApy: params.entryApy,
      capitalUsd: params.capitalUsd,
      currentValueUsd: params.capitalUsd,
      feesEarnedUsd: 0,
      impermanentLossUsd: 0,
    };

    this.state.lpPositions.set(id, position);

    // Record trade
    this.recordTrade({
      type: 'lp_entry',
      asset: params.poolName,
      side: 'deposit',
      amountUsd: params.capitalUsd,
      venue: params.dex,
      fees: 0,
      pnlUsd: 0,
    });

    this.scheduleSave();
    return id;
  }

  updateLPPosition(id: string, update: {
    currentValueUsd?: number;
    feesEarnedUsd?: number;
    impermanentLossUsd?: number;
  }): void {
    const position = this.state.lpPositions.get(id);
    if (!position) return;

    if (update.currentValueUsd !== undefined) position.currentValueUsd = update.currentValueUsd;
    if (update.feesEarnedUsd !== undefined) position.feesEarnedUsd = update.feesEarnedUsd;
    if (update.impermanentLossUsd !== undefined) position.impermanentLossUsd = update.impermanentLossUsd;

    // Update unrealized PnL
    this.state.unrealizedPnlUsd = this.calculateUnrealizedPnl();
    this.recalculateTotalValue();
    this.scheduleSave();
  }

  closeLPPosition(id: string, exitValueUsd: number): number {
    const position = this.state.lpPositions.get(id);
    if (!position) return 0;

    position.exitTime = Date.now();
    position.exitPriceUsd = exitValueUsd;
    position.realizedPnlUsd = exitValueUsd - position.capitalUsd;

    // Update realized PnL
    this.state.realizedPnlUsd += position.realizedPnlUsd;
    this.state.dailyPnlUsd += position.realizedPnlUsd;
    this.state.dailyTradeCount += 1;

    // Record trade
    this.recordTrade({
      type: 'lp_exit',
      asset: position.poolName,
      side: 'withdraw',
      amountUsd: exitValueUsd,
      venue: position.dex,
      fees: 0,
      pnlUsd: position.realizedPnlUsd,
    });

    this.state.unrealizedPnlUsd = this.calculateUnrealizedPnl();
    this.recalculateTotalValue();
    this.scheduleSave();

    return position.realizedPnlUsd;
  }

  getLPPosition(id: string): LPPosition | undefined {
    return this.state.lpPositions.get(id);
  }

  getOpenLPPositions(): LPPosition[] {
    return Array.from(this.state.lpPositions.values()).filter(p => !p.exitTime);
  }

  // ============= PERPS POSITION MANAGEMENT =============

  openPerpsPosition(params: {
    venue: string;
    market: string;
    side: 'long' | 'short';
    leverage: number;
    sizeUsd: number;
    collateralUsd: number;
    entryPrice: number;
  }): string {
    const id = `perps_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const position: PerpsPositionRecord = {
      id,
      venue: params.venue,
      market: params.market,
      side: params.side,
      leverage: params.leverage,
      entryTime: Date.now(),
      entryPrice: params.entryPrice,
      sizeUsd: params.sizeUsd,
      collateralUsd: params.collateralUsd,
      currentPrice: params.entryPrice,
      unrealizedPnlUsd: 0,
      fundingPaidUsd: 0,
    };

    this.state.perpsPositions.set(id, position);

    // Record trade
    this.recordTrade({
      type: 'perps_open',
      asset: params.market,
      side: params.side === 'long' ? 'buy' : 'sell',
      amountUsd: params.sizeUsd,
      venue: params.venue,
      fees: 0,
      pnlUsd: 0,
    });

    this.recalculateTotalValue();
    this.scheduleSave();
    logger.info('Perps position opened', { id, market: params.market, side: params.side, sizeUsd: params.sizeUsd });
    return id;
  }

  updatePerpsPosition(id: string, update: {
    currentPrice?: number;
    unrealizedPnlUsd?: number;
    fundingPaidUsd?: number;
  }): void {
    const position = this.state.perpsPositions.get(id);
    if (!position) return;

    if (update.currentPrice !== undefined) position.currentPrice = update.currentPrice;
    if (update.unrealizedPnlUsd !== undefined) position.unrealizedPnlUsd = update.unrealizedPnlUsd;
    if (update.fundingPaidUsd !== undefined) position.fundingPaidUsd = update.fundingPaidUsd;

    this.state.unrealizedPnlUsd = this.calculateUnrealizedPnl();
    this.recalculateTotalValue();
    this.scheduleSave();
  }

  closePerpsPosition(id: string, exitPrice: number, fees: number = 0): number {
    const position = this.state.perpsPositions.get(id);
    if (!position) return 0;

    position.exitTime = Date.now();
    position.exitPrice = exitPrice;

    // Calculate realized PnL
    const priceDiff = exitPrice - position.entryPrice;
    const direction = position.side === 'long' ? 1 : -1;
    position.realizedPnlUsd = (priceDiff / position.entryPrice) * position.sizeUsd * direction - position.fundingPaidUsd - fees;

    // Update state
    this.state.realizedPnlUsd += position.realizedPnlUsd;
    this.state.dailyPnlUsd += position.realizedPnlUsd;
    this.state.dailyTradeCount += 1;

    // Record trade
    this.recordTrade({
      type: 'perps_close',
      asset: position.market,
      side: position.side === 'long' ? 'sell' : 'buy',
      amountUsd: position.sizeUsd,
      venue: position.venue,
      fees,
      pnlUsd: position.realizedPnlUsd,
    });

    this.state.unrealizedPnlUsd = this.calculateUnrealizedPnl();
    this.recalculateTotalValue();
    this.scheduleSave();

    logger.info('Perps position closed', { id, pnlUsd: position.realizedPnlUsd });
    return position.realizedPnlUsd;
  }

  getPerpsPosition(id: string): PerpsPositionRecord | undefined {
    return this.state.perpsPositions.get(id);
  }

  getOpenPerpsPositions(): PerpsPositionRecord[] {
    return Array.from(this.state.perpsPositions.values()).filter(p => !p.exitTime);
  }

  // ============= TRADE HISTORY =============

  private recordTrade(params: {
    type: TradeRecord['type'];
    asset: string;
    side: TradeRecord['side'];
    amountUsd: number;
    venue: string;
    fees: number;
    pnlUsd: number;
    txSignature?: string;
    mlConfidence?: number;
    notes?: string;
  }): void {
    const id = `trade_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const trade: TradeRecord = {
      id,
      timestamp: Date.now(),
      ...params,
    };

    this.state.trades.push(trade);
    this.state.totalFeesUsd += params.fees;

    // Keep only last 1000 trades to prevent unbounded growth
    if (this.state.trades.length > 1000) {
      this.state.trades = this.state.trades.slice(-1000);
    }
  }

  /**
   * Public method for recording arbitrage trades
   */
  recordArbitrageTrade(params: {
    asset: string;
    amountUsd: number;
    venue: string;
    fees: number;
    pnlUsd: number;
    txSignature?: string;
    notes?: string;
  }): void {
    this.recordTrade({
      type: 'arbitrage',
      side: 'buy', // Arbitrage involves both buy and sell
      ...params,
    });

    // Update realized PnL
    this.state.realizedPnlUsd += params.pnlUsd;
    this.state.dailyPnlUsd += params.pnlUsd;
    this.state.dailyTradeCount += 1;

    this.scheduleSave();
    logger.info('Arbitrage trade recorded', { asset: params.asset, pnlUsd: params.pnlUsd });
  }

  getRecentTrades(limit = 50): TradeRecord[] {
    return this.state.trades.slice(-limit).reverse();
  }

  // ============= PNL CALCULATIONS =============

  private calculateUnrealizedPnl(): number {
    let unrealized = 0;

    // LP positions
    for (const pos of this.state.lpPositions.values()) {
      if (!pos.exitTime) {
        unrealized += pos.currentValueUsd - pos.capitalUsd;
      }
    }

    // Perps positions
    for (const pos of this.state.perpsPositions.values()) {
      if (!pos.exitTime) {
        unrealized += pos.unrealizedPnlUsd;
      }
    }

    return unrealized;
  }

  // ============= GETTERS =============

  getTotalValueUsd(): number {
    return this.state.totalValueUsd;
  }

  getRealizedPnlUsd(): number {
    return this.state.realizedPnlUsd;
  }

  getUnrealizedPnlUsd(): number {
    return this.state.unrealizedPnlUsd;
  }

  getDailyStats(): { pnlUsd: number; tradeCount: number } {
    return {
      pnlUsd: this.state.dailyPnlUsd,
      tradeCount: this.state.dailyTradeCount,
    };
  }

  getState(): PortfolioState {
    return this.state;
  }

  // ============= SUMMARY =============

  getSummary(): {
    totalValueUsd: number;
    realizedPnlUsd: number;
    unrealizedPnlUsd: number;
    totalPnlUsd: number;
    openLpPositions: number;
    openPerpsPositions: number;
    totalTrades: number;
    dailyPnlUsd: number;
  } {
    return {
      totalValueUsd: this.state.totalValueUsd,
      realizedPnlUsd: this.state.realizedPnlUsd,
      unrealizedPnlUsd: this.state.unrealizedPnlUsd,
      totalPnlUsd: this.state.realizedPnlUsd + this.state.unrealizedPnlUsd,
      openLpPositions: this.getOpenLPPositions().length,
      openPerpsPositions: Array.from(this.state.perpsPositions.values()).filter(p => !p.exitTime).length,
      totalTrades: this.state.trades.length,
      dailyPnlUsd: this.state.dailyPnlUsd,
    };
  }
}

// ============= SINGLETON =============

let portfolioManagerInstance: PortfolioManager | null = null;

export function getPortfolioManager(options?: {
  stateFilePath?: string;
  autoSave?: boolean;
  initialCapitalUsd?: number;
}): PortfolioManager {
  if (!portfolioManagerInstance) {
    portfolioManagerInstance = new PortfolioManager(options);
  }
  return portfolioManagerInstance;
}

