/**
 * Portfolio Manager
 *
 * Tracks wallet balances, positions, trade history, and PnL.
 * Persists state to SQLite for crash-safe recovery.
 */

import { logger } from './logger.js';
import { getDb } from './db/index.js';
import type Database from 'better-sqlite3';

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
  entryTime: number;
  entryPriceUsd: number;
  entryApy: number;
  capitalUsd: number;
  currentValueUsd: number;
  feesEarnedUsd: number;
  impermanentLossUsd: number;
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
  entryTime: number;
  entryPrice: number;
  sizeUsd: number;
  collateralUsd: number;
  currentPrice?: number;
  unrealizedPnlUsd: number;
  fundingPaidUsd: number;
  exitTime?: number;
  exitPrice?: number;
  realizedPnlUsd?: number;
}

export interface TradeRecord {
  id: string;
  type: 'arbitrage' | 'lp_entry' | 'lp_exit' | 'perps_open' | 'perps_close' | 'swap';
  timestamp: number;
  asset: string;
  side: 'buy' | 'sell' | 'deposit' | 'withdraw';
  amountUsd: number;
  venue: string;
  txSignature?: string;
  fees: number;
  slippage?: number;
  pnlUsd: number;
  mlConfidence?: number;
  notes?: string;
}

export interface PortfolioState {
  version: number;
  lastUpdated: number;
  balances: Map<string, TokenBalance>;
  totalValueUsd: number;
  lpPositions: Map<string, LPPosition>;
  perpsPositions: Map<string, PerpsPositionRecord>;
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
  private db: Database.Database;

  constructor(options: {
    initialCapitalUsd?: number;
    dbPath?: string;
  } = {}) {
    this.db = getDb(options.dbPath);

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
      this.persistPortfolioState();
      logger.info('PortfolioManager created new state', {
        initialCapital: options.initialCapitalUsd || 10000,
      });
    }

    this.checkDailyReset();
  }

  // ============= STATE PERSISTENCE (SQLite) =============

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
      this.state.dailyPnlUsd = 0;
      this.state.dailyTradeCount = 0;
      this.state.dayStartTimestamp = todayStart;
      this.persistPortfolioState();
      logger.info('Daily counters reset for new day');
    }
  }

  private loadState(): PortfolioState | null {
    try {
      const row = this.db.prepare('SELECT * FROM portfolio_state WHERE id = 1').get() as any;
      if (!row) return null;

      const balances = new Map<string, TokenBalance>();
      const balanceRows = this.db.prepare('SELECT * FROM balances').all() as any[];
      for (const b of balanceRows) {
        balances.set(b.symbol, {
          symbol: b.symbol,
          mint: b.mint,
          balance: b.balance,
          valueUsd: b.value_usd,
          lastUpdated: b.last_updated,
        });
      }

      const lpPositions = new Map<string, LPPosition>();
      const perpsPositions = new Map<string, PerpsPositionRecord>();
      const positionRows = this.db.prepare('SELECT * FROM positions').all() as any[];
      for (const p of positionRows) {
        const data = JSON.parse(p.data);
        if (p.type === 'lp') {
          lpPositions.set(p.id, data);
        } else {
          perpsPositions.set(p.id, data);
        }
      }

      const tradeRows = this.db.prepare(
        'SELECT * FROM trades ORDER BY timestamp DESC LIMIT 200'
      ).all() as any[];
      const trades: TradeRecord[] = tradeRows.map(t => ({
        id: t.id,
        type: t.type,
        timestamp: t.timestamp,
        asset: t.asset,
        side: t.side,
        amountUsd: t.amount_usd,
        venue: t.venue,
        txSignature: t.tx_signature || undefined,
        fees: t.fees,
        slippage: t.slippage ?? undefined,
        pnlUsd: t.pnl_usd,
        mlConfidence: t.ml_confidence ?? undefined,
        notes: t.notes || undefined,
      })).reverse();

      return {
        version: row.version,
        lastUpdated: row.last_updated,
        balances,
        totalValueUsd: row.total_value_usd,
        lpPositions,
        perpsPositions,
        trades,
        realizedPnlUsd: row.realized_pnl_usd,
        unrealizedPnlUsd: row.unrealized_pnl_usd,
        totalFeesUsd: row.total_fees_usd,
        dailyPnlUsd: row.daily_pnl_usd,
        dailyTradeCount: row.daily_trade_count,
        dayStartTimestamp: row.day_start_timestamp,
      };
    } catch (error) {
      logger.error('Failed to load portfolio state from DB', { error });
      return null;
    }
  }

  private persistPortfolioState(): void {
    try {
      this.db.prepare(`
        INSERT OR REPLACE INTO portfolio_state
        (id, version, last_updated, total_value_usd, realized_pnl_usd, unrealized_pnl_usd,
         total_fees_usd, daily_pnl_usd, daily_trade_count, day_start_timestamp)
        VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        this.state.version,
        Date.now(),
        this.state.totalValueUsd,
        this.state.realizedPnlUsd,
        this.state.unrealizedPnlUsd,
        this.state.totalFeesUsd,
        this.state.dailyPnlUsd,
        this.state.dailyTradeCount,
        this.state.dayStartTimestamp,
      );
    } catch (error) {
      logger.error('Failed to persist portfolio state', { error });
    }
  }

  private persistBalance(b: TokenBalance): void {
    this.db.prepare(`
      INSERT OR REPLACE INTO balances (symbol, mint, balance, value_usd, last_updated)
      VALUES (?, ?, ?, ?, ?)
    `).run(b.symbol, b.mint, b.balance, b.valueUsd, b.lastUpdated);
  }

  private persistPosition(type: 'lp' | 'perps', pos: LPPosition | PerpsPositionRecord): void {
    const isOpen = !('exitTime' in pos && pos.exitTime);
    this.db.prepare(`
      INSERT OR REPLACE INTO positions (id, type, data, is_open, created_at, closed_at)
      VALUES (?, ?, ?, ?, ?, ?)
    `).run(
      pos.id,
      type,
      JSON.stringify(pos),
      isOpen ? 1 : 0,
      'entryTime' in pos ? pos.entryTime : Date.now(),
      'exitTime' in pos && pos.exitTime ? pos.exitTime : null,
    );
  }

  private persistTrade(trade: TradeRecord): void {
    this.db.prepare(`
      INSERT OR REPLACE INTO trades
      (id, type, timestamp, asset, side, amount_usd, venue, tx_signature, fees, slippage, pnl_usd, ml_confidence, notes)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      trade.id, trade.type, trade.timestamp, trade.asset, trade.side,
      trade.amountUsd, trade.venue, trade.txSignature || null,
      trade.fees, trade.slippage ?? null, trade.pnlUsd,
      trade.mlConfidence ?? null, trade.notes || null,
    );
  }

  saveState(): void {
    try {
      const txn = this.db.transaction(() => {
        this.persistPortfolioState();
        for (const b of this.state.balances.values()) {
          this.persistBalance(b);
        }
        for (const pos of this.state.lpPositions.values()) {
          this.persistPosition('lp', pos);
        }
        for (const pos of this.state.perpsPositions.values()) {
          this.persistPosition('perps', pos);
        }
      });
      txn();
    } catch (error) {
      logger.error('Failed to save portfolio state', { error });
    }
  }

  private scheduleSave(): void {
    this.persistPortfolioState();
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
    this.persistBalance(tokenBalance);
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
    for (const position of this.state.lpPositions.values()) {
      if (!position.exitTime) {
        total += position.currentValueUsd;
      }
    }
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
    this.persistPosition('lp', position);

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

    this.persistPosition('lp', position);
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

    this.state.realizedPnlUsd += position.realizedPnlUsd;
    this.state.dailyPnlUsd += position.realizedPnlUsd;
    this.state.dailyTradeCount += 1;

    this.persistPosition('lp', position);

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
    this.persistPosition('perps', position);

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

    this.persistPosition('perps', position);
    this.state.unrealizedPnlUsd = this.calculateUnrealizedPnl();
    this.recalculateTotalValue();
    this.scheduleSave();
  }

  closePerpsPosition(id: string, exitPrice: number, fees: number = 0): number {
    const position = this.state.perpsPositions.get(id);
    if (!position) return 0;

    position.exitTime = Date.now();
    position.exitPrice = exitPrice;

    const priceDiff = exitPrice - position.entryPrice;
    const direction = position.side === 'long' ? 1 : -1;
    position.realizedPnlUsd = (priceDiff / position.entryPrice) * position.sizeUsd * direction - position.fundingPaidUsd - fees;

    this.state.realizedPnlUsd += position.realizedPnlUsd;
    this.state.dailyPnlUsd += position.realizedPnlUsd;
    this.state.dailyTradeCount += 1;

    this.persistPosition('perps', position);

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
    this.persistTrade(trade);
  }

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
      side: 'buy',
      ...params,
    });

    this.state.realizedPnlUsd += params.pnlUsd;
    this.state.dailyPnlUsd += params.pnlUsd;
    this.state.dailyTradeCount += 1;

    this.scheduleSave();
    logger.info('Arbitrage trade recorded', { asset: params.asset, pnlUsd: params.pnlUsd });
  }

  getRecentTrades(limit = 50): TradeRecord[] {
    const rows = this.db.prepare(
      'SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?'
    ).all(limit) as any[];

    return rows.map(t => ({
      id: t.id,
      type: t.type,
      timestamp: t.timestamp,
      asset: t.asset,
      side: t.side,
      amountUsd: t.amount_usd,
      venue: t.venue,
      txSignature: t.tx_signature || undefined,
      fees: t.fees,
      slippage: t.slippage ?? undefined,
      pnlUsd: t.pnl_usd,
      mlConfidence: t.ml_confidence ?? undefined,
      notes: t.notes || undefined,
    }));
  }

  // ============= PNL CALCULATIONS =============

  private calculateUnrealizedPnl(): number {
    let unrealized = 0;
    for (const pos of this.state.lpPositions.values()) {
      if (!pos.exitTime) {
        unrealized += pos.currentValueUsd - pos.capitalUsd;
      }
    }
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
      totalTrades: (this.db.prepare('SELECT COUNT(*) as cnt FROM trades').get() as any).cnt,
      dailyPnlUsd: this.state.dailyPnlUsd,
    };
  }
}

// ============= SINGLETON =============

let portfolioManagerInstance: PortfolioManager | null = null;

export function getPortfolioManager(options?: {
  initialCapitalUsd?: number;
  dbPath?: string;
}): PortfolioManager {
  if (!portfolioManagerInstance) {
    portfolioManagerInstance = new PortfolioManager(options);
  }
  return portfolioManagerInstance;
}

export function resetPortfolioManager(): void {
  portfolioManagerInstance = null;
}
