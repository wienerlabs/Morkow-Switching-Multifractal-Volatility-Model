/**
 * P&L Tracker for Perps ML Trading Agent
 * 
 * Tracks:
 * - Realized P&L per trade
 * - Unrealized P&L for open positions
 * - Cumulative P&L over time
 * - Exports to CSV: logs/perps_pnl.csv
 */
import * as fs from 'fs';
import * as path from 'path';
import type { PositionSide } from '../../../types/perps.js';
import type { CloseReason, TrackedPosition } from './tradingAgent.js';
import { logger } from '../../logger.js';

// ============= TYPES =============

export interface TradeRecord {
  id: string;
  market: string;
  side: PositionSide;
  sizeUsd: number;
  entryPrice: number;
  exitPrice: number;
  entryTime: Date;
  exitTime: Date;
  holdTimeHours: number;
  pnlUsd: number;
  pnlPercent: number;
  closeReason: CloseReason;
  fundingPaid: number;
  venue: string;
}

export interface PnLStats {
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  totalPnL: number;
  avgPnL: number;
  avgWin: number;
  avgLoss: number;
  maxWin: number;
  maxLoss: number;
  profitFactor: number;
  avgHoldTime: number;
  todayPnL: number;
  hourlyPnL: number;
}

// ============= P&L TRACKER =============

class PnLTracker {
  private trades: TradeRecord[] = [];
  private csvPath: string;
  private logDir: string;

  constructor() {
    this.logDir = process.env.PERPS_LOG_DIR || './logs';
    this.csvPath = path.join(this.logDir, 'perps_pnl.csv');
    this.loadFromCSV();
  }

  /** Load existing trades from CSV */
  private loadFromCSV(): void {
    if (!fs.existsSync(this.csvPath)) return;

    try {
      const content = fs.readFileSync(this.csvPath, 'utf-8');
      const lines = content.trim().split('\n').slice(1); // Skip header
      
      for (const line of lines) {
        const [id, market, side, sizeUsd, entryPrice, exitPrice, entryTime, exitTime, 
               holdTimeHours, pnlUsd, pnlPercent, closeReason, fundingPaid, venue] = line.split(',');
        
        this.trades.push({
          id,
          market,
          side: side as PositionSide,
          sizeUsd: parseFloat(sizeUsd),
          entryPrice: parseFloat(entryPrice),
          exitPrice: parseFloat(exitPrice),
          entryTime: new Date(entryTime),
          exitTime: new Date(exitTime),
          holdTimeHours: parseFloat(holdTimeHours),
          pnlUsd: parseFloat(pnlUsd),
          pnlPercent: parseFloat(pnlPercent),
          closeReason: closeReason as CloseReason,
          fundingPaid: parseFloat(fundingPaid),
          venue,
        });
      }
    } catch (err) {
      logger.warn('Could not load existing P&L data', { error: String(err) });
    }
  }

  /** Save trade to CSV */
  private saveToCSV(trade: TradeRecord): void {
    // Ensure directory exists
    if (!fs.existsSync(this.logDir)) {
      fs.mkdirSync(this.logDir, { recursive: true });
    }

    // Add header if file doesn't exist
    if (!fs.existsSync(this.csvPath)) {
      const header = 'id,market,side,sizeUsd,entryPrice,exitPrice,entryTime,exitTime,holdTimeHours,pnlUsd,pnlPercent,closeReason,fundingPaid,venue\n';
      fs.writeFileSync(this.csvPath, header);
    }

    // Append trade
    const line = `${trade.id},${trade.market},${trade.side},${trade.sizeUsd},${trade.entryPrice},${trade.exitPrice},${trade.entryTime.toISOString()},${trade.exitTime.toISOString()},${trade.holdTimeHours.toFixed(2)},${trade.pnlUsd.toFixed(4)},${trade.pnlPercent.toFixed(6)},${trade.closeReason},${trade.fundingPaid.toFixed(4)},${trade.venue}\n`;
    fs.appendFileSync(this.csvPath, line);
  }

  /** Record a closed trade */
  recordTrade(
    position: TrackedPosition,
    exitPrice: number,
    pnlPercent: number,
    closeReason: CloseReason,
    fundingPaid: number = 0,
    venue: string = 'drift'
  ): TradeRecord {
    const now = new Date();
    const holdTimeHours = (now.getTime() - position.entryTime.getTime()) / (1000 * 60 * 60);
    const pnlUsd = position.size * pnlPercent;

    const trade: TradeRecord = {
      id: position.id,
      market: position.market,
      side: position.side,
      sizeUsd: position.size,
      entryPrice: position.entryPrice,
      exitPrice,
      entryTime: position.entryTime,
      exitTime: now,
      holdTimeHours,
      pnlUsd,
      pnlPercent,
      closeReason,
      fundingPaid,
      venue,
    };

    this.trades.push(trade);
    this.saveToCSV(trade);

    return trade;
  }

  /** Get P&L statistics */
  getStats(): PnLStats {
    const wins = this.trades.filter(t => t.pnlUsd > 0);
    const losses = this.trades.filter(t => t.pnlUsd <= 0);
    const totalPnL = this.trades.reduce((sum, t) => sum + t.pnlUsd, 0);
    const grossWins = wins.reduce((sum, t) => sum + t.pnlUsd, 0);
    const grossLosses = Math.abs(losses.reduce((sum, t) => sum + t.pnlUsd, 0));
    
    // Today's P&L
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const todayTrades = this.trades.filter(t => t.exitTime >= today);
    const todayPnL = todayTrades.reduce((sum, t) => sum + t.pnlUsd, 0);
    
    // Last hour P&L
    const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000);
    const hourTrades = this.trades.filter(t => t.exitTime >= oneHourAgo);
    const hourlyPnL = hourTrades.reduce((sum, t) => sum + t.pnlUsd, 0);

    return {
      totalTrades: this.trades.length,
      winningTrades: wins.length,
      losingTrades: losses.length,
      winRate: this.trades.length > 0 ? wins.length / this.trades.length : 0,
      totalPnL,
      avgPnL: this.trades.length > 0 ? totalPnL / this.trades.length : 0,
      avgWin: wins.length > 0 ? grossWins / wins.length : 0,
      avgLoss: losses.length > 0 ? grossLosses / losses.length : 0,
      maxWin: wins.length > 0 ? Math.max(...wins.map(t => t.pnlUsd)) : 0,
      maxLoss: losses.length > 0 ? Math.min(...losses.map(t => t.pnlUsd)) : 0,
      profitFactor: grossLosses > 0 ? grossWins / grossLosses : grossWins > 0 ? Infinity : 0,
      avgHoldTime: this.trades.length > 0 
        ? this.trades.reduce((sum, t) => sum + t.holdTimeHours, 0) / this.trades.length 
        : 0,
      todayPnL,
      hourlyPnL,
    };
  }

  /** Get recent trades */
  getRecentTrades(limit = 5): TradeRecord[] {
    return this.trades.slice(-limit);
  }

  /** Calculate unrealized P&L for open positions */
  calculateUnrealizedPnL(positions: TrackedPosition[], currentPrices: Map<string, number>): number {
    let unrealized = 0;
    for (const pos of positions) {
      const currentPrice = currentPrices.get(pos.market) || pos.entryPrice;
      const priceChange = (currentPrice - pos.entryPrice) / pos.entryPrice;
      const pnl = pos.side === 'long' ? priceChange * pos.size : -priceChange * pos.size;
      unrealized += pnl;
    }
    return unrealized;
  }
}

// Singleton instance
export const pnlTracker = new PnLTracker();

