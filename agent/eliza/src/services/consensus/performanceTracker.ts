/**
 * Performance Tracker
 * 
 * Tracks agent performance and calculates voting weights
 * based on historical accuracy and profitability.
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { logger } from '../logger.js';
import type {
  AgentPerformance,
  TradeRecord,
} from './types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DEFAULT_STATE_FILE = path.join(__dirname, '../../../data/performance_tracker_state.json');

// ============= TYPES =============

export interface PerformanceConfig {
  lookbackDays: number;
  minTradesForWeight: number;
  baseWeight: number;
  maxWeight: number;
  minWeight: number;
  decayFactor: number;  // How much older trades count less
}

const DEFAULT_PERFORMANCE_CONFIG: PerformanceConfig = {
  lookbackDays: 30,
  minTradesForWeight: 5,
  baseWeight: 1.0,
  maxWeight: 2.0,
  minWeight: 0.5,
  decayFactor: 0.95,  // 5% decay per day
};

// ============= PERFORMANCE TRACKER =============

export class PerformanceTracker {
  private config: PerformanceConfig;
  private tradeRecords: Map<string, TradeRecord[]> = new Map();
  private performanceCache: Map<string, AgentPerformance> = new Map();
  private weightCache: Map<string, number> = new Map();
  private lastCacheUpdate: Date = new Date(0);
  private readonly CACHE_TTL_MS = 5 * 60 * 1000;  // 5 minutes
  private stateFilePath: string;

  constructor(config: Partial<PerformanceConfig> = {}, stateFilePath?: string) {
    this.config = { ...DEFAULT_PERFORMANCE_CONFIG, ...config };
    this.stateFilePath = stateFilePath || DEFAULT_STATE_FILE;
    this.loadState();
    logger.info('[PerformanceTracker] Initialized', { config: this.config, stateFile: this.stateFilePath });
  }

  /**
   * Record a trade decision by an agent
   */
  recordTrade(record: TradeRecord): void {
    const records = this.tradeRecords.get(record.agentId) || [];
    records.push(record);
    this.tradeRecords.set(record.agentId, records);
    
    // Invalidate cache
    this.weightCache.delete(record.agentId);
    this.performanceCache.delete(record.agentId);

    logger.debug('[PerformanceTracker] Trade recorded', {
      agentId: record.agentId,
      asset: record.asset,
      decision: record.decision,
    });

    this.saveState();
  }

  /**
   * Update a trade with outcome
   */
  updateTradeOutcome(
    agentId: string, 
    asset: string, 
    entryTime: Date,
    exitPrice: number,
    pnl: number
  ): void {
    const records = this.tradeRecords.get(agentId) || [];
    const record = records.find(
      r => r.asset === asset && r.entryTime.getTime() === entryTime.getTime()
    );

    if (record) {
      record.exitPrice = exitPrice;
      record.pnl = pnl;
      record.pnlPct = ((exitPrice - record.entryPrice) / record.entryPrice) * 100;
      record.exitTime = new Date();
      record.outcome = pnl > 0 ? 'WIN' : 'LOSS';

      // Invalidate cache
      this.weightCache.delete(agentId);
      this.performanceCache.delete(agentId);

      logger.debug('[PerformanceTracker] Trade outcome updated', {
        agentId,
        asset,
        outcome: record.outcome,
        pnl,
      });

      this.saveState();
    }
  }

  /**
   * Get performance-based weight for an agent
   */
  async getAgentWeight(agentId: string): Promise<number> {
    // Check cache
    if (this.weightCache.has(agentId) && this.isCacheValid()) {
      return this.weightCache.get(agentId)!;
    }

    const performance = await this.calculatePerformance(agentId);
    const weight = this.calculateWeight(performance);
    
    this.weightCache.set(agentId, weight);
    return weight;
  }

  /**
   * Get full performance metrics for an agent
   */
  async getAgentPerformance(agentId: string): Promise<AgentPerformance> {
    if (this.performanceCache.has(agentId) && this.isCacheValid()) {
      return this.performanceCache.get(agentId)!;
    }

    const performance = await this.calculatePerformance(agentId);
    this.performanceCache.set(agentId, performance);
    return performance;
  }

  /**
   * Get all agent weights for logging/display
   */
  async getAllWeights(): Promise<Map<string, number>> {
    const weights = new Map<string, number>();
    for (const agentId of this.tradeRecords.keys()) {
      weights.set(agentId, await this.getAgentWeight(agentId));
    }
    return weights;
  }

  // ============= STATE PERSISTENCE =============

  private loadState(): void {
    try {
      if (!fs.existsSync(this.stateFilePath)) {
        return;
      }
      const json = fs.readFileSync(this.stateFilePath, 'utf-8');
      const data = JSON.parse(json) as Record<string, Array<TradeRecord & { entryTime: string; exitTime?: string }>>;

      for (const [agentId, records] of Object.entries(data)) {
        this.tradeRecords.set(agentId, records.map(r => ({
          ...r,
          entryTime: new Date(r.entryTime),
          exitTime: r.exitTime ? new Date(r.exitTime) : undefined,
        })));
      }

      logger.info('[PerformanceTracker] State loaded from disk', {
        agents: Object.keys(data).length,
        path: this.stateFilePath,
      });
    } catch (error) {
      logger.warn('[PerformanceTracker] Failed to load state, starting fresh', { error });
    }
  }

  private saveState(): void {
    try {
      const dir = path.dirname(this.stateFilePath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }

      const data: Record<string, TradeRecord[]> = {};
      for (const [agentId, records] of this.tradeRecords.entries()) {
        data[agentId] = records;
      }

      const tmpPath = this.stateFilePath + '.tmp';
      fs.writeFileSync(tmpPath, JSON.stringify(data, null, 2));
      fs.renameSync(tmpPath, this.stateFilePath);
    } catch (error) {
      logger.error('[PerformanceTracker] Failed to save state', { error });
    }
  }

  // ============= PRIVATE METHODS =============

  private isCacheValid(): boolean {
    return Date.now() - this.lastCacheUpdate.getTime() < this.CACHE_TTL_MS;
  }

  private async calculatePerformance(agentId: string): Promise<AgentPerformance> {
    const records = this.tradeRecords.get(agentId) || [];
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - this.config.lookbackDays);

    // Filter to recent completed trades
    const recentTrades = records.filter(
      r => r.outcome && r.entryTime >= cutoffDate
    );

    if (recentTrades.length === 0) {
      return this.getDefaultPerformance(agentId);
    }

    // Calculate metrics
    const wins = recentTrades.filter(r => r.outcome === 'WIN').length;
    const winRate = wins / recentTrades.length;

    const returns = recentTrades.map(r => r.pnlPct || 0);
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;

    // Sharpe ratio (simplified - using returns std dev)
    const stdDev = this.calculateStdDev(returns);
    const sharpeRatio = stdDev > 0 ? avgReturn / stdDev : 0;

    // Profit factor (gross profit / gross loss)
    const grossProfit = recentTrades.filter(r => (r.pnl || 0) > 0)
      .reduce((sum, r) => sum + (r.pnl || 0), 0);
    const grossLoss = Math.abs(recentTrades.filter(r => (r.pnl || 0) < 0)
      .reduce((sum, r) => sum + (r.pnl || 0), 0));
    const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? 10 : 0;

    // Max drawdown (simplified)
    const maxDrawdown = Math.abs(Math.min(...returns, 0));

    return {
      agentId,
      winRate,
      sharpeRatio,
      totalTrades: recentTrades.length,
      profitFactor,
      avgReturn,
      maxDrawdown,
      recentAccuracy: winRate,
      lastUpdated: new Date(),
    };
  }

  private calculateWeight(performance: AgentPerformance): number {
    // Not enough trades - use base weight
    if (performance.totalTrades < this.config.minTradesForWeight) {
      return this.config.baseWeight;
    }

    // Weight based on win rate and profit factor
    let weight = this.config.baseWeight;

    // Win rate contribution (+/- 0.3)
    weight += (performance.winRate - 0.5) * 0.6;

    // Profit factor contribution (+/- 0.2)
    const pfScore = Math.min(2, performance.profitFactor) / 2;
    weight += (pfScore - 0.5) * 0.4;

    // Clamp to min/max
    return Math.max(
      this.config.minWeight,
      Math.min(this.config.maxWeight, weight)
    );
  }

  private getDefaultPerformance(agentId: string): AgentPerformance {
    return {
      agentId,
      winRate: 0.5,
      sharpeRatio: 0,
      totalTrades: 0,
      profitFactor: 1,
      avgReturn: 0,
      maxDrawdown: 0,
      recentAccuracy: 0.5,
      lastUpdated: new Date(),
    };
  }

  private calculateStdDev(values: number[]): number {
    if (values.length === 0) return 0;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
    return Math.sqrt(squaredDiffs.reduce((a, b) => a + b, 0) / values.length);
  }
}

// ============= SINGLETON =============

let instance: PerformanceTracker | null = null;

export function getPerformanceTracker(config?: Partial<PerformanceConfig>): PerformanceTracker {
  if (!instance) {
    instance = new PerformanceTracker(config);
  }
  return instance;
}

export function resetPerformanceTracker(): void {
  instance = null;
}
