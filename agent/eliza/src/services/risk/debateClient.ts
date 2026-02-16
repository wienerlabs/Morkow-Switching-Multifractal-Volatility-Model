/**
 * Adversarial Debate HTTP Client
 *
 * Bridges the TypeScript runtime (ElizaOS) to the Python adversarial debate
 * pipeline. Provides debate execution and outcome circuit breaker management.
 *
 * The Python API runs at CORTEX_API_URL (default http://localhost:8000) and
 * exposes:
 *   POST /guardian/debate                — run standalone debate
 *   POST /guardian/assess                — full assessment with optional debate
 *   POST /guardian/trade-outcome/strategy — record trade outcome for circuit breakers
 *   GET  /guardian/circuit-breakers/outcomes — outcome-based circuit breaker states
 */

import { logger } from '../logger.js';
import type { DebateResult, OutcomeCircuitBreakerStatus } from './types.js';

// ============= CONFIGURATION =============

const DEFAULT_API_URL = process.env.CORTEX_API_URL || 'http://localhost:8000/api/v1';
const DEFAULT_TIMEOUT_MS = parseInt(process.env.DEBATE_TIMEOUT_MS || '10000', 10);

interface DebateClientConfig {
  apiUrl: string;
  timeoutMs: number;
}

// ============= DEBATE CLIENT =============

export class DebateClient {
  private config: DebateClientConfig;
  private abortController: AbortController | null = null;

  constructor(config?: Partial<DebateClientConfig>) {
    this.config = {
      apiUrl: config?.apiUrl ?? DEFAULT_API_URL,
      timeoutMs: config?.timeoutMs ?? DEFAULT_TIMEOUT_MS,
    };
    logger.info('[DebateClient] Initialized', {
      apiUrl: this.config.apiUrl,
      timeoutMs: this.config.timeoutMs,
    });
  }

  /**
   * Run a standalone adversarial debate for a trade proposal.
   */
  async runDebate(params: {
    token: string;
    direction: string;
    trade_size_usd: number;
    strategy?: string;
    run_debate?: boolean;
  }): Promise<DebateResult> {
    const url = `${this.config.apiUrl}/guardian/debate`;
    this.abortController = new AbortController();
    const timeout = setTimeout(() => this.abortController?.abort(), this.config.timeoutMs);

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          token: params.token,
          direction: params.direction,
          trade_size_usd: params.trade_size_usd,
          strategy: params.strategy ?? 'spot',
          run_debate: true,
        }),
        signal: this.abortController.signal,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Debate API error ${response.status}: ${text}`);
      }

      return (await response.json()) as DebateResult;
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Debate request timed out after ${this.config.timeoutMs}ms`);
      }
      throw error;
    } finally {
      clearTimeout(timeout);
      this.abortController = null;
    }
  }

  /**
   * Record a trade outcome for strategy-specific circuit breakers.
   *
   * Feeds the outcome-based breakers:
   * - LP: 3 consecutive IL → pause
   * - Arb: 5 consecutive failed executions → pause
   * - Perp: 2 consecutive stop-losses → pause
   */
  async recordTradeOutcome(params: {
    strategy: string;
    success: boolean;
    pnl?: number;
    loss_type?: string;
    details?: string;
  }): Promise<OutcomeCircuitBreakerStatus> {
    const url = new URL(`${this.config.apiUrl}/guardian/trade-outcome/strategy`);
    url.searchParams.set('strategy', params.strategy);
    url.searchParams.set('success', String(params.success));
    if (params.pnl !== undefined) url.searchParams.set('pnl', String(params.pnl));
    if (params.loss_type) url.searchParams.set('loss_type', params.loss_type);
    if (params.details) url.searchParams.set('details', params.details);

    this.abortController = new AbortController();
    const timeout = setTimeout(() => this.abortController?.abort(), this.config.timeoutMs);

    try {
      const response = await fetch(url.toString(), {
        method: 'POST',
        signal: this.abortController.signal,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Outcome record API error ${response.status}: ${text}`);
      }

      return (await response.json()) as OutcomeCircuitBreakerStatus;
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Outcome record timed out after ${this.config.timeoutMs}ms`);
      }
      throw error;
    } finally {
      clearTimeout(timeout);
      this.abortController = null;
    }
  }

  /**
   * Get outcome-based circuit breaker states.
   */
  async getOutcomeBreakerStates(): Promise<{
    outcome_breakers: OutcomeCircuitBreakerStatus[];
    timestamp: number;
  }> {
    const url = `${this.config.apiUrl}/guardian/circuit-breakers/outcomes`;
    this.abortController = new AbortController();
    const timeout = setTimeout(() => this.abortController?.abort(), this.config.timeoutMs);

    try {
      const response = await fetch(url, {
        method: 'GET',
        signal: this.abortController.signal,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Outcome breakers API error ${response.status}: ${text}`);
      }

      return (await response.json()) as {
        outcome_breakers: OutcomeCircuitBreakerStatus[];
        timestamp: number;
      };
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Outcome breakers request timed out after ${this.config.timeoutMs}ms`);
      }
      throw error;
    } finally {
      clearTimeout(timeout);
      this.abortController = null;
    }
  }

  /**
   * Check if a strategy is blocked by outcome circuit breakers.
   */
  async isStrategyBlocked(strategy: string): Promise<boolean> {
    try {
      const states = await this.getOutcomeBreakerStates();
      const breaker = states.outcome_breakers.find(b => b.strategy === strategy);
      return (breaker?.state === 'open') || false;
    } catch (error) {
      logger.warn('[DebateClient] Failed to check strategy block status', {
        strategy,
        error: error instanceof Error ? error.message : error,
      });
      return false;
    }
  }

  // ============= DEBATE TRANSCRIPT STORE =============

  /**
   * Get recent debate transcripts from the Python debate store.
   */
  async getRecentTranscripts(limit: number = 20): Promise<DebateTranscriptResponse> {
    const url = `${this.config.apiUrl}/guardian/debates/recent?limit=${limit}`;
    return this._get<DebateTranscriptResponse>(url);
  }

  /**
   * Get aggregate debate decision statistics over a time window.
   */
  async getDebateStats(hours: number = 24): Promise<DebateStatsResponse> {
    const url = `${this.config.apiUrl}/guardian/debates/stats?hours=${hours}`;
    return this._get<DebateStatsResponse>(url);
  }

  /**
   * Get debate transcripts filtered by strategy.
   */
  async getTranscriptsByStrategy(strategy: string, limit: number = 50): Promise<DebateTranscriptResponse> {
    const url = `${this.config.apiUrl}/guardian/debates/by-strategy/${encodeURIComponent(strategy)}?limit=${limit}`;
    return this._get<DebateTranscriptResponse>(url);
  }

  /**
   * Get debate storage tier statistics (HOT/WARM/COLD).
   */
  async getStorageStats(): Promise<DebateStorageStats> {
    const url = `${this.config.apiUrl}/guardian/debates/storage/stats`;
    return this._get<DebateStorageStats>(url);
  }

  /**
   * Generic GET helper with timeout and abort handling.
   */
  private async _get<T>(url: string): Promise<T> {
    this.abortController = new AbortController();
    const timeout = setTimeout(() => this.abortController?.abort(), this.config.timeoutMs);

    try {
      const response = await fetch(url, {
        method: 'GET',
        signal: this.abortController.signal,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`API error ${response.status}: ${text}`);
      }

      return (await response.json()) as T;
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Request timed out after ${this.config.timeoutMs}ms`);
      }
      throw error;
    } finally {
      clearTimeout(timeout);
      this.abortController = null;
    }
  }

  /**
   * Cancel any in-flight requests.
   */
  cancel(): void {
    this.abortController?.abort();
  }
}

// ============= TRANSCRIPT TYPES =============

export interface DebateTranscript {
  id: string;
  timestamp: string;
  epoch_ms: number;
  token: string;
  direction: string;
  trade_size_usd: number;
  strategy: string;
  final_decision: string;
  final_confidence: number;
  approval_score: number;
  recommended_size_pct: number;
  decision_changed: boolean;
  original_approved: boolean;
  num_rounds: number;
  rounds: Array<Record<string, unknown>>;
  evidence_summary: Record<string, unknown>;
  elapsed_ms: number;
}

export interface DebateTranscriptResponse {
  transcripts: DebateTranscript[];
  count: number;
}

export interface DebateStatsResponse {
  period_hours: number;
  total_debates: number;
  decisions: Record<string, number>;
  avg_confidence: number;
  avg_rounds: number;
  avg_elapsed_ms: number;
  decision_changed_count: number;
  by_strategy: Record<string, Record<string, number>>;
}

export interface DebateStorageStats {
  hot_count: number;
  hot_capacity: number;
  warm_file_bytes: number;
  warm_retention_hours: number;
  cold_archive_count: number;
  cold_total_bytes: number;
  cold_archives: string[];
}

// ============= SINGLETON =============

let instance: DebateClient | null = null;

export function getDebateClient(config?: Partial<DebateClientConfig>): DebateClient {
  if (!instance) {
    instance = new DebateClient(config);
  }
  return instance;
}

export function resetDebateClient(): void {
  instance?.cancel();
  instance = null;
}
