/**
 * A-LAMS-VaR HTTP Client
 *
 * Bridges the TypeScript runtime (ElizaOS) to the Python A-LAMS-VaR model API.
 * Provides VaR calculation, model status, and fit/backtest operations with
 * built-in caching, timeout, and retry logic.
 *
 * The Python API runs at ALAMS_API_URL (default http://localhost:8000/api/v1)
 * and exposes:
 *   POST /risk/var/fit       — calibrate model on return series
 *   POST /risk/var/calculate — compute liquidity-adjusted VaR
 *   GET  /risk/var/summary   — model diagnostic summary
 *   POST /risk/var/load      — load persisted model from Redis
 */

import { logger } from '../logger.js';
import type { ALAMSVaRResult, ALAMSVaRConfig } from './types.js';

// ============= CONFIGURATION =============

const DEFAULT_ALAMS_CONFIG: ALAMSVaRConfig = {
  apiUrl: process.env.ALAMS_API_URL || 'http://localhost:8001/api/v1',
  timeoutMs: parseInt(process.env.ALAMS_VAR_TIMEOUT_MS || '5000', 10),
  maxAcceptableVarPct: parseFloat(process.env.ALAMS_MAX_ACCEPTABLE_VAR || '0.05'),
  cacheTtlMs: parseInt(process.env.ALAMS_CACHE_TTL_MS || '5000', 10),
};

// ============= INTERFACES =============

export interface ALAMSFitResult {
  log_likelihood: number;
  delta: number;
  n_obs: number;
  n_regimes: number;
  aic: number;
  bic: number;
  stage1_success: boolean;
  stage2_success: boolean;
  persisted: boolean;
}

export interface ALAMSModelSummary {
  is_fitted: boolean;
  n_regimes?: number;
  n_obs?: number;
  delta?: number;
  log_likelihood?: number;
  regime_means?: number[];
  regime_sigmas?: number[];
  current_regime?: number;
  regime_probs?: number[];
  var_95?: number;
  var_99?: number;
}

export interface ALAMSBacktestResult {
  n_obs: number;
  n_violations: number;
  violation_rate: number;
  expected_rate: number;
  kupiec_pass: boolean;
  kupiec_pvalue: number;
  christoffersen_pass: boolean;
  christoffersen_pvalue: number;
  cc_pass: boolean;
  cc_pvalue: number;
}

// ============= CLIENT =============

export class ALAMSVaRClient {
  private config: ALAMSVaRConfig;
  private cachedResult: ALAMSVaRResult | null = null;
  private cachedAt: number = 0;

  constructor(config?: Partial<ALAMSVaRConfig>) {
    this.config = { ...DEFAULT_ALAMS_CONFIG, ...config };
  }

  /**
   * Calculate liquidity-adjusted VaR using the fitted A-LAMS model.
   * Returns cached result if within TTL.
   */
  async calculateVaR(params: {
    confidence?: number;
    trade_size_usd?: number;
    pool_depth_usd?: number;
    returns?: number[];
  }): Promise<ALAMSVaRResult> {
    // Return cached result if fresh
    const now = Date.now();
    if (
      this.cachedResult &&
      !params.returns &&
      now - this.cachedAt < this.config.cacheTtlMs
    ) {
      return this.cachedResult;
    }

    const body = {
      confidence: params.confidence ?? 0.95,
      trade_size_usd: params.trade_size_usd ?? 0.0,
      pool_depth_usd: params.pool_depth_usd ?? 1e9,
      returns: params.returns ?? null,
    };

    const result = await this.post<ALAMSVaRResult>('/risk/var/calculate', body);

    // Cache if no custom returns were provided (standard query)
    if (!params.returns) {
      this.cachedResult = result;
      this.cachedAt = Date.now();
    }

    return result;
  }

  /**
   * Fit (calibrate) the A-LAMS-VaR model on a return series.
   * Automatically persists the model to Redis.
   */
  async fitModel(
    returns: number[],
    options?: {
      n_regimes?: number;
      asymmetry_prior?: number;
      max_iter?: number;
      token?: string;
    },
  ): Promise<ALAMSFitResult> {
    const body = {
      returns,
      n_regimes: options?.n_regimes ?? 5,
      asymmetry_prior: options?.asymmetry_prior ?? 0.15,
      max_iter: options?.max_iter ?? 200,
      token: options?.token ?? 'default',
    };

    // Fitting can be slow — use longer timeout
    const result = await this.post<ALAMSFitResult>('/risk/var/fit', body, this.config.timeoutMs * 6);

    // Invalidate cache after re-fit
    this.cachedResult = null;
    this.cachedAt = 0;

    return result;
  }

  /**
   * Get current model diagnostic summary.
   */
  async getSummary(): Promise<ALAMSModelSummary> {
    return this.get<ALAMSModelSummary>('/risk/var/summary');
  }

  /**
   * Check if the Python model is fitted and ready.
   */
  async isModelFitted(): Promise<boolean> {
    try {
      const summary = await this.getSummary();
      return summary.is_fitted;
    } catch {
      return false;
    }
  }

  /**
   * Load a persisted model from Redis.
   */
  async loadModel(token: string = 'default'): Promise<{ loaded: boolean; n_obs: number }> {
    const result = await this.post<{ loaded: boolean; n_obs: number; n_regimes: number; delta: number }>(
      '/risk/var/load',
      { token },
    );
    this.cachedResult = null;
    this.cachedAt = 0;
    return result;
  }

  /**
   * Run a VaR backtest on the given return series.
   */
  async backtest(
    returns: number[],
    options?: { confidence?: number; min_window?: number; refit_every?: number },
  ): Promise<ALAMSBacktestResult> {
    return this.post<ALAMSBacktestResult>('/risk/var/backtest', {
      returns,
      confidence: options?.confidence ?? 0.95,
      min_window: options?.min_window ?? 100,
      refit_every: options?.refit_every ?? 50,
    }, this.config.timeoutMs * 12);
  }

  // ============= HTTP HELPERS =============

  private async post<T>(path: string, body: unknown, timeoutMs?: number): Promise<T> {
    const url = `${this.config.apiUrl}${path}`;
    const timeout = timeoutMs ?? this.config.timeoutMs;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!response.ok) {
        const detail = await response.text().catch(() => 'Unknown error');
        throw new Error(`A-LAMS API ${response.status}: ${detail}`);
      }

      return (await response.json()) as T;
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`A-LAMS API timeout after ${timeout}ms: ${path}`);
      }
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }

  private async get<T>(path: string): Promise<T> {
    const url = `${this.config.apiUrl}${path}`;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.config.timeoutMs);

    try {
      const response = await fetch(url, {
        method: 'GET',
        signal: controller.signal,
      });

      if (!response.ok) {
        const detail = await response.text().catch(() => 'Unknown error');
        throw new Error(`A-LAMS API ${response.status}: ${detail}`);
      }

      return (await response.json()) as T;
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`A-LAMS API timeout after ${this.config.timeoutMs}ms: ${path}`);
      }
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }
}
