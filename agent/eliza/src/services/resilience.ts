/**
 * Resilience utilities for API calls
 *
 * Shared retry + queue infrastructure using p-retry and p-queue.
 * Each external service gets a named queue with its own concurrency/rate limits.
 */

import pRetry, { AbortError, type Options as RetryOptions } from 'p-retry';
import PQueue from 'p-queue';
import { logger } from './logger.js';

// ============= QUEUE REGISTRY =============

const queues = new Map<string, PQueue>();

export function getQueue(name: string, opts?: { concurrency?: number; intervalCap?: number; interval?: number }): PQueue {
  let q = queues.get(name);
  if (!q) {
    q = new PQueue({
      concurrency: opts?.concurrency ?? 3,
      intervalCap: opts?.intervalCap,
      interval: opts?.interval,
    });
    queues.set(name, q);
  }
  return q;
}

// Pre-configured queues for known services
export const Queues: Record<string, () => PQueue> = {
  /** Binance: generous rate limits, 3 concurrent */
  binance: () => getQueue('binance', { concurrency: 3 }),
  /** Coinbase: serial to be safe */
  coinbase: () => getQueue('coinbase', { concurrency: 2 }),
  /** Kraken: serial */
  kraken: () => getQueue('kraken', { concurrency: 2 }),
  /** Birdeye: 60 rpm → 1/sec, serial */
  birdeye: () => getQueue('birdeye', { concurrency: 1, intervalCap: 1, interval: 1100 }),
  /** Helius: 100k/day ≈ ~69/min, generous but serial for RPC */
  helius: () => getQueue('helius', { concurrency: 2 }),
  /** Jupiter price API */
  jupiter: () => getQueue('jupiter', { concurrency: 2 }),
};

// ============= RETRY WRAPPER =============

export interface ResilientFetchOptions {
  /** Queue name for rate-limiting */
  queue?: string | PQueue;
  /** Number of retries (default: 3) */
  retries?: number;
  /** Min timeout before first retry in ms (default: 1000) */
  minTimeout?: number;
  /** Max timeout between retries in ms (default: 15000) */
  maxTimeout?: number;
  /** Fetch timeout in ms (default: 20000) */
  fetchTimeout?: number;
  /** Label for logging */
  label?: string;
  /** Custom retry options override */
  retryOptions?: Partial<RetryOptions>;
}

/**
 * Resilient fetch with retry + optional queue
 *
 * Wraps the native fetch with:
 * 1. AbortController timeout
 * 2. p-retry with exponential backoff
 * 3. Optional p-queue for rate-limiting
 */
export async function resilientFetch(
  url: string,
  init?: RequestInit,
  opts: ResilientFetchOptions = {},
): Promise<Response> {
  const {
    retries = 3,
    minTimeout = 1000,
    maxTimeout = 15000,
    fetchTimeout = 20000,
    label = url.split('?')[0].split('/').slice(-2).join('/'),
  } = opts;

  const attempt = async () => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), fetchTimeout);

    try {
      const resp = await fetch(url, {
        ...init,
        signal: controller.signal,
      });
      clearTimeout(timer);

      // Retry on 429 (rate limit) and 5xx
      if (resp.status === 429) {
        throw new AbortError(`Rate limited (429) on ${label}`);
      }
      if (resp.status >= 500) {
        throw new Error(`Server error ${resp.status} on ${label}`);
      }

      return resp;
    } catch (err: any) {
      clearTimeout(timer);
      // AbortError from timeout
      if (err.name === 'AbortError' || err.cause?.code === 'UND_ERR_CONNECT_TIMEOUT') {
        throw new Error(`Timeout (${fetchTimeout}ms) on ${label}`);
      }
      throw err;
    }
  };

  const retryOpts: RetryOptions = {
    retries,
    minTimeout,
    maxTimeout,
    onFailedAttempt: (ctx) => {
      logger.warn(`[Resilience] ${label} attempt ${ctx.attemptNumber}/${retries + 1} failed`, {
        error: ctx.error.message,
        retriesLeft: ctx.retriesLeft,
      });
    },
    ...opts.retryOptions,
  };

  // If a queue is specified, run inside it
  const q = typeof opts.queue === 'string' ? getQueue(opts.queue) : opts.queue;

  if (q) {
    return q.add(() => pRetry(attempt, retryOpts)) as Promise<Response>;
  }

  return pRetry(attempt, retryOpts);
}

/**
 * Resilient JSON fetch — fetches and parses JSON in one call
 */
export async function resilientFetchJson<T>(
  url: string,
  init?: RequestInit,
  opts: ResilientFetchOptions = {},
): Promise<T> {
  const resp = await resilientFetch(url, init, opts);
  return resp.json() as Promise<T>;
}
