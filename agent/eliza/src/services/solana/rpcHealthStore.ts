/**
 * RPC Health Store — Redis-backed shared health state for cross-process RPC failover.
 *
 * Both TypeScript (agent) and Python (cortex API) write endpoint health to Redis
 * so that when one process marks an endpoint as down, the other stops hitting it.
 *
 * Key format: `rpc:health:{sha256(url)[:16]}`
 * Value: JSON with status, failCount, lastFailure, writer, updatedAt
 * TTL: 120 seconds (stale entries auto-expire)
 *
 * Writes are fire-and-forget (non-blocking).
 * Reads are cached locally and refreshed on a 30s timer.
 */

import crypto from 'crypto';
import { config as prodConfig } from '../../config/production.js';
import { logger } from '../logger.js';

// Lazy-load ioredis to avoid hard failure if not installed.
// We use `any` for the client since ioredis is dynamically imported.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let redisClient: any = null;
let isConnected = false;

const KEY_PREFIX = 'rpc:health:';
const TTL_SECONDS = 120;

export interface SharedEndpointHealth {
  url: string;
  status: 'healthy' | 'degraded' | 'down';
  failCount: number;
  lastFailure: number;
  avgLatencyMs: number;
  successRate: number;
  writer: 'typescript' | 'python';
  updatedAt: number;
}

/** Cached shared health state, refreshed every 30s. */
let sharedHealthCache: Map<string, SharedEndpointHealth> = new Map();

function urlToKey(url: string): string {
  const hash = crypto.createHash('sha256').update(url).digest('hex').slice(0, 16);
  return `${KEY_PREFIX}${hash}`;
}

/**
 * Initialize the Redis connection for RPC health sharing.
 * No-op if Redis is disabled or URL is empty.
 */
export async function initRpcHealthStore(): Promise<void> {
  if (!prodConfig.redis.enabled || !prodConfig.redis.url) {
    logger.info('[RpcHealthStore] Redis disabled or no URL — shared health sync off');
    return;
  }

  try {
    const ioredis = await import('ioredis');
    const RedisClass = ioredis.default;

    redisClient = new RedisClass(prodConfig.redis.url, {
      maxRetriesPerRequest: 1,
      lazyConnect: true,
      connectTimeout: 5000,
      retryStrategy(times: number) {
        if (times > 3) return null; // Stop reconnecting after 3 attempts
        return Math.min(times * 500, 3000);
      },
    });

    redisClient.on('connect', () => {
      isConnected = true;
      logger.info('[RpcHealthStore] Redis connected for RPC health sync');
    });

    redisClient.on('error', (err: Error) => {
      isConnected = false;
      logger.warn('[RpcHealthStore] Redis error', { error: err.message });
    });

    redisClient.on('close', () => {
      isConnected = false;
    });

    await redisClient.connect();
  } catch (err) {
    logger.warn('[RpcHealthStore] Failed to init Redis — shared health sync off', {
      error: err instanceof Error ? err.message : String(err),
    });
    redisClient = null;
  }
}

/**
 * Gracefully close the Redis connection.
 */
export async function closeRpcHealthStore(): Promise<void> {
  if (redisClient) {
    try {
      await redisClient.quit();
    } catch {
      // Ignore close errors
    }
    redisClient = null;
    isConnected = false;
  }
}

/**
 * Fire-and-forget write of endpoint health to Redis.
 */
export function writeEndpointHealth(url: string, health: Omit<SharedEndpointHealth, 'url' | 'writer' | 'updatedAt'>): void {
  if (!redisClient || !isConnected) return;

  const key = urlToKey(url);
  const value: SharedEndpointHealth = {
    url,
    ...health,
    writer: 'typescript',
    updatedAt: Date.now(),
  };

  // Fire-and-forget — don't await
  redisClient.setex(key, TTL_SECONDS, JSON.stringify(value)).catch((err: Error) => {
    logger.warn('[RpcHealthStore] Write failed', { error: err.message });
  });
}

/**
 * Read all shared endpoint health entries from Redis.
 * Updates the local cache. Called on a 30s timer.
 */
export async function refreshSharedHealth(): Promise<void> {
  if (!redisClient || !isConnected) return;

  try {
    const keys: string[] = [];
    let cursor = '0';
    do {
      const [nextCursor, batch] = await redisClient.scan(cursor, 'MATCH', `${KEY_PREFIX}*`, 'COUNT', 50);
      cursor = nextCursor;
      keys.push(...batch);
    } while (cursor !== '0');

    if (keys.length === 0) {
      sharedHealthCache = new Map();
      return;
    }

    const values = await redisClient.mget(...keys);
    const newCache = new Map<string, SharedEndpointHealth>();

    for (const val of values) {
      if (val) {
        try {
          const parsed = JSON.parse(val) as SharedEndpointHealth;
          newCache.set(parsed.url, parsed);
        } catch {
          // Skip malformed entries
        }
      }
    }

    sharedHealthCache = newCache;
  } catch (err) {
    logger.warn('[RpcHealthStore] Refresh failed', {
      error: err instanceof Error ? err.message : String(err),
    });
  }
}

/**
 * Check if an endpoint is flagged as down by the OTHER process.
 * Used by pickEndpoint() to skip endpoints flagged by Python.
 */
export function isEndpointFlaggedDown(url: string): boolean {
  const shared = sharedHealthCache.get(url);
  if (!shared) return false;
  // Only consider entries from the other writer
  if (shared.writer === 'typescript') return false;
  // Consider stale entries (>2 min) as expired
  if (Date.now() - shared.updatedAt > TTL_SECONDS * 1000) return false;
  return shared.status === 'down';
}

/**
 * Get the cached shared health map (for health endpoint reporting).
 */
export function getSharedHealthCache(): Map<string, SharedEndpointHealth> {
  return sharedHealthCache;
}

/**
 * Check if the Redis health store is available.
 */
export function isRedisHealthStoreAvailable(): boolean {
  return isConnected;
}
