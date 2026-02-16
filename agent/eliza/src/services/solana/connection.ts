/**
 * Solana RPC Connection with Failover
 *
 * Provides a resilient Solana connection that:
 * - Tries primary RPC, falls back to secondary, then public endpoint
 * - Tracks endpoint health and avoids failing endpoints for a cooldown period
 * - WebSocket failover with auto-reconnect and exponential backoff
 * - Re-exports a singleton for shared usage across the agent
 */

import { Connection } from '@solana/web3.js';
import { logger } from '../logger.js';
import { writeEndpointHealth, isEndpointFlaggedDown } from './rpcHealthStore.js';

const LATENCY_WINDOW_SIZE = 20;

interface EndpointState {
  url: string;
  failCount: number;
  lastFailure: number;
  totalRequests: number;
  successCount: number;
  avgLatencyMs: number;
  lastSuccessAt: number | null;
  lastFailureAt: number | null;
  latencyWindow: number[];
}

export interface EndpointHealthMetrics {
  url: string;
  status: 'healthy' | 'degraded' | 'down';
  totalRequests: number;
  successCount: number;
  failCount: number;
  avgLatencyMs: number;
  successRate: number;
  lastSuccessAt: string | null;
  lastFailureAt: string | null;
}

export interface RpcHealthReport {
  status: 'healthy' | 'degraded' | 'down';
  endpoints: EndpointHealthMetrics[];
  activeEndpoint: string;
}

const COOLDOWN_MS = parseInt(process.env.SOLANA_RPC_COOLDOWN_MS || '60000', 10);
const MAX_FAIL_BEFORE_COOLDOWN = parseInt(process.env.SOLANA_RPC_MAX_FAILURES || '3', 10);
const PUBLIC_ENDPOINT = 'https://api.mainnet-beta.solana.com';
const PUBLIC_WS_ENDPOINT = 'wss://api.mainnet-beta.solana.com';

let endpoints: EndpointState[] = [];
let activeConnection: Connection | null = null;
let activeEndpointUrl: string = '';

function makeEndpoint(url: string): EndpointState {
  return {
    url,
    failCount: 0,
    lastFailure: 0,
    totalRequests: 0,
    successCount: 0,
    avgLatencyMs: 0,
    lastSuccessAt: null,
    lastFailureAt: null,
    latencyWindow: [],
  };
}

function buildEndpoints(): EndpointState[] {
  const result: EndpointState[] = [];

  const primary = process.env.SOLANA_RPC_URL;
  if (primary) {
    result.push(makeEndpoint(primary));
  }

  const secondary = process.env.SOLANA_RPC_URL_SECONDARY;
  if (secondary) {
    result.push(makeEndpoint(secondary));
  }

  const jito = process.env.JITO_RPC_URL;
  if (jito && !result.some(e => e.url === jito)) {
    result.push(makeEndpoint(jito));
  }

  if (!result.some(e => e.url === PUBLIC_ENDPOINT)) {
    result.push(makeEndpoint(PUBLIC_ENDPOINT));
  }

  return result;
}

function isEndpointHealthy(ep: EndpointState): boolean {
  // Check if other process (Python) flagged this endpoint as down
  if (isEndpointFlaggedDown(ep.url)) return false;
  if (ep.failCount < MAX_FAIL_BEFORE_COOLDOWN) return true;
  return Date.now() - ep.lastFailure > COOLDOWN_MS;
}

/**
 * Get a healthy RPC endpoint, preferring primary → secondary → public
 */
function pickEndpoint(): EndpointState {
  if (endpoints.length === 0) {
    endpoints = buildEndpoints();
  }

  for (const ep of endpoints) {
    if (isEndpointHealthy(ep)) {
      return ep;
    }
  }

  // All endpoints are in cooldown — reset the oldest and try it
  endpoints.sort((a, b) => a.lastFailure - b.lastFailure);
  const oldest = endpoints[0];
  oldest.failCount = 0;
  return oldest;
}

/**
 * Record a failure on the active endpoint
 */
export function recordRpcFailure(latencyMs?: number): void {
  const ep = endpoints.find(e => e.url === activeEndpointUrl);
  if (ep) {
    ep.failCount++;
    ep.lastFailure = Date.now();
    ep.lastFailureAt = Date.now();
    ep.totalRequests++;

    if (latencyMs !== undefined) {
      pushLatency(ep, latencyMs);
    }

    logger.warn('[RPC] Endpoint failure recorded', {
      url: ep.url.slice(0, 30) + '...',
      failCount: ep.failCount,
    });

    // Sync to Redis for cross-process awareness
    writeEndpointHealth(ep.url, {
      status: deriveEndpointStatus(ep),
      failCount: ep.failCount,
      lastFailure: ep.lastFailure,
      avgLatencyMs: ep.avgLatencyMs,
      successRate: ep.totalRequests > 0 ? ep.successCount / ep.totalRequests : 1,
    });

    // Force a new connection on next call
    if (ep.failCount >= MAX_FAIL_BEFORE_COOLDOWN) {
      activeConnection = null;
      activeEndpointUrl = '';
    }
  }
}

/**
 * Record a success, resetting failure count
 */
export function recordRpcSuccess(latencyMs?: number): void {
  const ep = endpoints.find(e => e.url === activeEndpointUrl);
  if (ep) {
    ep.successCount++;
    ep.totalRequests++;
    ep.lastSuccessAt = Date.now();
    if (ep.failCount > 0) {
      ep.failCount = 0;
    }
    if (latencyMs !== undefined) {
      pushLatency(ep, latencyMs);
    }

    // Sync to Redis for cross-process awareness
    writeEndpointHealth(ep.url, {
      status: deriveEndpointStatus(ep),
      failCount: ep.failCount,
      lastFailure: ep.lastFailure,
      avgLatencyMs: ep.avgLatencyMs,
      successRate: ep.totalRequests > 0 ? ep.successCount / ep.totalRequests : 1,
    });
  }
}

/**
 * Get the shared Solana connection with failover support.
 * Re-creates the connection if the current endpoint is unhealthy.
 */
export function getSolanaConnection(): Connection {
  if (activeConnection) {
    const ep = endpoints.find(e => e.url === activeEndpointUrl);
    if (ep && isEndpointHealthy(ep)) {
      return activeConnection;
    }
  }

  const ep = pickEndpoint();
  activeEndpointUrl = ep.url;
  activeConnection = new Connection(ep.url, 'confirmed');

  logger.info('[RPC] Using endpoint', {
    url: ep.url.slice(0, 30) + '...',
    isPublic: ep.url === PUBLIC_ENDPOINT,
  });

  return activeConnection;
}

/**
 * Get the current active RPC URL
 */
export function getActiveRpcUrl(): string {
  if (!activeEndpointUrl) {
    getSolanaConnection(); // force initialization
  }
  return activeEndpointUrl;
}

function pushLatency(ep: EndpointState, ms: number): void {
  ep.latencyWindow.push(ms);
  if (ep.latencyWindow.length > LATENCY_WINDOW_SIZE) {
    ep.latencyWindow.shift();
  }
  ep.avgLatencyMs = ep.latencyWindow.reduce((a, b) => a + b, 0) / ep.latencyWindow.length;
}

function deriveEndpointStatus(ep: EndpointState): 'healthy' | 'degraded' | 'down' {
  if (ep.failCount >= MAX_FAIL_BEFORE_COOLDOWN) return 'down';
  if (ep.failCount > 0 || (ep.totalRequests > 0 && ep.successCount / ep.totalRequests < 0.9)) return 'degraded';
  return 'healthy';
}

export function getHealthMetrics(): RpcHealthReport {
  if (endpoints.length === 0) {
    endpoints = buildEndpoints();
  }

  const epMetrics: EndpointHealthMetrics[] = endpoints.map(ep => ({
    url: ep.url,
    status: deriveEndpointStatus(ep),
    totalRequests: ep.totalRequests,
    successCount: ep.successCount,
    failCount: ep.failCount,
    avgLatencyMs: Math.round(ep.avgLatencyMs * 100) / 100,
    successRate: ep.totalRequests > 0 ? Math.round((ep.successCount / ep.totalRequests) * 1000) / 1000 : 1,
    lastSuccessAt: ep.lastSuccessAt ? new Date(ep.lastSuccessAt).toISOString() : null,
    lastFailureAt: ep.lastFailureAt ? new Date(ep.lastFailureAt).toISOString() : null,
  }));

  const activeEp = epMetrics.find(e => e.url === activeEndpointUrl);
  let overallStatus: 'healthy' | 'degraded' | 'down' = 'healthy';
  if (activeEp?.status === 'down') {
    overallStatus = 'down';
  } else if (activeEp?.status === 'degraded' || epMetrics.some(e => e.status === 'down')) {
    overallStatus = 'degraded';
  }

  return {
    status: overallStatus,
    endpoints: epMetrics,
    activeEndpoint: activeEndpointUrl,
  };
}

export function resetHealthMetrics(): void {
  for (const ep of endpoints) {
    ep.totalRequests = 0;
    ep.successCount = 0;
    ep.failCount = 0;
    ep.avgLatencyMs = 0;
    ep.lastSuccessAt = null;
    ep.lastFailureAt = null;
    ep.latencyWindow = [];
  }
}

/**
 * Wrap an RPC call to automatically measure latency and record success/failure.
 */
export async function withRpcMetrics<T>(fn: () => Promise<T>): Promise<T> {
  const start = Date.now();
  try {
    const result = await fn();
    recordRpcSuccess(Date.now() - start);
    return result;
  } catch (err) {
    recordRpcFailure(Date.now() - start);
    throw err;
  }
}

/**
 * Reset connections (for testing)
 */
export function resetSolanaConnection(): void {
  activeConnection = null;
  activeEndpointUrl = '';
  endpoints = [];
  stopWsReconnect();
  wsEndpoints = [];
  activeWsIndex = 0;
  wsStatus = 'disconnected';
  wsReconnectTimer = null;
  wsBackoffMs = WS_BACKOFF_INITIAL_MS;
  wsConsecutiveFailures = 0;
}

// =============================================================================
// WebSocket Failover
// =============================================================================

export type WsConnectionStatus = 'connected' | 'disconnected' | 'reconnecting';

interface WsEndpoint {
  url: string;
  failCount: number;
}

const WS_BACKOFF_INITIAL_MS = 1_000;
const WS_BACKOFF_MAX_MS = 30_000;
const WS_MAX_FAIL_BEFORE_FAILOVER = parseInt(process.env.SOLANA_WS_MAX_FAILURES || '3', 10);

let wsEndpoints: WsEndpoint[] = [];
let activeWsIndex = 0;
let wsStatus: WsConnectionStatus = 'disconnected';
let wsReconnectTimer: ReturnType<typeof setTimeout> | null = null;
let wsBackoffMs = WS_BACKOFF_INITIAL_MS;
let wsConsecutiveFailures = 0;

function httpToWss(httpUrl: string): string {
  return httpUrl.replace(/^https?:\/\//, 'wss://');
}

function buildWsEndpoints(): WsEndpoint[] {
  const result: WsEndpoint[] = [];

  // Prefer explicit WS env vars, fall back to deriving from HTTP RPC URLs
  const primary = process.env.SOLANA_WS_URL
    || (process.env.SOLANA_RPC_URL ? httpToWss(process.env.SOLANA_RPC_URL) : '');
  if (primary) {
    result.push({ url: primary, failCount: 0 });
  }

  const secondary = process.env.SOLANA_WS_URL_SECONDARY
    || (process.env.SOLANA_RPC_URL_SECONDARY ? httpToWss(process.env.SOLANA_RPC_URL_SECONDARY) : '');
  if (secondary) {
    result.push({ url: secondary, failCount: 0 });
  }

  const jitoWs = process.env.JITO_RPC_URL ? httpToWss(process.env.JITO_RPC_URL) : '';
  if (jitoWs && !result.some(e => e.url === jitoWs)) {
    result.push({ url: jitoWs, failCount: 0 });
  }

  if (!result.some(e => e.url === PUBLIC_WS_ENDPOINT)) {
    result.push({ url: PUBLIC_WS_ENDPOINT, failCount: 0 });
  }

  return result;
}

function ensureWsEndpoints(): void {
  if (wsEndpoints.length === 0) {
    wsEndpoints = buildWsEndpoints();
  }
}

function stopWsReconnect(): void {
  if (wsReconnectTimer) {
    clearTimeout(wsReconnectTimer);
    wsReconnectTimer = null;
  }
}

/**
 * Get the current healthy WebSocket endpoint URL.
 * Mirrors the HTTP failover pattern — primary → secondary → public.
 */
export function getWebSocketEndpoint(): string {
  ensureWsEndpoints();
  return wsEndpoints[activeWsIndex].url;
}

/**
 * Get the current WebSocket connection status.
 */
export function getWsStatus(): WsConnectionStatus {
  return wsStatus;
}

/**
 * Notify the WS manager that the connection is alive.
 * Resets backoff and failure counters.
 */
export function recordWsConnected(): void {
  wsStatus = 'connected';
  wsBackoffMs = WS_BACKOFF_INITIAL_MS;
  wsConsecutiveFailures = 0;
  stopWsReconnect();

  ensureWsEndpoints();
  const ep = wsEndpoints[activeWsIndex];
  if (ep) ep.failCount = 0;

  logger.info('[WS] Connected', { url: ep?.url.slice(0, 40) + '...' });
}

/**
 * Notify the WS manager that the connection dropped.
 * Triggers exponential backoff reconnect and endpoint failover after repeated failures.
 *
 * Returns the endpoint URL to reconnect to (may be a failover endpoint).
 */
export function recordWsDisconnect(): string {
  ensureWsEndpoints();
  wsConsecutiveFailures++;

  const ep = wsEndpoints[activeWsIndex];
  if (ep) {
    ep.failCount++;
  }

  // Failover to next endpoint after N consecutive failures on current
  if (wsConsecutiveFailures >= WS_MAX_FAIL_BEFORE_FAILOVER && wsEndpoints.length > 1) {
    const prevUrl = ep?.url ?? 'unknown';
    activeWsIndex = (activeWsIndex + 1) % wsEndpoints.length;
    wsConsecutiveFailures = 0;
    wsBackoffMs = WS_BACKOFF_INITIAL_MS;

    const nextEp = wsEndpoints[activeWsIndex];
    logger.warn('[WS] Failover to next endpoint', {
      from: prevUrl.slice(0, 40) + '...',
      to: nextEp.url.slice(0, 40) + '...',
    });
  }

  wsStatus = 'reconnecting';
  logger.warn('[WS] Disconnected, will reconnect', {
    backoffMs: wsBackoffMs,
    consecutiveFailures: wsConsecutiveFailures,
    endpoint: wsEndpoints[activeWsIndex].url.slice(0, 40) + '...',
  });

  return wsEndpoints[activeWsIndex].url;
}

/**
 * Schedule a reconnect attempt with exponential backoff.
 * Calls the provided `connectFn` after the backoff delay.
 * Returns a promise that resolves when the reconnect is scheduled (not when it fires).
 */
export function scheduleWsReconnect(connectFn: () => void): void {
  stopWsReconnect();

  wsReconnectTimer = setTimeout(() => {
    wsReconnectTimer = null;
    connectFn();
  }, wsBackoffMs);

  // Exponential backoff with cap
  wsBackoffMs = Math.min(wsBackoffMs * 2, WS_BACKOFF_MAX_MS);
}
