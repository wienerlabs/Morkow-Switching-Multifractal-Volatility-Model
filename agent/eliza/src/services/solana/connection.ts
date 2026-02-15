/**
 * Solana RPC Connection with Failover
 *
 * Provides a resilient Solana connection that:
 * - Tries primary RPC, falls back to secondary, then public endpoint
 * - Tracks endpoint health and avoids failing endpoints for a cooldown period
 * - Re-exports a singleton for shared usage across the agent
 */

import { Connection } from '@solana/web3.js';
import { logger } from '../logger.js';

interface EndpointState {
  url: string;
  failCount: number;
  lastFailure: number;
}

const COOLDOWN_MS = 60_000; // 1 minute cooldown after failure
const MAX_FAIL_BEFORE_COOLDOWN = 3;
const PUBLIC_ENDPOINT = 'https://api.mainnet-beta.solana.com';

let endpoints: EndpointState[] = [];
let activeConnection: Connection | null = null;
let activeEndpointUrl: string = '';

function buildEndpoints(): EndpointState[] {
  const result: EndpointState[] = [];

  const primary = process.env.SOLANA_RPC_URL;
  if (primary) {
    result.push({ url: primary, failCount: 0, lastFailure: 0 });
  }

  const secondary = process.env.SOLANA_RPC_URL_SECONDARY;
  if (secondary) {
    result.push({ url: secondary, failCount: 0, lastFailure: 0 });
  }

  // Always have public endpoint as last resort
  if (!result.some(e => e.url === PUBLIC_ENDPOINT)) {
    result.push({ url: PUBLIC_ENDPOINT, failCount: 0, lastFailure: 0 });
  }

  return result;
}

function isEndpointHealthy(ep: EndpointState): boolean {
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
export function recordRpcFailure(): void {
  const ep = endpoints.find(e => e.url === activeEndpointUrl);
  if (ep) {
    ep.failCount++;
    ep.lastFailure = Date.now();

    logger.warn('[RPC] Endpoint failure recorded', {
      url: ep.url.slice(0, 30) + '...',
      failCount: ep.failCount,
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
export function recordRpcSuccess(): void {
  const ep = endpoints.find(e => e.url === activeEndpointUrl);
  if (ep && ep.failCount > 0) {
    ep.failCount = 0;
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

/**
 * Reset connections (for testing)
 */
export function resetSolanaConnection(): void {
  activeConnection = null;
  activeEndpointUrl = '';
  endpoints = [];
}
