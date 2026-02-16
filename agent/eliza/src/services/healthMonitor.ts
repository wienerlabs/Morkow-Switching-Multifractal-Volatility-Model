/**
 * Health Monitor — centralized agent health state.
 *
 * Tracks:
 * - Trading loop heartbeat (last cycle, consecutive errors)
 * - Model readiness (A-LAMS calibration)
 * - Circuit breaker state
 * - RPC endpoint health
 * - Redis connectivity
 *
 * Used by the /health and /ready endpoints in start.ts.
 */

import { getHealthMetrics } from './solana/connection.js';
import { isRedisHealthStoreAvailable } from './solana/rpcHealthStore.js';

export interface HealthCheck {
  healthy: boolean;
  status: 'healthy' | 'degraded' | 'unhealthy';
  checks: {
    tradingLoop: { ok: boolean; lastCycleAt: string | null; consecutiveErrors: number; totalCycles: number };
    rpc: { ok: boolean; activeEndpoint: string; allDown: boolean };
    redis: { ok: boolean; connected: boolean };
    calibration: { ok: boolean; calibrated: boolean; calibratedAt: string | null };
    circuitBreaker: { ok: boolean; state: string };
  };
  uptimeMs: number;
}

// Module-level state
const state = {
  startedAt: Date.now(),

  // Trading loop
  lastCycleAt: null as number | null,
  consecutiveErrors: 0,
  totalCycles: 0,
  isRunning: false,

  // Calibration
  alamsCalibrated: false,
  calibrationTimestamp: null as number | null,

  // Circuit breaker
  circuitBreakerState: 'ACTIVE' as string,

  // RPC
  allRpcDown: false,
  activeRpcEndpoint: '',

  // Redis
  redisConnected: false,
};

// ── Mutations ──

export function recordCycleStart(): void {
  state.isRunning = true;
}

export function recordCycleSuccess(): void {
  state.lastCycleAt = Date.now();
  state.consecutiveErrors = 0;
  state.totalCycles++;
  state.isRunning = false;
}

export function recordCycleError(): void {
  state.consecutiveErrors++;
  state.totalCycles++;
  state.isRunning = false;
}

export function setCalibrationStatus(calibrated: boolean): void {
  state.alamsCalibrated = calibrated;
  if (calibrated) {
    state.calibrationTimestamp = Date.now();
  }
}

export function setCircuitBreakerState(cbState: string): void {
  state.circuitBreakerState = cbState;
}

export function setRpcHealth(activeEndpoint: string, allDown: boolean): void {
  state.activeRpcEndpoint = activeEndpoint;
  state.allRpcDown = allDown;
}

export function setRedisConnected(connected: boolean): void {
  state.redisConnected = connected;
}

// ── Health Check ──

const STALE_LOOP_MS = 5 * 60 * 1000; // 5 minutes
const MAX_CONSECUTIVE_ERRORS = 10;

export function performHealthCheck(): HealthCheck {
  // Refresh RPC state from connection module
  const rpcReport = getHealthMetrics();
  state.activeRpcEndpoint = rpcReport.activeEndpoint;
  state.allRpcDown = rpcReport.status === 'down';

  // Refresh Redis state
  state.redisConnected = isRedisHealthStoreAvailable();

  const now = Date.now();

  // Trading loop check
  const loopStale = state.lastCycleAt !== null && (now - state.lastCycleAt) > STALE_LOOP_MS;
  const tooManyErrors = state.consecutiveErrors >= MAX_CONSECUTIVE_ERRORS;
  const tradingLoopOk = !loopStale && !tooManyErrors;

  // RPC check
  const rpcOk = !state.allRpcDown;

  // Redis check — not critical, just informational
  const redisOk = state.redisConnected;

  // Calibration — not critical
  const calibrationOk = true; // Always OK — calibration is best-effort

  // Circuit breaker check
  const cbOk = state.circuitBreakerState !== 'LOCKDOWN';

  // Overall status
  let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
  const healthy = tradingLoopOk && rpcOk && cbOk;

  if (!healthy) {
    status = 'unhealthy';
  } else if (!redisOk || state.circuitBreakerState !== 'ACTIVE') {
    status = 'degraded';
  }

  return {
    healthy,
    status,
    checks: {
      tradingLoop: {
        ok: tradingLoopOk,
        lastCycleAt: state.lastCycleAt ? new Date(state.lastCycleAt).toISOString() : null,
        consecutiveErrors: state.consecutiveErrors,
        totalCycles: state.totalCycles,
      },
      rpc: {
        ok: rpcOk,
        activeEndpoint: state.activeRpcEndpoint,
        allDown: state.allRpcDown,
      },
      redis: {
        ok: redisOk,
        connected: state.redisConnected,
      },
      calibration: {
        ok: calibrationOk,
        calibrated: state.alamsCalibrated,
        calibratedAt: state.calibrationTimestamp ? new Date(state.calibrationTimestamp).toISOString() : null,
      },
      circuitBreaker: {
        ok: cbOk,
        state: state.circuitBreakerState,
      },
    },
    uptimeMs: now - state.startedAt,
  };
}
