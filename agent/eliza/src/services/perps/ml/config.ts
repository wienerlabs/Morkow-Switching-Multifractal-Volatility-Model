/**
 * Perps ML Trading Agent Configuration
 * 
 * Environment-specific configurations for the ML-powered perps trading agent.
 * Includes devnet (testing) and mainnet (production) presets.
 */
import type { TradingAgentConfig } from './tradingAgent.js';

// ============= ENVIRONMENT CONFIGS =============

/**
 * DEVNET Configuration - Lower thresholds for testing
 *
 * - Small position sizes ($10)
 * - Low funding threshold (0.05%) to trigger trades
 * - 50% confidence threshold for testing
 * - Single market (SOL-PERP)
 * - Drift venue (best devnet support)
 */
export const DEVNET_CONFIG: TradingAgentConfig = {
  // Polling: every 10 seconds base interval
  pollingIntervalMs: 10_000,
  // Venue-specific intervals for position monitoring
  jupiterPollingIntervalMs: 5_000,   // 5s - Jupiter has no native SL/TP
  driftPollingIntervalMs: 30_000,    // 30s - Drift has on-chain SL/TP
  adrenaPollingIntervalMs: 30_000,   // 30s - Adrena has on-chain SL/TP
  flashPollingIntervalMs: 60_000,    // 60s - Flash is simulated

  // LOW thresholds for devnet testing
  minConfidence: 0.50,              // 50% confidence threshold
  fundingThreshold: 0.0005,         // 0.05% - very low to trigger trades

  // Small position sizes for testing
  positionSizeUsd: 10,              // $10 per position
  leverage: 1,                      // No leverage for safety

  // Single market
  markets: ['SOL-PERP'],

  // Drift has best devnet support
  preferredVenue: 'drift',

  // REAL TRADING ON DEVNET
  dryRun: false,

  // Limit concurrent positions
  maxPositions: 2,

  // Position close retry settings
  maxCloseRetries: 3,

  // Default SL/TP percentages (can be overridden per position)
  defaultStopLossPercent: 0.05,     // -5% stop loss
  defaultTakeProfitPercent: 0.10,   // +10% take profit
};

/**
 * MAINNET Configuration - Production settings
 * 
 * - Larger positions ($100-500)
 * - Standard confidence (0.65+)
 * - Multiple markets
 * - MEV protection enabled
 */
export const MAINNET_CONFIG: TradingAgentConfig = {
  // Polling: 10s base interval for responsive monitoring
  pollingIntervalMs: 10_000,
  // Venue-specific intervals - more aggressive for Jupiter
  jupiterPollingIntervalMs: 5_000,   // 5s - critical, no native SL/TP
  driftPollingIntervalMs: 30_000,    // 30s - backup for on-chain SL/TP
  adrenaPollingIntervalMs: 30_000,   // 30s - backup for on-chain SL/TP
  flashPollingIntervalMs: 60_000,    // 60s - simulated, low priority

  minConfidence: 0.65,              // 65% confidence for production
  fundingThreshold: 0.0025,         // 0.25% threshold

  positionSizeUsd: 20,  // $20 minimum for mainnet
  leverage: 1,                      // 1x leverage for safety

  markets: ['SOL-PERP', 'BTC-PERP', 'ETH-PERP'],
  preferredVenue: 'drift',

  dryRun: false,
  maxPositions: 5,

  // Position close retry settings
  maxCloseRetries: 3,

  // Default SL/TP percentages (can be overridden per position)
  defaultStopLossPercent: 0.05,     // -5% stop loss
  defaultTakeProfitPercent: 0.10,   // +10% take profit
};

/**
 * DRY RUN Configuration - For testing without execution
 */
export const DRY_RUN_CONFIG: TradingAgentConfig = {
  // Polling: 10s base for faster testing
  pollingIntervalMs: 10_000,
  // Venue-specific intervals
  jupiterPollingIntervalMs: 5_000,   // 5s - simulating Jupiter monitoring
  driftPollingIntervalMs: 30_000,    // 30s - Drift backup
  adrenaPollingIntervalMs: 30_000,   // 30s - Adrena backup
  flashPollingIntervalMs: 60_000,    // 60s - Flash is simulated

  minConfidence: 0.50,              // Lower threshold to see more signals
  fundingThreshold: 0.0010,         // 0.10% to see more opportunities

  positionSizeUsd: 20,  // $20 minimum
  leverage: 1,

  markets: ['SOL-PERP'],
  preferredVenue: 'drift',

  dryRun: true,                     // No actual trades
  maxPositions: 10,

  // Position close retry settings
  maxCloseRetries: 3,

  // Default SL/TP percentages (can be overridden per position)
  defaultStopLossPercent: 0.05,     // -5% stop loss
  defaultTakeProfitPercent: 0.10,   // +10% take profit
};

// ============= HELPER =============

export type Environment = 'devnet' | 'mainnet' | 'dryrun';

export function getConfig(env: Environment): TradingAgentConfig {
  switch (env) {
    case 'devnet':
      return DEVNET_CONFIG;
    case 'mainnet':
      return MAINNET_CONFIG;
    case 'dryrun':
    default:
      return DRY_RUN_CONFIG;
  }
}

/**
 * Merge custom config with environment defaults
 */
export function createConfig(
  env: Environment,
  overrides: Partial<TradingAgentConfig> = {}
): TradingAgentConfig {
  const base = getConfig(env);
  return { ...base, ...overrides };
}

