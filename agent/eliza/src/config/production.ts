/**
 * Production Configuration for Cortex Agent
 * 
 * Centralized configuration management for:
 * - Simulation vs Production mode
 * - RPC endpoints
 * - Wallet management
 * - Rate limits and retry logic
 */

import { logger } from '../services/logger.js';

export interface CortexConfig {
  // Mode
  simulationMode: boolean;
  
  // Solana
  solanaRpcUrl: string;
  solanaWsUrl: string;
  commitment: 'processed' | 'confirmed' | 'finalized';
  
  // API Keys
  birdeyeApiKey: string;
  jupiterApiKey?: string;
  heliusApiKey?: string;
  
  // Wallet (only in production)
  walletPrivateKey?: string;
  
  // Monitoring
  monitoringEnabled: boolean;
  monitoringIntervalMs: number;
  autoRebalanceEnabled: boolean;
  rebalanceThreshold: number;
  
  // Rate Limits
  birdeyeRpmLimit: number;
  jupiterRpmLimit: number;
  
  // Retry
  maxRetries: number;
  retryDelayMs: number;
  
  // Logging
  logLevel: 'debug' | 'info' | 'warn' | 'error';
  logToFile: boolean;
  logDir: string;
  
  // Cooldowns
  rebalanceCooldownMs: number;

  // Trading cycle
  tradingCycleMs: number;

  // Slippage configuration (basis points)
  slippage: {
    arbitrage: number;
    spot: number;
    lending: number;
    lpDeposit: number;
    lpWithdraw: number;
  };

  // Redis
  redis: {
    url: string;
    enabled: boolean;
  };
}

/**
 * Load configuration from environment variables
 */
export function loadConfig(): CortexConfig {
  const simulationMode = process.env.SIMULATION_MODE !== 'false';
  
  return {
    // Mode
    simulationMode,
    
    // Solana
    solanaRpcUrl: process.env.SOLANA_RPC_URL || 
      (simulationMode 
        ? 'https://api.devnet.solana.com' 
        : 'https://api.mainnet-beta.solana.com'),
    solanaWsUrl: process.env.SOLANA_WS_URL || 
      (simulationMode
        ? 'wss://api.devnet.solana.com'
        : 'wss://api.mainnet-beta.solana.com'),
    commitment: (process.env.SOLANA_COMMITMENT as CortexConfig['commitment']) || 'confirmed',
    
    // API Keys
    birdeyeApiKey: process.env.BIRDEYE_API_KEY || '',
    jupiterApiKey: process.env.JUPITER_API_KEY,
    heliusApiKey: process.env.HELIUS_API_KEY,
    
    // Wallet - only load in production mode
    walletPrivateKey: simulationMode ? undefined : process.env.SOLANA_PRIVATE_KEY,
    
    // Monitoring
    monitoringEnabled: process.env.MONITORING_ENABLED === 'true',
    monitoringIntervalMs: parseInt(process.env.MONITORING_INTERVAL_MS || '3600000', 10), // 1 hour
    autoRebalanceEnabled: process.env.AUTO_REBALANCE_ENABLED === 'true',
    rebalanceThreshold: parseFloat(process.env.REBALANCE_THRESHOLD || '0.90'),
    
    // Rate Limits
    birdeyeRpmLimit: parseInt(process.env.BIRDEYE_RPM_LIMIT || '60', 10),
    jupiterRpmLimit: parseInt(process.env.JUPITER_RPM_LIMIT || '600', 10),
    
    // Retry
    maxRetries: parseInt(process.env.MAX_RETRIES || '3', 10),
    retryDelayMs: parseInt(process.env.RETRY_DELAY_MS || '1000', 10),
    
    // Logging
    logLevel: (process.env.LOG_LEVEL as CortexConfig['logLevel']) || 'info',
    logToFile: process.env.LOG_TO_FILE !== 'false',
    logDir: process.env.LOG_DIR || './logs',
    
    // Cooldowns
    rebalanceCooldownMs: parseInt(process.env.REBALANCE_COOLDOWN_MS || '86400000', 10), // 24 hours

    // Trading cycle
    tradingCycleMs: parseInt(process.env.TRADING_CYCLE_MS || '60000', 10),

    // Slippage configuration (basis points)
    slippage: {
      arbitrage: parseInt(process.env.SLIPPAGE_ARBITRAGE_BPS || '50', 10),
      spot: parseInt(process.env.SLIPPAGE_SPOT_BPS || '100', 10),
      lending: parseInt(process.env.SLIPPAGE_LENDING_BPS || '50', 10),
      lpDeposit: parseInt(process.env.SLIPPAGE_LP_DEPOSIT_BPS || '100', 10),
      lpWithdraw: parseInt(process.env.SLIPPAGE_LP_WITHDRAW_BPS || '50', 10),
    },

    // Redis
    redis: {
      url: process.env.REDIS_URL || process.env.PERSISTENCE_REDIS_URL || '',
      enabled: process.env.REDIS_ENABLED !== 'false',
    },
  };
}

/**
 * Validate configuration
 */
export function validateConfig(config: CortexConfig): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  
  if (!config.birdeyeApiKey) {
    errors.push('BIRDEYE_API_KEY is required');
  }
  
  if (!config.simulationMode && !config.walletPrivateKey) {
    errors.push('SOLANA_PRIVATE_KEY is required in production mode');
  }
  
  if (config.rebalanceThreshold < 0 || config.rebalanceThreshold > 1) {
    errors.push('REBALANCE_THRESHOLD must be between 0 and 1');
  }
  
  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Print configuration summary (safe - no secrets)
 */
export function printConfigSummary(config: CortexConfig): void {
  logger.info('\n📋 Cortex Configuration:');
  logger.info('═'.repeat(40));
  logger.info(`  Mode: ${config.simulationMode ? '🎮 SIMULATION' : '🚀 PRODUCTION'}`);
  logger.info(`  RPC: ${config.solanaRpcUrl.slice(0, 30)}...`);
  logger.info(`  Birdeye API: ${config.birdeyeApiKey ? '✅' : '❌'}`);
  logger.info(`  Wallet: ${config.walletPrivateKey ? '✅ Configured' : '❌ Not set'}`);
  logger.info(`  Monitoring: ${config.monitoringEnabled ? '✅ Enabled' : '❌ Disabled'}`);
  logger.info(`  Auto-Rebalance: ${config.autoRebalanceEnabled ? '✅ Enabled' : '❌ Disabled'}`);
  logger.info(`  Threshold: ${(config.rebalanceThreshold * 100).toFixed(0)}%`);
  logger.info('═'.repeat(40));
}

/**
 * Validate agent startup configuration
 */
export function validateAgentConfig(agentConfig: { minConfidence: number; portfolioValueUsd: number }): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  if (agentConfig.minConfidence < 0.50) {
    errors.push(`minConfidence too low: ${agentConfig.minConfidence} (minimum: 0.50)`);
  }

  if (agentConfig.portfolioValueUsd < 1) {
    errors.push(`portfolioValueUsd too low: $${agentConfig.portfolioValueUsd}`);
  }

  return { valid: errors.length === 0, errors };
}

// Export singleton config
export const config = loadConfig();

