#!/usr/bin/env tsx
/**
 * CRTX Agent - Main Startup Script
 * 
 * Starts the CRTX trading agent with interactive mode selection.
 * User can choose between NORMAL (conservative) and AGGRESSIVE (higher risk) modes.
 * 
 * Usage:
 *   npm start
 *   npm run start:agent
 * 
 * Or directly:
 *   npx tsx src/start.ts
 * 
 * Environment Variables:
 *   TRADING_MODE - Set to 'NORMAL' or 'AGGRESSIVE' to skip interactive prompt
 */

import 'dotenv/config';
import { createServer } from 'node:http';
import { CRTXAgent } from './agents/crtxAgent.js';
import { logger } from './services/logger.js';
import { validateAgentConfig } from './config/production.js';
import { getHealthMetrics, resetHealthMetrics } from './services/solana/connection.js';
import { initCalibration, stopCalibrationSchedule } from './services/calibration.js';

async function main() {
  logger.info('\n╔═══════════════════════════════════════════════════════════╗');
  logger.info('║  🤖 CORTEX CRTX AGENT - PRODUCTION STARTUP               ║');
  logger.info('╚═══════════════════════════════════════════════════════════╝\n');

  try {
    const portfolioValueUsd = parseFloat(process.env.PORTFOLIO_VALUE_USD || '100');
    const minConfidence = 0.60;

    // Validate config before creating agent
    const configCheck = validateAgentConfig({ minConfidence, portfolioValueUsd });
    if (!configCheck.valid) {
      logger.error('❌ Invalid agent configuration:');
      configCheck.errors.forEach(err => logger.error(`   - ${err}`));
      process.exit(1);
    }

    // Create agent with interactive mode selection
    logger.info('Initializing agent with interactive mode selection...\n');

    const agent = await CRTXAgent.createInteractive({
      portfolioValueUsd,
      minConfidence,
      minRiskAdjustedReturn: 1.0,
      dryRun: false,
      volatility24h: 0.05,
    });

    logger.info('\n✅ Agent initialized successfully!\n');

    // Calibrate A-LAMS model (fire-and-forget — does not block startup)
    initCalibration().catch((err) => {
      logger.error('[Calibration] Unexpected error during init', { error: String(err) });
    });

    // Display wallet assets dynamically from blockchain
    logger.info('[CRTX] 💰 Fetching wallet assets from Solana blockchain...');
    await agent.displayWalletAssets();

    // Display strategy allocation based on wallet value
    logger.info('[CRTX] 📊 Calculating strategy allocation...');
    await agent.displayStrategyAllocation();

    logger.info('Starting continuous trading loop...');
    logger.info('Starting LP position monitoring...');
    logger.info('Starting Spot position monitoring...');
    logger.info('Starting Lending position monitoring...');
    logger.info('Press Ctrl+C to stop\n');

    // Start LP position monitoring
    agent.startLPMonitoring();

    // Start Spot position monitoring
    agent.startSpotMonitoring();

    // Start Lending position monitoring
    agent.startLendingMonitoring();

    // Start lightweight health monitoring server
    const healthPort = parseInt(process.env.HEALTH_PORT || '9090', 10);
    const healthServer = createServer((req, res) => {
      if (req.method === 'GET' && req.url === '/health/rpc') {
        const report = getHealthMetrics();
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(report));
        return;
      }
      if (req.method === 'POST' && req.url === '/health/rpc/reset') {
        resetHealthMetrics();
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: true }));
        return;
      }
      res.writeHead(404);
      res.end();
    });
    healthServer.listen(healthPort, () => {
      logger.info(`[Health] RPC health endpoint at http://localhost:${healthPort}/health/rpc`);
    });

    // Handle graceful shutdown
    process.on('SIGINT', () => {
      logger.info('\n\n🛑 Shutting down gracefully...');
      healthServer.close();
      stopCalibrationSchedule();

      agent.stopLPMonitoring();
      agent.stopSpotMonitoring();
      agent.stopLendingMonitoring();

      const state = agent.getRiskState();
      logger.info('\n📊 Final State:');
      logger.info(`   Daily PnL: ${state.dailyPnL.toFixed(2)}%`);
      logger.info(`   Daily Trades: ${state.dailyTradeCount}`);
      logger.info(`   Current Position: ${state.currentPositionPct.toFixed(2)}%`);
      logger.info('\n👋 Goodbye!\n');
      process.exit(0);
    });

    // Run continuous trading loop
    const runCycle = async () => {
      try {
        logger.info('\n[CRTX] 🔄 Starting trading cycle...');
        const result = await agent.run();

        if (result) {
          logger.info('\n[CRTX] ✅ Opportunity found:');
          logger.info(`   Type: ${result.type}`);
          logger.info(`   Name: ${result.name}`);
          logger.info(`   Expected Return: +${result.expectedReturn.toFixed(2)}%`);
          logger.info(`   Risk-Adjusted Return: +${result.riskAdjustedReturn.toFixed(2)}%`);
          logger.info(`   Confidence: ${(result.confidence * 100).toFixed(0)}%`);
          logger.info(`   Risk Score: ${result.riskScore}/10`);

          if (result.warnings && result.warnings.length > 0) {
            logger.info(`   ⚠️  Warnings: ${result.warnings.join(', ')}`);
          }
        } else {
          logger.info('\n[CRTX] ℹ️  No opportunities found this cycle');
        }

        // Show risk state
        const state = agent.getRiskState();
        logger.info(`\n[CRTX] 📊 Risk State:`);
        logger.info(`   Daily PnL: ${state.dailyPnL.toFixed(2)}%`);
        logger.info(`   Daily Trades: ${state.dailyTradeCount}`);
        logger.info(`   Current Position: ${state.currentPositionPct.toFixed(2)}%`);

      } catch (error: any) {
        logger.error('[CRTX] Error in trading cycle', {
          error: error.message,
          stack: error.stack,
        });
      }
    };

    // Initial run
    await runCycle();

    // Periodic runs
    const tradingCycleMs = parseInt(process.env.TRADING_CYCLE_MS || '60000', 10);
    logger.info(`\n[CRTX] ⏰ Next cycle in ${tradingCycleMs / 1000}s...\n`);
    setInterval(runCycle, tradingCycleMs);

  } catch (error: any) {
    logger.error('[CRTX] Fatal error during startup', {
      error: error.message,
      stack: error.stack,
    });
    process.exit(1);
  }
}

// Start the agent
main().catch((error) => {
  logger.error('[CRTX] Unhandled error', { error: String(error) });
  process.exit(1);
});

