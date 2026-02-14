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
import { CRTXAgent } from './agents/crtxAgent.js';
import { logger } from './services/logger.js';
import { validateAgentConfig } from './config/production.js';

async function main() {
  console.log('\n╔═══════════════════════════════════════════════════════════╗');
  console.log('║  🤖 CORTEX CRTX AGENT - PRODUCTION STARTUP               ║');
  console.log('╚═══════════════════════════════════════════════════════════╝\n');

  try {
    const portfolioValueUsd = parseFloat(process.env.PORTFOLIO_VALUE_USD || '100');
    const minConfidence = 0.60;

    // Validate config before creating agent
    const configCheck = validateAgentConfig({ minConfidence, portfolioValueUsd });
    if (!configCheck.valid) {
      console.error('❌ Invalid agent configuration:');
      configCheck.errors.forEach(err => console.error(`   - ${err}`));
      process.exit(1);
    }

    // Create agent with interactive mode selection
    console.log('Initializing agent with interactive mode selection...\n');

    const agent = await CRTXAgent.createInteractive({
      portfolioValueUsd,
      minConfidence,
      minRiskAdjustedReturn: 1.0,
      dryRun: false,
      volatility24h: 0.05,
    });

    console.log('\n✅ Agent initialized successfully!\n');

    // Display wallet assets dynamically from blockchain
    console.log('[CRTX] 💰 Fetching wallet assets from Solana blockchain...');
    await agent.displayWalletAssets();

    // Display strategy allocation based on wallet value
    console.log('[CRTX] 📊 Calculating strategy allocation...');
    await agent.displayStrategyAllocation();

    console.log('Starting continuous trading loop...');
    console.log('Starting LP position monitoring...');
    console.log('Starting Spot position monitoring...');
    console.log('Starting Lending position monitoring...');
    console.log('Press Ctrl+C to stop\n');

    // Start LP position monitoring
    agent.startLPMonitoring();

    // Start Spot position monitoring
    agent.startSpotMonitoring();

    // Start Lending position monitoring
    agent.startLendingMonitoring();

    // Handle graceful shutdown
    process.on('SIGINT', () => {
      console.log('\n\n🛑 Shutting down gracefully...');

      // Stop LP monitoring
      agent.stopLPMonitoring();

      // Stop Spot monitoring
      agent.stopSpotMonitoring();

      // Stop Lending monitoring
      agent.stopLendingMonitoring();

      const state = agent.getRiskState();
      console.log('\n📊 Final State:');
      console.log(`   Daily PnL: ${state.dailyPnL.toFixed(2)}%`);
      console.log(`   Daily Trades: ${state.dailyTradeCount}`);
      console.log(`   Current Position: ${state.currentPositionPct.toFixed(2)}%`);
      console.log('\n👋 Goodbye!\n');
      process.exit(0);
    });

    // Run continuous trading loop
    const runCycle = async () => {
      try {
        console.log('\n[CRTX] 🔄 Starting trading cycle...');
        const result = await agent.run();

        if (result) {
          console.log('\n[CRTX] ✅ Opportunity found:');
          console.log(`   Type: ${result.type}`);
          console.log(`   Name: ${result.name}`);
          console.log(`   Expected Return: +${result.expectedReturn.toFixed(2)}%`);
          console.log(`   Risk-Adjusted Return: +${result.riskAdjustedReturn.toFixed(2)}%`);
          console.log(`   Confidence: ${(result.confidence * 100).toFixed(0)}%`);
          console.log(`   Risk Score: ${result.riskScore}/10`);
          
          if (result.warnings && result.warnings.length > 0) {
            console.log(`   ⚠️  Warnings: ${result.warnings.join(', ')}`);
          }
        } else {
          console.log('\n[CRTX] ℹ️  No opportunities found this cycle');
        }

        // Show risk state
        const state = agent.getRiskState();
        console.log(`\n[CRTX] 📊 Risk State:`);
        console.log(`   Daily PnL: ${state.dailyPnL.toFixed(2)}%`);
        console.log(`   Daily Trades: ${state.dailyTradeCount}`);
        console.log(`   Current Position: ${state.currentPositionPct.toFixed(2)}%`);

      } catch (error: any) {
        logger.error('[CRTX] Error in trading cycle', {
          error: error.message,
          stack: error.stack,
        });
        console.error('\n[CRTX] ❌ Error:', error.message);
      }
    };

    // Initial run
    await runCycle();

    // Periodic runs
    const tradingCycleMs = parseInt(process.env.TRADING_CYCLE_MS || '60000', 10);
    console.log(`\n[CRTX] ⏰ Next cycle in ${tradingCycleMs / 1000}s...\n`);
    setInterval(runCycle, tradingCycleMs);

  } catch (error: any) {
    logger.error('[CRTX] Fatal error during startup', {
      error: error.message,
      stack: error.stack,
    });
    console.error('\n❌ Fatal error:', error.message);
    console.error('\nStack trace:', error.stack);
    process.exit(1);
  }
}

// Start the agent
main().catch((error) => {
  console.error('\n❌ Unhandled error:', error);
  process.exit(1);
});

