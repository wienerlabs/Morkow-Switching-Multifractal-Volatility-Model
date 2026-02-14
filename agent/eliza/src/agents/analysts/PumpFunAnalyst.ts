/**
 * PumpFunAnalyst - Pump.fun Memecoin Evaluator
 *
 * Fetches and evaluates tokens from Pump.fun API
 * Only active in AGGRESSIVE trading mode
 *
 * Features:
 * - Risk analysis via PumpFun risk checker
 * - Executable trade proposals with buy/sell functions
 * - Exit strategy integration
 */

import { BaseAnalyst, type Opportunity, type AnalystConfig, DEFAULT_ANALYST_CONFIG } from './BaseAnalyst.js';
import type { ModeConfig } from '../../config/tradingModes.js';
import { PumpFunClient, type TransactionResult } from '../../services/pumpfun/pumpfunClient.js';
import { filterPumpFunTokens, type FilteredPumpFunToken } from '../../services/pumpfun/pumpfunFilter.js';
import { checkPumpFunRisks, type PumpFunRiskCheck } from '../../services/pumpfun/riskChecker.js';
import { loadPumpFunConfig } from '../../services/pumpfun/configLoader.js';
import { logger } from '../../services/logger.js';

// ============= TYPES =============

export interface PumpFunAnalysisInput {
  limit?: number;
  offset?: number;
}

export interface PumpFunOpportunityResult extends Opportunity {
  type: 'pumpfun';
  token: FilteredPumpFunToken;
  riskCheck: PumpFunRiskCheck;
  trading: {
    direction: 'LONG';
    leverage: 1;
    entryPrice: number;
    stopLoss: number;
    takeProfit1: number;
    takeProfit2: number;
    takeProfit3: number;
    positionSizePct: number;
    amountSol: number;
    slippageBps: number;
  };
  /** Execute this trade - returns transaction result */
  execute: () => Promise<TransactionResult>;
}

export interface PumpFunAnalystConfig extends AnalystConfig {
  minConfidenceThreshold: number;
  maxTokens: number;
}

export const DEFAULT_PUMPFUN_CONFIG: PumpFunAnalystConfig = {
  ...DEFAULT_ANALYST_CONFIG,
  minConfidenceThreshold: 0.35,  // 35% minimum (lower for memecoins)
  maxTokens: 10,                  // Max 10 Pump.fun tokens to evaluate
};

// ============= ANALYST CLASS =============

export class PumpFunAnalyst extends BaseAnalyst<PumpFunAnalysisInput, PumpFunOpportunityResult> {
  private pumpfunConfig: PumpFunAnalystConfig;
  private pumpfunClient: PumpFunClient;
  private modeConfig: ModeConfig;

  constructor(modeConfig: ModeConfig, config: Partial<PumpFunAnalystConfig> = {}) {
    super({ ...DEFAULT_PUMPFUN_CONFIG, ...config });
    this.pumpfunConfig = { ...DEFAULT_PUMPFUN_CONFIG, ...config };
    this.modeConfig = modeConfig;

    // Initialize Pump.fun client
    this.pumpfunClient = new PumpFunClient();

    if (!modeConfig.enablePumpFun) {
      logger.info('[PumpFunAnalyst] PumpFun disabled in current mode (requires AGGRESSIVE)');
    } else {
      logger.info('[PumpFunAnalyst] Initialized', {
        config: this.pumpfunConfig,
        mode: modeConfig.mode,
        minTVL: modeConfig.minTVL,
        minHolders: modeConfig.minHolders,
      });
    }
  }

  getName(): string {
    return 'PumpFunAnalyst';
  }

  /**
   * Analyze Pump.fun tokens and return opportunities
   */
  async analyze(input: PumpFunAnalysisInput = {}): Promise<PumpFunOpportunityResult[]> {
    if (!this.modeConfig.enablePumpFun) {
      return [];
    }
    const { limit = 50, offset = 0 } = input;

    try {
      // 1. Fetch tokens from Pump.fun API
      logger.info('[PumpFunAnalyst] Fetching tokens from Pump.fun', { limit, offset });
      const tokens = await this.pumpfunClient.getTokens(limit, offset);
      logger.info('[PumpFunAnalyst] Fetched tokens', { count: tokens.length });

      // 2. Filter by safety criteria
      const filtered = await filterPumpFunTokens(tokens, {
        minTvl: this.modeConfig.minTVL,
        minHolders: this.modeConfig.minHolders,
        minAgeHours: 24,
        maxTopHolderPct: 50,
      });
      logger.info('[PumpFunAnalyst] Filtered tokens', { count: filtered.length });

      // 3. Evaluate each token
      const opportunities: PumpFunOpportunityResult[] = [];
      for (const token of filtered.slice(0, this.pumpfunConfig.maxTokens)) {
        const opportunity = await this.evaluateToken(token);
        if (opportunity) {
          opportunities.push(opportunity);
        }
      }

      logger.info('[PumpFunAnalyst] Evaluated opportunities', { count: opportunities.length });
      return opportunities;

    } catch (error: any) {
      logger.error('[PumpFunAnalyst] Analysis failed', { error: error.message });
      return [];
    }
  }

  /**
   * Evaluate a single Pump.fun token
   */
  private async evaluateToken(token: FilteredPumpFunToken): Promise<PumpFunOpportunityResult | null> {
    // Load PumpFun config for position sizing and slippage
    const pumpfunTradeConfig = loadPumpFunConfig();

    // Fetch full token data for risk analysis
    const fullToken = await this.pumpfunClient.getTokenByMint(token.mint);
    if (!fullToken) {
      logger.warn('[PumpFunAnalyst] Could not fetch full token data', { mint: token.mint });
      return null;
    }

    // Perform comprehensive risk analysis
    const riskCheck = await checkPumpFunRisks(fullToken);

    // Reject tokens that fail risk checks
    if (riskCheck.isRugPull) {
      logger.info('[PumpFunAnalyst] Token rejected due to rug pull risk', {
        symbol: token.symbol,
        riskScore: riskCheck.riskScore,
        flags: riskCheck.riskFlags,
      });
      return null;
    }

    // Calculate confidence score based on token metrics
    const confidence = this.calculateConfidence(token);

    // Check if meets minimum threshold
    if (confidence < this.pumpfunConfig.minConfidenceThreshold) {
      return null;
    }

    // Calculate risk score (1-10, lower is safer) - incorporate risk check
    const baseRiskScore = this.calculateRiskScore(token);
    const riskScore = Math.min(10, baseRiskScore + (riskCheck.riskScore / 20)); // Add up to 5 from risk check

    // Calculate expected return (memecoins are volatile)
    const expectedReturn = 0.40; // 40% target for memecoins

    // Calculate risk-adjusted return
    const riskAdjustedReturn = expectedReturn / (riskScore / 10);

    // Calculate position size (smaller for memecoins)
    const basePositionPct = 0.02; // 2% base
    const positionSizePct = Math.min(
      basePositionPct * confidence,
      this.modeConfig.maxPositionSize
    );

    // Calculate actual SOL amount (capped by max position size)
    const amountSol = Math.min(
      pumpfunTradeConfig.maxPositionSol * positionSizePct * 10, // Scale position pct
      pumpfunTradeConfig.maxPositionSol
    );

    // Get slippage from config (default 5%)
    const slippageBps = 500;

    // Estimate entry price (use market cap / total supply)
    const entryPrice = token.marketCap / 1_000_000_000; // Rough estimate

    // Create execute function that buys the token
    const tokenMint = token.mint;
    const client = this.pumpfunClient;
    const execute = async (): Promise<TransactionResult> => {
      logger.info('[PumpFunAnalyst] Executing buy', { mint: tokenMint, amountSol, slippageBps });
      return client.buy({
        tokenMint,
        amountSol,
        slippageBps,
      });
    };

    // Build warnings list
    const warnings = [
      '⚠️ PUMP.FUN MEMECOIN: Extremely high risk',
      `⚠️ Age: ${token.ageHours.toFixed(0)}h (very new)`,
      `⚠️ Holders: ${token.holderCount} (low distribution)`,
      ...riskCheck.warnings,
    ];

    return {
      type: 'pumpfun',
      name: `${token.symbol} (Pump.fun)`,
      expectedReturn,
      riskScore,
      confidence,
      riskAdjustedReturn,
      approved: true,
      warnings,
      raw: token,
      token,
      riskCheck,
      trading: {
        direction: 'LONG',
        leverage: 1,
        entryPrice,
        stopLoss: entryPrice * 0.50,     // -50% stop loss (from config)
        takeProfit1: entryPrice * 2.00,  // +100% (2x)
        takeProfit2: entryPrice * 3.00,  // +200% (3x)
        takeProfit3: entryPrice * 6.00,  // +500% (6x)
        positionSizePct,
        amountSol,
        slippageBps,
      },
      execute,
    };
  }

  /**
   * Calculate confidence score based on token metrics
   */
  private calculateConfidence(token: FilteredPumpFunToken): number {
    let score = 0.3; // Base score for passing filters

    // TVL bonus (up to +0.2)
    if (token.tvl > 50_000) score += 0.1;
    if (token.tvl > 100_000) score += 0.1;

    // Holder count bonus (up to +0.2)
    if (token.holderCount > 100) score += 0.1;
    if (token.holderCount > 200) score += 0.1;

    // Age bonus (up to +0.1)
    if (token.ageHours > 48) score += 0.05;
    if (token.ageHours > 72) score += 0.05;

    // Social presence bonus (up to +0.2)
    if (token.twitter) score += 0.1;
    if (token.telegram) score += 0.05;
    if (token.website) score += 0.05;

    return Math.min(score, 0.9); // Cap at 90%
  }

  /**
   * Calculate risk score (1-10, lower is safer)
   */
  private calculateRiskScore(token: FilteredPumpFunToken): number {
    let risk = 8; // Base risk for memecoins

    // Reduce risk for higher TVL
    if (token.tvl > 50_000) risk -= 0.5;
    if (token.tvl > 100_000) risk -= 0.5;

    // Reduce risk for more holders
    if (token.holderCount > 100) risk -= 0.5;
    if (token.holderCount > 200) risk -= 0.5;

    // Reduce risk for older tokens
    if (token.ageHours > 48) risk -= 0.5;
    if (token.ageHours > 72) risk -= 0.5;

    return Math.max(risk, 5); // Minimum risk of 5 for memecoins
  }
}

