/**
 * FundamentalAnalyst
 * 
 * Evaluates token health using on-chain metrics from Helius API.
 * 
 * Extends BaseAnalyst pattern for consistency with other analysts.
 * 
 * Input: { tokens: string[], pools?: LPPool[] }
 * Output: FundamentalOpportunityResult[]
 * 
 * Scoring:
 * - Health score > 60 = APPROVED
 * - Health score 40-60 = CAUTION
 * - Health score < 40 = REJECTED
 */

import { BaseAnalyst, type AnalystConfig, DEFAULT_ANALYST_CONFIG } from './BaseAnalyst.js';
import { getHeliusClient } from '../../services/onchain/heliusClient.js';
import { getTokenHealthScorer } from '../../services/onchain/tokenHealthScorer.js';
import { logger } from '../../services/logger.js';
import { TOKEN_MINTS } from '../../services/lpExecutor/index.js';
import { getTradingMode, type ModeConfig } from '../../config/tradingModes.js';

// ============= TYPES =============

export interface FundamentalInput {
  tokens: string[]; // Token symbols (SOL, BTC, etc.)
  pools?: Array<{ name: string; tvl: number }>; // Optional LP pools for TVL data
}

export interface FundamentalOpportunityResult {
  type: 'fundamental';
  token: string;
  name: string;
  healthScore: number; // 0-100
  rating: 'EXCELLENT' | 'GOOD' | 'FAIR' | 'POOR';
  metrics: {
    holderConcentration: number;
    liquidityDepth: number;
    tokenAge: number;
    whaleActivity: number;
  };
  details: {
    holderCount: number; // -1 = too many to count, 0+ = actual count
    topHoldersPercentage: number;
    tvlUsd: number;
    ageInDays: number;
    whaleTransferCount: number;
    isHighlyDistributed: boolean;
  };
  approved: boolean;
  rejectReason?: string;
  warnings: string[];
  expectedReturn: number; // Not applicable for fundamental analysis
  riskScore: number; // Inverse of health score
  confidence: number; // Based on data availability
  riskAdjustedReturn: number;
  raw: any; // Raw on-chain data
}

export interface FundamentalAnalystConfig extends AnalystConfig {
  minHealthScore: number; // Minimum health score to approve
  cautionThreshold: number; // Threshold for caution warnings
  tradingMode?: ModeConfig; // Trading mode override (optional)
}

// ============= DEFAULT CONFIG =============

export const DEFAULT_FUNDAMENTAL_CONFIG: FundamentalAnalystConfig = {
  ...DEFAULT_ANALYST_CONFIG,
  minHealthScore: 60,
  cautionThreshold: 40,
};

// ============= ANALYST =============

export class FundamentalAnalyst extends BaseAnalyst<FundamentalInput, FundamentalOpportunityResult> {
  private heliusClient = getHeliusClient();
  private healthScorer = getTokenHealthScorer();
  protected config: FundamentalAnalystConfig;
  private tradingMode: ModeConfig;

  constructor(config: Partial<FundamentalAnalystConfig> = {}) {
    super();
    this.config = { ...DEFAULT_FUNDAMENTAL_CONFIG, ...config };

    // Get trading mode from config or environment
    this.tradingMode = config.tradingMode || getTradingMode();

    // Override minHealthScore with trading mode threshold if not explicitly set
    if (!config.minHealthScore) {
      this.config.minHealthScore = this.tradingMode.minHealthScore;
    }

    logger.info('[FundamentalAnalyst] Initialized', {
      config: {
        minHealthScore: this.config.minHealthScore,
        cautionThreshold: this.config.cautionThreshold,
        verbose: this.config.verbose,
      },
      tradingMode: {
        mode: this.tradingMode.mode,
        minHealthScore: this.tradingMode.minHealthScore,
        enablePumpFun: this.tradingMode.enablePumpFun,
        riskMultiplier: this.tradingMode.riskMultiplier,
      },
    });
  }

  /**
   * Get analyst name
   */
  getName(): string {
    return 'FundamentalAnalyst';
  }

  /**
   * Analyze token fundamentals
   */
  async analyze(input: FundamentalInput): Promise<FundamentalOpportunityResult[]> {
    const results: FundamentalOpportunityResult[] = [];

    logger.info('[FundamentalAnalyst] Starting analysis', {
      tokenCount: input.tokens.length,
      tokens: input.tokens,
    });

    for (const token of input.tokens) {
      try {
        const result = await this.evaluateToken(token, input.pools);
        results.push(result);

        if (this.config.verbose) {
          this.logEvaluation(result);
        }
      } catch (error: any) {
        logger.error('[FundamentalAnalyst] Token evaluation failed', {
          token,
          error: error.message,
        });

        // Return rejected result on error
        results.push(this.createRejectedResult(token, `Evaluation failed: ${error.message}`));
      }
    }

    logger.info('[FundamentalAnalyst] Analysis complete', {
      total: results.length,
      approved: results.filter(r => r.approved).length,
      rejected: results.filter(r => !r.approved).length,
    });

    return results;
  }

  /**
   * Evaluate a single token
   */
  private async evaluateToken(
    token: string,
    pools?: Array<{ name: string; tvl: number }>
  ): Promise<FundamentalOpportunityResult> {
    // Get mint address for token
    const mintAddress = this.getMintAddress(token);
    
    if (!mintAddress) {
      return this.createRejectedResult(token, 'Unknown token mint address');
    }

    // Fetch on-chain data
    const onChainData = await this.heliusClient.fetchTokenData(mintAddress);

    if (!onChainData) {
      return this.createRejectedResult(token, 'Failed to fetch on-chain data');
    }

    // Get TVL from pools if available
    const tvlUsd = this.getTvlForToken(token, pools);

    // Calculate health score
    const healthScore = this.healthScorer.calculateHealthScore(onChainData, tvlUsd);

    // Determine approval
    const approved = healthScore.totalScore >= this.config.minHealthScore;
    const warnings: string[] = [];
    let rejectReason: string | undefined;

    if (!approved) {
      if (healthScore.totalScore < this.config.cautionThreshold) {
        rejectReason = `Health score too low: ${healthScore.totalScore}/100 (min: ${this.config.minHealthScore})`;
      } else {
        rejectReason = `Health score below threshold: ${healthScore.totalScore}/100 (min: ${this.config.minHealthScore})`;
        warnings.push('Token health is marginal');
      }
    }

    // Add AGGRESSIVE MODE warning if health is below normal threshold
    if (this.tradingMode.mode === 'AGGRESSIVE' && approved && healthScore.totalScore < 60) {
      warnings.push(`⚠️ AGGRESSIVE MODE: Health score ${healthScore.totalScore} below normal threshold (60). Higher risk of loss.`);
    }

    // Add specific warnings based on metrics
    if (healthScore.details.topHoldersPercentage > 70) {
      warnings.push('High holder concentration (>70%)');
    }
    if (healthScore.details.tvlUsd < 1_000_000) {
      warnings.push('Low liquidity (<$1M TVL)');
    }
    if (healthScore.details.ageInDays < 90) {
      warnings.push('New token (<3 months old)');
    }
    if (healthScore.details.whaleTransferCount > 2) {
      warnings.push('High whale activity (>2 large transfers in 7d)');
    }

    // Calculate risk score (inverse of health score)
    const riskScore = 10 - (healthScore.totalScore / 10);

    // Confidence based on data availability
    const confidence = onChainData.holders.length > 0 ? 0.9 : 0.5;

    return {
      type: 'fundamental',
      token,
      name: `${token} Fundamental Analysis`,
      healthScore: healthScore.totalScore,
      rating: healthScore.rating,
      metrics: healthScore.metrics,
      details: healthScore.details,
      approved,
      rejectReason,
      warnings,
      expectedReturn: 0, // Not applicable for fundamental analysis
      riskScore,
      confidence,
      riskAdjustedReturn: 0,
      raw: onChainData,
    };
  }

  /**
   * Get mint address for token symbol
   */
  private getMintAddress(token: string): string | null {
    const mint = TOKEN_MINTS[token as keyof typeof TOKEN_MINTS];
    return mint || null;
  }

  /**
   * Get TVL for token from pools
   */
  private getTvlForToken(token: string, pools?: Array<{ name: string; tvl: number }>): number {
    if (!pools) {
      return 0;
    }

    // Sum TVL from all pools containing this token
    const totalTvl = pools
      .filter(pool => pool.name.includes(token))
      .reduce((sum, pool) => sum + pool.tvl, 0);

    return totalTvl;
  }

  /**
   * Create rejected result
   */
  private createRejectedResult(token: string, reason: string): FundamentalOpportunityResult {
    return {
      type: 'fundamental',
      token,
      name: `${token} Fundamental Analysis`,
      healthScore: 0,
      rating: 'POOR',
      metrics: {
        holderConcentration: 0,
        liquidityDepth: 0,
        tokenAge: 0,
        whaleActivity: 0,
      },
      details: {
        holderCount: 0,
        topHoldersPercentage: 0,
        tvlUsd: 0,
        ageInDays: 0,
        whaleTransferCount: 0,
        isHighlyDistributed: false,
      },
      approved: false,
      rejectReason: reason,
      warnings: [],
      expectedReturn: 0,
      riskScore: 10,
      confidence: 0,
      riskAdjustedReturn: 0,
      raw: null,
    };
  }

  /**
   * Log evaluation result
   */
  private logEvaluation(result: FundamentalOpportunityResult): void {
    const emoji = result.approved ? '✅' : '❌';
    const ratingEmoji = {
      EXCELLENT: '🌟',
      GOOD: '👍',
      FAIR: '⚠️',
      POOR: '❌',
    }[result.rating];

    // Format holder count display
    const holderCountDisplay = result.details.holderCount === -1
      ? '>1M (highly distributed)'
      : result.details.holderCount.toLocaleString();

    logger.info(`\n${emoji} FUNDAMENTAL: ${result.name}`);
    logger.info(`   ${ratingEmoji} Rating: ${result.rating} | Score: ${result.healthScore}/100`);
    logger.info(`   📊 Metrics: Holders=${result.metrics.holderConcentration}/40 Liquidity=${result.metrics.liquidityDepth}/30 Age=${result.metrics.tokenAge}/20 Whale=${result.metrics.whaleActivity}/10`);
    logger.info(`   🎯 Details: Holders=${holderCountDisplay} Top10=${result.details.topHoldersPercentage.toFixed(1)}% TVL=$${(result.details.tvlUsd / 1_000_000).toFixed(2)}M Age=${result.details.ageInDays}d Whale=${result.details.whaleTransferCount}`);
    logger.info(`   🎲 Risk: ${result.riskScore.toFixed(1)}/10 | Confidence: ${(result.confidence * 100).toFixed(0)}%`);

    if (!result.approved && result.rejectReason) {
      logger.info(`   ⚠️ Rejected: ${result.rejectReason}`);
    }

    if (result.warnings.length > 0) {
      logger.info(`   ⚠️ Warnings: ${result.warnings.join(', ')}`);
    }
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<FundamentalAnalystConfig>): void {
    this.config = { ...this.config, ...config };
  }
}


