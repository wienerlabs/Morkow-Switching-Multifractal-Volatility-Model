/**
 * BaseAnalyst
 *
 * Abstract base class for all analyst agents.
 * Analysts are independent, stateless evaluators that receive market data
 * and return opportunities. They can run in parallel with other analysts.
 */

/**
 * Generic opportunity result from analyst evaluation
 */
export interface Opportunity {
  /** Opportunity type identifier */
  type: string;
  /** Human-readable name for the opportunity */
  name: string;
  /** Expected return percentage */
  expectedReturn: number;
  /** Risk score (1-10, lower is safer) */
  riskScore: number;
  /** Model confidence (0-1) */
  confidence: number;
  /** Risk-adjusted return */
  riskAdjustedReturn: number;
  /** Whether the opportunity passed all checks */
  approved: boolean;
  /** Reason for rejection if not approved */
  rejectReason?: string;
  /** Warning messages */
  warnings: string[];
  /** Raw underlying data */
  raw: unknown;
}

/**
 * Configuration for analyst behavior
 */
export interface AnalystConfig {
  /** Minimum confidence threshold for approval (0-1) */
  minConfidence: number;
  /** Current portfolio value in USD */
  portfolioValueUsd: number;
  /** Current 24h volatility */
  volatility24h: number;
  /** Enable verbose logging */
  verbose?: boolean;
}

/**
 * Default analyst configuration
 */
export const DEFAULT_ANALYST_CONFIG: AnalystConfig = {
  minConfidence: 0.50,
  portfolioValueUsd: 10000,
  volatility24h: 0.05,
  verbose: true,
};

/**
 * Abstract base class for analyst agents
 *
 * Analysts are designed to:
 * - Be stateless and independent
 * - Run in parallel with other analysts
 * - Receive market data as input
 * - Return typed opportunities as output
 * - Have no dependency on orchestrator state
 */
export abstract class BaseAnalyst<
  TInput = unknown,
  TOutput extends Opportunity = Opportunity,
> {
  protected config: AnalystConfig;

  constructor(config: Partial<AnalystConfig> = {}) {
    this.config = { ...DEFAULT_ANALYST_CONFIG, ...config };
  }

  /**
   * Get the analyst name for identification
   */
  abstract getName(): string;

  /**
   * Analyze input data and return opportunities
   *
   * @param data - Market data or opportunities to analyze
   * @returns Array of evaluated opportunities
   */
  abstract analyze(data: TInput): Promise<TOutput[]>;

  /**
   * Update analyst configuration
   */
  updateConfig(config: Partial<AnalystConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration
   */
  getConfig(): AnalystConfig {
    return { ...this.config };
  }
}

