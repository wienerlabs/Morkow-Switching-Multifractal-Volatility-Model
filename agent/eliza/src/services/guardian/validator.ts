/**
 * Guardian Validator Service
 * 
 * Pre-execution transaction security layer that validates ALL transactions
 * before execution to prevent malformed transactions, invalid parameters,
 * security threats, and economic nonsense.
 */

import { Connection, PublicKey } from '@solana/web3.js';
import * as fs from 'fs';
import * as path from 'path';
import * as yaml from 'js-yaml';

import { guardianLogger } from './logger.js';
import type {
  ValidationResult,
  SecurityResult,
  SanityResult,
  GuardianResult,
  GuardianTradeParams,
  GuardianConfig,
  SolanaTransaction,
  TransactionInfo,
  TokenValidation,
} from './types.js';
import { SOLANA_ADDRESS_REGEX, KNOWN_STABLE_MINTS, KNOWN_MAJOR_MINTS } from './types.js';

class GuardianValidator {
  private config: GuardianConfig;
  private connection: Connection | null = null;
  private configPath: string;
  private lastConfigLoad: number = 0;
  private configReloadIntervalMs: number = 60000; // Reload config every minute

  constructor() {
    this.configPath = path.resolve(process.cwd(), 'config/guardian_config.yaml');
    this.config = this.loadConfig();
  }

  /**
   * Load configuration from YAML file
   */
  private loadConfig(): GuardianConfig {
    try {
      // Try multiple paths for config
      const possiblePaths = [
        this.configPath,
        path.resolve(process.cwd(), '../config/guardian_config.yaml'),
        path.resolve(__dirname, '../../../../config/guardian_config.yaml'),
      ];

      let configContent: string | null = null;
      let loadedPath: string | null = null;

      for (const p of possiblePaths) {
        if (fs.existsSync(p)) {
          configContent = fs.readFileSync(p, 'utf-8');
          loadedPath = p;
          break;
        }
      }

      if (!configContent) {
        guardianLogger.warn('Guardian config not found, using defaults');
        return this.getDefaultConfig();
      }

      const parsed = yaml.load(configContent) as Record<string, any>;
      this.lastConfigLoad = Date.now();
      guardianLogger.info('Guardian config loaded', { path: loadedPath });

      // Transform snake_case YAML to camelCase TypeScript
      return this.transformConfig(parsed);
    } catch (error: any) {
      guardianLogger.error('Failed to load Guardian config', { error: error.message });
      return this.getDefaultConfig();
    }
  }

  /**
   * Transform snake_case YAML config to camelCase TypeScript config
   */
  private transformConfig(parsed: Record<string, any>): GuardianConfig {
    const defaults = this.getDefaultConfig();

    // Transform strategy overrides from snake_case to camelCase
    const strategyOverrides: Record<string, Partial<GuardianConfig>> = {};
    const rawOverrides = parsed.strategy_overrides || parsed.strategyOverrides || {};

    for (const [strategy, overrides] of Object.entries(rawOverrides)) {
      if (overrides && typeof overrides === 'object') {
        const rawOv = overrides as Record<string, any>;
        strategyOverrides[strategy] = {
          minSlippagePercent: rawOv.min_slippage_percent ?? rawOv.minSlippagePercent,
          maxSlippagePercent: rawOv.max_slippage_percent ?? rawOv.maxSlippagePercent,
          minPositionUsd: rawOv.min_position_usd ?? rawOv.minPositionUsd,
          maxPositionUsd: rawOv.max_position_usd ?? rawOv.maxPositionUsd,
          maxGasSol: rawOv.max_gas_sol ?? rawOv.maxGasSol,
          maxPriorityFeeLamports: rawOv.max_priority_fee_lamports ?? rawOv.maxPriorityFeeLamports,
          minLiquidityUsd: rawOv.min_liquidity_usd ?? rawOv.minLiquidityUsd,
          maxPriceImpactPercent: rawOv.max_price_impact_percent ?? rawOv.maxPriceImpactPercent,
        };
      }
    }

    return {
      enabled: parsed.enabled ?? defaults.enabled,
      minSlippagePercent: parsed.min_slippage_percent ?? parsed.minSlippagePercent ?? defaults.minSlippagePercent,
      maxSlippagePercent: parsed.max_slippage_percent ?? parsed.maxSlippagePercent ?? defaults.maxSlippagePercent,
      minPositionUsd: parsed.min_position_usd ?? parsed.minPositionUsd ?? defaults.minPositionUsd,
      maxPositionUsd: parsed.max_position_usd ?? parsed.maxPositionUsd ?? defaults.maxPositionUsd,
      maxGasSol: parsed.max_gas_sol ?? parsed.maxGasSol ?? defaults.maxGasSol,
      maxPriorityFeeLamports: parsed.max_priority_fee_lamports ?? parsed.maxPriorityFeeLamports ?? defaults.maxPriorityFeeLamports,
      minLiquidityUsd: parsed.min_liquidity_usd ?? parsed.minLiquidityUsd ?? defaults.minLiquidityUsd,
      maxPriceImpactPercent: parsed.max_price_impact_percent ?? parsed.maxPriceImpactPercent ?? defaults.maxPriceImpactPercent,
      blacklistedAddresses: parsed.blacklisted_addresses ?? parsed.blacklistedAddresses ?? defaults.blacklistedAddresses,
      suspiciousPatterns: parsed.suspicious_patterns ?? parsed.suspiciousPatterns ?? defaults.suspiciousPatterns,
      strategyOverrides,
    };
  }

  /**
   * Get default configuration (fallback)
   */
  private getDefaultConfig(): GuardianConfig {
    return {
      enabled: true,
      minSlippagePercent: 0.1,
      maxSlippagePercent: 5.0,
      minPositionUsd: 10,
      maxPositionUsd: 10000,
      maxGasSol: 0.01,
      maxPriorityFeeLamports: 500000,
      minLiquidityUsd: 50000,
      maxPriceImpactPercent: 3.0,
      blacklistedAddresses: [],
      suspiciousPatterns: [],
      strategyOverrides: {},
    };
  }

  /**
   * Reload config if stale
   */
  private ensureFreshConfig(): void {
    if (Date.now() - this.lastConfigLoad > this.configReloadIntervalMs) {
      this.config = this.loadConfig();
    }
  }

  /**
   * Set connection for on-chain validation
   */
  setConnection(connection: Connection): void {
    this.connection = connection;
  }

  /**
   * Get effective config for a strategy (with overrides)
   */
  private getEffectiveConfig(strategy: string): GuardianConfig {
    const overrides = this.config.strategyOverrides[strategy] || {};
    return { ...this.config, ...overrides };
  }

  /**
   * Check if Guardian is enabled
   */
  isEnabled(): boolean {
    this.ensureFreshConfig();
    return this.config.enabled;
  }

  // ============= MAIN VALIDATION ENTRY POINT =============

  /**
   * Full validation pipeline - validates parameters before execution
   */
  async validate(params: GuardianTradeParams): Promise<GuardianResult> {
    this.ensureFreshConfig();

    const timestamp = new Date();

    // If Guardian is disabled, approve everything
    if (!this.config.enabled) {
      return this.createApprovedResult(timestamp);
    }

    // Run all validation checks
    const validationResult = this.validateParameters(params);
    const securityResult = await this.securityCheck(params);
    const sanityResult = this.sanityCheck(params);

    // Determine if transaction should be blocked
    const shouldBlock = this.shouldBlock(validationResult, securityResult, sanityResult);
    const blockReason = this.getBlockReason(validationResult, securityResult, sanityResult);

    const result: GuardianResult = {
      approved: !shouldBlock,
      validationResult,
      securityResult,
      sanityResult,
      timestamp,
      executionAllowed: !shouldBlock,
      blockReason: shouldBlock ? blockReason : undefined,
    };

    // Log the validation attempt
    guardianLogger.logValidation(params, result);

    return result;
  }

  // ============= PARAMETER VALIDATION =============

  /**
   * Validate trade parameters
   */
  validateParameters(params: GuardianTradeParams): ValidationResult {
    const config = this.getEffectiveConfig(params.strategy);
    const issues: string[] = [];

    // Validate token addresses (Solana format)
    if (!this.isValidSolanaAddress(params.inputMint)) {
      issues.push(`Invalid input mint address: ${params.inputMint}`);
    }
    if (!this.isValidSolanaAddress(params.outputMint)) {
      issues.push(`Invalid output mint address: ${params.outputMint}`);
    }
    if (!this.isValidSolanaAddress(params.walletAddress)) {
      issues.push(`Invalid wallet address: ${params.walletAddress}`);
    }

    // Validate amount > 0
    if (params.amountIn <= 0) {
      issues.push(`Amount must be positive: ${params.amountIn}`);
    }

    // Validate amount within position limits
    if (params.amountInUsd < config.minPositionUsd) {
      issues.push(`Amount $${params.amountInUsd} below minimum $${config.minPositionUsd}`);
    }
    if (params.amountInUsd > config.maxPositionUsd) {
      issues.push(`Amount $${params.amountInUsd} exceeds maximum $${config.maxPositionUsd}`);
    }

    // Validate slippage within limits (convert bps to percent)
    const slippagePercent = params.slippageBps / 100;
    if (slippagePercent < config.minSlippagePercent) {
      issues.push(`Slippage ${slippagePercent}% below minimum ${config.minSlippagePercent}%`);
    }
    if (slippagePercent > config.maxSlippagePercent) {
      issues.push(`Slippage ${slippagePercent}% exceeds maximum ${config.maxSlippagePercent}%`);
    }

    // Validate gas fees if provided
    if (params.estimatedGasSol !== undefined && params.estimatedGasSol > config.maxGasSol) {
      issues.push(`Gas ${params.estimatedGasSol} SOL exceeds maximum ${config.maxGasSol} SOL`);
    }

    // Validate priority fee if provided
    if (params.priorityFeeLamports !== undefined && params.priorityFeeLamports > config.maxPriorityFeeLamports) {
      issues.push(`Priority fee ${params.priorityFeeLamports} exceeds maximum ${config.maxPriorityFeeLamports}`);
    }

    return {
      valid: issues.length === 0,
      reason: issues.length > 0 ? issues.join('; ') : undefined,
      details: { issues, params: { inputMint: params.inputMint, outputMint: params.outputMint, amountUsd: params.amountInUsd } },
    };
  }

  // ============= SECURITY CHECKS =============

  /**
   * Security check - detect threats and suspicious patterns
   */
  async securityCheck(params: GuardianTradeParams): Promise<SecurityResult> {
    const threats: string[] = [];
    let riskScore = 0;

    // Check for blacklisted addresses
    if (this.config.blacklistedAddresses.includes(params.inputMint)) {
      threats.push(`Input mint is blacklisted: ${params.inputMint}`);
      riskScore += 100;
    }
    if (this.config.blacklistedAddresses.includes(params.outputMint)) {
      threats.push(`Output mint is blacklisted: ${params.outputMint}`);
      riskScore += 100;
    }

    // Check if trading with self (suspicious pattern)
    if (params.inputMint === params.outputMint) {
      threats.push('Input and output mints are identical');
      riskScore += 50;
    }

    // Check for unknown tokens (not in known lists)
    const isInputKnown = this.isKnownToken(params.inputMint);
    const isOutputKnown = this.isKnownToken(params.outputMint);

    if (!isInputKnown && !isOutputKnown) {
      riskScore += 30;
    } else if (!isInputKnown || !isOutputKnown) {
      riskScore += 15;
    }

    // Validate liquidity if connection available
    if (this.connection && params.strategy !== 'perps') {
      const tokenValidation = await this.validateTokenLiquidity(params.outputMint);
      if (tokenValidation.isHoneypot) {
        threats.push(`Potential honeypot detected: ${params.outputMint}`);
        riskScore += 100;
      }
      if (!tokenValidation.hasLiquidity) {
        threats.push(`Insufficient liquidity for ${params.outputMint}`);
        riskScore += 40;
      }
      threats.push(...tokenValidation.riskFlags);
    }

    return {
      safe: threats.length === 0 && riskScore < 50,
      threats,
      riskScore: Math.min(100, riskScore),
    };
  }

  // ============= SANITY CHECKS =============

  /**
   * Sanity check - ensure economic sense
   */
  sanityCheck(params: GuardianTradeParams): SanityResult {
    const config = this.getEffectiveConfig(params.strategy);
    const issues: string[] = [];
    const warnings: string[] = [];

    // Check for dust trades
    if (params.amountInUsd < 1) {
      issues.push(`Dust trade detected: $${params.amountInUsd} is too small`);
    }

    // Check price impact
    if (params.priceImpactPct !== undefined) {
      if (params.priceImpactPct > config.maxPriceImpactPercent) {
        issues.push(`Price impact ${params.priceImpactPct}% exceeds maximum ${config.maxPriceImpactPercent}%`);
      } else if (params.priceImpactPct > config.maxPriceImpactPercent * 0.7) {
        warnings.push(`High price impact: ${params.priceImpactPct}%`);
      }
    }

    // Check expected output makes sense
    if (params.expectedAmountOut !== undefined && params.expectedAmountOut <= 0) {
      issues.push(`Invalid expected output: ${params.expectedAmountOut}`);
    }

    // Check for excessive slippage tolerance
    const slippagePercent = params.slippageBps / 100;
    if (slippagePercent > 3.0) {
      warnings.push(`High slippage tolerance: ${slippagePercent}%`);
    }

    return {
      sane: issues.length === 0,
      issues,
      warnings,
    };
  }

  // ============= DECISION METHODS =============

  /**
   * Final decision: should transaction be blocked?
   */
  private shouldBlock(
    validation: ValidationResult,
    security: SecurityResult,
    sanity: SanityResult
  ): boolean {
    // Block if validation fails
    if (!validation.valid) return true;

    // Block if security threats detected
    if (!security.safe) return true;

    // Block if sanity check fails
    if (!sanity.sane) return true;

    return false;
  }

  /**
   * Get reason for blocking
   */
  private getBlockReason(
    validation: ValidationResult,
    security: SecurityResult,
    sanity: SanityResult
  ): string {
    const reasons: string[] = [];

    if (!validation.valid && validation.reason) {
      reasons.push(`Validation: ${validation.reason}`);
    }
    if (!security.safe && security.threats.length > 0) {
      reasons.push(`Security: ${security.threats.join(', ')}`);
    }
    if (!sanity.sane && sanity.issues.length > 0) {
      reasons.push(`Sanity: ${sanity.issues.join(', ')}`);
    }

    return reasons.join(' | ');
  }

  /**
   * Create approved result for when Guardian is disabled
   */
  private createApprovedResult(timestamp: Date): GuardianResult {
    return {
      approved: true,
      validationResult: { valid: true },
      securityResult: { safe: true, threats: [], riskScore: 0 },
      sanityResult: { sane: true, issues: [], warnings: [] },
      timestamp,
      executionAllowed: true,
    };
  }

  // ============= HELPER METHODS =============

  /**
   * Validate Solana address format
   */
  private isValidSolanaAddress(address: string): boolean {
    if (!address || typeof address !== 'string') return false;
    if (!SOLANA_ADDRESS_REGEX.test(address)) return false;

    try {
      new PublicKey(address);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Check if token is in known lists
   */
  private isKnownToken(mint: string): boolean {
    return KNOWN_STABLE_MINTS.includes(mint) || KNOWN_MAJOR_MINTS.includes(mint);
  }

  /**
   * Validate token liquidity (requires connection)
   */
  private async validateTokenLiquidity(mint: string): Promise<TokenValidation> {
    const result: TokenValidation = {
      mint,
      isValid: true,
      isBlacklisted: this.config.blacklistedAddresses.includes(mint),
      hasLiquidity: true,
      isHoneypot: false,
      riskFlags: [],
    };

    if (!this.connection) {
      return result;
    }

    try {
      // Verify mint account exists
      const mintPubkey = new PublicKey(mint);
      const accountInfo = await this.connection.getAccountInfo(mintPubkey);

      if (!accountInfo) {
        result.isValid = false;
        result.hasLiquidity = false;
        result.riskFlags.push('Mint account does not exist');
        return result;
      }

      // Check if it's a token mint (SPL Token program)
      const TOKEN_PROGRAM_ID = 'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA';
      if (accountInfo.owner.toBase58() !== TOKEN_PROGRAM_ID) {
        result.riskFlags.push('Not a valid SPL token');
      }

      // Check actual liquidity and honeypot indicators
      try {
        const largestAccounts = await this.connection.getTokenLargestAccounts(mintPubkey);

        if (!largestAccounts.value || largestAccounts.value.length === 0) {
          result.hasLiquidity = false;
          result.riskFlags.push('No token accounts found - zero liquidity');
        } else {
          const topAccount = largestAccounts.value[0];
          const totalSupplyInfo = await this.connection.getTokenSupply(mintPubkey);
          const totalSupply = Number(totalSupplyInfo.value.amount);

          if (totalSupply > 0) {
            const topHolderPct = (Number(topAccount.amount) / totalSupply) * 100;

            if (topHolderPct > 90) {
              result.isHoneypot = true;
              result.riskFlags.push(`Top holder owns ${topHolderPct.toFixed(1)}% of supply (potential honeypot)`);
            }
          }

          if (largestAccounts.value.length < 5) {
            result.isHoneypot = true;
            result.riskFlags.push(`Very few holders (${largestAccounts.value.length}) - high honeypot risk`);
          }
        }
      } catch (liquidityError: any) {
        result.riskFlags.push(`Liquidity check failed: ${liquidityError.message}`);
      }
    } catch (error: any) {
      result.riskFlags.push(`Failed to validate mint: ${error.message}`);
    }

    return result;
  }
}

// Singleton instance
export const guardian = new GuardianValidator();

// Export class for testing
export { GuardianValidator };
