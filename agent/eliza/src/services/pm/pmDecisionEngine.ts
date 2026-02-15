/**
 * PM Decision Engine
 * 
 * Determines whether a trade requires PM approval based on configurable rules.
 * All thresholds are loaded from configuration - no hardcoded values.
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import * as yaml from 'js-yaml';
import { logger } from '../logger.js';
import { approvalQueue } from './approvalQueue.js';
import type { PMConfig, ApprovalRules, QueueTradeParams } from './types.js';

class PMDecisionEngine {
  private config: PMConfig;
  private configPath: string;
  private lastConfigLoad: number = 0;
  private configReloadIntervalMs: number = 60000;

  constructor() {
    this.configPath = path.resolve(process.cwd(), 'config/pm_config.yaml');
    this.config = this.loadConfig();
    approvalQueue.setRules(this.config.rules);
  }

  /**
   * Load configuration from YAML file
   */
  private loadConfig(): PMConfig {
    try {
      // ESM-compatible path resolution
      const __filename = fileURLToPath(import.meta.url);
      const __dirname = path.dirname(__filename);

      const possiblePaths = [
        this.configPath,
        path.resolve(process.cwd(), '../config/pm_config.yaml'),
        path.resolve(__dirname, '../../../../config/pm_config.yaml'),
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
        logger.warn('[PM] Config not found, using defaults');
        return this.getDefaultConfig();
      }

      const raw = yaml.load(configContent) as any;
      const config = this.parseConfig(raw);
      this.lastConfigLoad = Date.now();
      logger.info('[PM] Config loaded', { path: loadedPath });
      return config;
    } catch (error: any) {
      logger.error('[PM] Failed to load config', { error: error.message });
      return this.getDefaultConfig();
    }
  }

  /**
   * Parse raw YAML config to typed config
   */
  private parseConfig(raw: any): PMConfig {
    return {
      enabled: raw.enabled ?? true,
      rules: {
        minPositionForApproval: raw.rules?.min_position_usd ?? 1000,
        minPercentageForApproval: raw.rules?.min_percentage_portfolio ?? 10,
        requireApprovalIfVolatilityAbove: raw.rules?.max_volatility_auto ?? 30,
        requireApprovalIfConfidenceBelow: raw.rules?.min_confidence_auto ?? 70,
        requireApprovalIfRiskScoreAbove: raw.rules?.max_risk_score_auto ?? 70,
        alwaysRequireApprovalFor: raw.rules?.always_approve ?? [],
        neverRequireApprovalFor: raw.rules?.never_approve ?? [],
        approvalTimeoutSeconds: raw.approval_timeout_seconds ?? 300,
        autoApproveWhenPmOffline: raw.auto_approve_when_pm_offline ?? false,
      },
      notifications: {
        enabled: raw.notifications?.enabled ?? true,
        channels: raw.notifications?.channels ?? ['console'],
      },
      logging: {
        logAllSubmissions: raw.logging?.log_all_submissions ?? true,
        logApprovals: raw.logging?.log_approvals ?? true,
        logRejections: raw.logging?.log_rejections ?? true,
        logExpirations: raw.logging?.log_expirations ?? true,
      },
    };
  }

  /**
   * Get default configuration
   */
  private getDefaultConfig(): PMConfig {
    return {
      enabled: true,
      rules: {
        minPositionForApproval: 1000,
        minPercentageForApproval: 10,
        requireApprovalIfVolatilityAbove: 30,
        requireApprovalIfConfidenceBelow: 70,
        requireApprovalIfRiskScoreAbove: 70,
        alwaysRequireApprovalFor: ['perps'],
        neverRequireApprovalFor: ['lending', 'lp'],
        approvalTimeoutSeconds: 300,
        autoApproveWhenPmOffline: false,
      },
      notifications: {
        enabled: true,
        channels: ['console'],
      },
      logging: {
        logAllSubmissions: true,
        logApprovals: true,
        logRejections: true,
        logExpirations: true,
      },
    };
  }

  /**
   * Reload config if stale
   */
  private reloadIfStale(): void {
    if (Date.now() - this.lastConfigLoad > this.configReloadIntervalMs) {
      this.config = this.loadConfig();
      approvalQueue.setRules(this.config.rules);
    }
  }

  /**
   * Check if PM approval is enabled
   */
  isEnabled(): boolean {
    this.reloadIfStale();
    return this.config.enabled;
  }

  /**
   * Get current rules
   */
  getRules(): ApprovalRules {
    this.reloadIfStale();
    return this.config.rules;
  }

  /**
   * Determine if a trade needs PM approval
   */
  needsApproval(params: QueueTradeParams, portfolioValueUsd: number): boolean {
    this.reloadIfStale();

    if (!this.config.enabled) {
      return false;
    }

    const rules = this.config.rules;

    // Check never-approve strategies (auto-approve)
    if (rules.neverRequireApprovalFor.includes(params.strategy)) {
      return false;
    }

    // Check always-approve strategies
    if (rules.alwaysRequireApprovalFor.includes(params.strategy)) {
      return true;
    }

    // Check size-based threshold
    if (params.amountUsd >= rules.minPositionForApproval) {
      logger.debug('[PM] Approval required: size threshold', {
        amountUsd: params.amountUsd,
        threshold: rules.minPositionForApproval,
      });
      return true;
    }

    // Check percentage-based threshold
    if (portfolioValueUsd > 0) {
      const percentage = (params.amountUsd / portfolioValueUsd) * 100;
      if (percentage >= rules.minPercentageForApproval) {
        logger.debug('[PM] Approval required: portfolio % threshold', {
          percentage,
          threshold: rules.minPercentageForApproval,
        });
        return true;
      }
    }

    // Check volatility threshold
    if (params.risk.volatility > rules.requireApprovalIfVolatilityAbove) {
      logger.debug('[PM] Approval required: volatility threshold', {
        volatility: params.risk.volatility,
        threshold: rules.requireApprovalIfVolatilityAbove,
      });
      return true;
    }

    // Check confidence threshold (low confidence = needs approval)
    const confidencePercent = params.confidence * 100;
    if (confidencePercent < rules.requireApprovalIfConfidenceBelow) {
      logger.debug('[PM] Approval required: low confidence', {
        confidence: confidencePercent,
        threshold: rules.requireApprovalIfConfidenceBelow,
      });
      return true;
    }

    // Check risk score threshold
    if (params.risk.riskScore > rules.requireApprovalIfRiskScoreAbove) {
      logger.debug('[PM] Approval required: high risk score', {
        riskScore: params.risk.riskScore,
        threshold: rules.requireApprovalIfRiskScoreAbove,
      });
      return true;
    }

    // All checks passed - no approval needed
    return false;
  }

  /**
   * Check if auto-approve when PM offline is enabled
   */
  shouldAutoApproveWhenOffline(): boolean {
    this.reloadIfStale();
    return this.config.rules.autoApproveWhenPmOffline;
  }

  /**
   * Get approval timeout in seconds
   */
  getApprovalTimeoutSeconds(): number {
    this.reloadIfStale();
    return this.config.rules.approvalTimeoutSeconds;
  }

  /**
   * Get full config (for debugging/admin)
   */
  getConfig(): PMConfig {
    this.reloadIfStale();
    return this.config;
  }
}

// Singleton instance
export const pmDecisionEngine = new PMDecisionEngine();
export { PMDecisionEngine };

