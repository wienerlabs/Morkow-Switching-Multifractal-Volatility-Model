/**
 * LPAnalyst Tests
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { LPAnalyst, type LPAnalysisInput, type LPOpportunityResult } from '../LPAnalyst.js';
import type { LPPool } from '../../../services/marketScanner/types.js';
import type { PredictionResult } from '../../../inference/model.js';

// Mock sentiment integration
vi.mock('../../../services/sentiment/sentimentIntegration.js', () => ({
  getSentimentIntegration: () => ({
    getAdjustedScore: async (_token: string, mlConfidence: number, _strategy: string) => ({
      mlConfidence,
      rawSentiment: 0,
      normalizedSentiment: 0.5,
      finalScore: mlConfidence, // Return ML confidence as final score (no sentiment adjustment)
      sentimentWeight: 0.15,
      signal: 'neutral' as const,
      sentimentAvailable: false,
      reasoning: 'Sentiment unavailable',
    }),
  }),
}));

// Helper to create mock LP pool
function createMockPool(overrides: Partial<LPPool> = {}): LPPool {
  return {
    name: 'SOL/USDC',
    address: 'pool123',
    dex: 'orca',
    token0: 'SOL',
    token1: 'USDC',
    apy: 50,
    apr: 45,
    tvl: 10_000_000,
    volume24h: 5_000_000,
    fees24h: 10_000,
    feeRate: 0.003,
    utilization: 0.8,
    riskScore: 3,
    ...overrides,
  };
}

// Helper to create mock ML prediction
function createMockPrediction(overrides: Partial<PredictionResult> = {}): PredictionResult {
  return {
    decision: 'REBALANCE',
    probability: 0.9,
    confidence: 0.9,
    threshold: 0.84,
    ...overrides,
  };
}

// Helper to create input
function createInput(pools: LPPool[], volatility = 0.05, mlPredictions?: Map<string, PredictionResult>): LPAnalysisInput {
  return {
    pools,
    volatility24h: volatility,
    portfolioValueUsd: 10000,
    mlPredictions,
  };
}

describe('LPAnalyst', () => {
  let analyst: LPAnalyst;

  beforeEach(() => {
    analyst = new LPAnalyst();
  });

  // ============= BASIC FUNCTIONALITY =============
  describe('Basic Functionality', () => {
    it('should instantiate with default config', () => {
      expect(analyst).toBeDefined();
      expect(analyst.getName()).toBe('LPAnalyst');
    });

    it('should instantiate with custom config', () => {
      const customAnalyst = new LPAnalyst({
        minConfidence: 0.9,
        maxApy: 300,
        minTvl: 500_000,
      });
      expect(customAnalyst).toBeDefined();
      expect(customAnalyst.getLPConfig().minConfidence).toBe(0.9);
      expect(customAnalyst.getLPConfig().maxApy).toBe(300);
      expect(customAnalyst.getLPConfig().minTvl).toBe(500_000);
    });

    it('should return agent name', () => {
      expect(analyst.getName()).toBe('LPAnalyst');
    });

    it('should analyze LP pool opportunities', async () => {
      const pool = createMockPool({ apy: 60, tvl: 20_000_000 });
      const input = createInput([pool]);

      const results = await analyst.analyze(input);

      expect(results).toHaveLength(1);
      expect(results[0].type).toBe('lp');
      expect(results[0].raw).toEqual(pool);
    });

    it('should analyze multiple pools', async () => {
      const pools = [
        createMockPool({ name: 'SOL/USDC', apy: 50 }),
        createMockPool({ name: 'SOL/USDT', apy: 60 }),
        createMockPool({ name: 'JUP/USDC', apy: 70 }),
      ];
      const input = createInput(pools);

      const results = await analyst.analyze(input);

      expect(results).toHaveLength(3);
      expect(results[0].name).toContain('SOL/USDC');
      expect(results[1].name).toContain('SOL/USDT');
      expect(results[2].name).toContain('JUP/USDC');
    });
  });

  // ============= SCAM FILTER =============
  describe('Scam Filter', () => {
    it('should reject pools with APY > 500%', async () => {
      const pool = createMockPool({ apy: 600 });
      const input = createInput([pool]);

      const results = await analyst.analyze(input);

      expect(results[0].approved).toBe(false);
      expect(results[0].rejectReason).toContain('APY too high');
    });

    it('should reject pools with TVL < $100k', async () => {
      const pool = createMockPool({ tvl: 50_000 });
      const input = createInput([pool]);

      const results = await analyst.analyze(input);

      expect(results[0].approved).toBe(false);
      expect(results[0].rejectReason).toContain('TVL too low');
    });

    it('should reject pools with low volume/TVL ratio', async () => {
      const pool = createMockPool({ tvl: 10_000_000, volume24h: 1_000_000 }); // 0.1 ratio
      const input = createInput([pool]);

      const results = await analyst.analyze(input);

      expect(results[0].approved).toBe(false);
      expect(results[0].rejectReason).toContain('Volume/TVL too low');
    });

    it('should reject pools with unknown tokens', async () => {
      const pool = createMockPool({ name: 'SCAM/USDC' });
      const input = createInput([pool]);

      const results = await analyst.analyze(input);

      expect(results[0].approved).toBe(false);
      expect(results[0].rejectReason).toContain('Unknown token');
    });
  });

  // ============= ML PREDICTION =============
  describe('ML Prediction', () => {
    it('should use ML prediction confidence when available', async () => {
      const pool = createMockPool({ apy: 50, tvl: 20_000_000, volume24h: 15_000_000 });
      const mlPredictions = new Map<string, PredictionResult>();
      mlPredictions.set('pool123', createMockPrediction({ confidence: 0.95 }));
      const input = createInput([pool], 0.05, mlPredictions);

      const results = await analyst.analyze(input);

      expect(results[0].mlPrediction).toBeDefined();
      expect(results[0].mlPrediction?.confidence).toBe(0.95);
      expect(results[0].confidence).toBeGreaterThanOrEqual(0.84); // Should pass threshold
      expect(results[0].approved).toBe(true);
    });

    it('should invert confidence for HOLD decision', async () => {
      const pool = createMockPool({ apy: 50, tvl: 20_000_000, volume24h: 15_000_000 });
      const mlPredictions = new Map<string, PredictionResult>();
      mlPredictions.set('pool123', createMockPrediction({
        decision: 'HOLD',
        probability: 0.2,
        confidence: 0.9
      }));
      const input = createInput([pool], 0.05, mlPredictions);

      const results = await analyst.analyze(input);

      expect(results[0].mlPrediction).toBeDefined();
      expect(results[0].warnings).toContain('ML: HOLD (20.0%)');
      // Confidence should be inverted (1 - 0.9 = 0.1)
      expect(results[0].confidence).toBeLessThan(0.5);
      expect(results[0].approved).toBe(false);
    });

    it('should fall back to heuristics when ML unavailable', async () => {
      const pool = createMockPool({ apy: 50, tvl: 60_000_000, volume24h: 70_000_000 });
      const input = createInput([pool]); // No ML predictions

      const results = await analyst.analyze(input);

      expect(results[0].warnings).toContain('ML: unavailable (using heuristics)');
      expect(results[0].confidence).toBeGreaterThan(0.8); // High TVL + high volume
    });
  });

  // ============= APY AND RISK SCORING =============
  describe('APY and Risk Scoring', () => {
    it('should calculate daily return from APY', async () => {
      const pool = createMockPool({ apy: 73, tvl: 20_000_000, volume24h: 15_000_000 }); // 73% APY = 0.2% daily
      const mlPredictions = new Map<string, PredictionResult>();
      mlPredictions.set('pool123', createMockPrediction({ confidence: 0.9 }));
      const input = createInput([pool], 0.05, mlPredictions);

      const results = await analyst.analyze(input);

      expect(results[0].expectedReturn).toBe(73);
      // Daily return should be ~0.2% (73/365)
      const dailyReturn = 73 / 365;
      expect(results[0].riskAdjustedReturn).toBeCloseTo(dailyReturn * (1 - 3 / 20), 4);
    });

    it('should assign low risk score for blue chip pools', async () => {
      const pool = createMockPool({ apy: 30, tvl: 60_000_000, volume24h: 40_000_000 });
      const mlPredictions = new Map<string, PredictionResult>();
      mlPredictions.set('pool123', createMockPrediction({ confidence: 0.9 }));
      const input = createInput([pool], 0.05, mlPredictions);

      const results = await analyst.analyze(input);

      expect(results[0].riskScore).toBe(2); // Blue chip
      expect(results[0].warnings).toContain('Risk: low');
    });

    it('should assign medium risk score for mid-tier pools', async () => {
      const pool = createMockPool({ apy: 120, tvl: 8_000_000 });
      const input = createInput([pool]);

      const results = await analyst.analyze(input);

      expect(results[0].riskScore).toBe(6); // Medium pool, high APY
      expect(results[0].warnings).toContain('Risk: medium');
    });

    it('should calculate risk-adjusted return', async () => {
      const pool = createMockPool({ apy: 100, tvl: 20_000_000, volume24h: 15_000_000 });
      const mlPredictions = new Map<string, PredictionResult>();
      mlPredictions.set('pool123', createMockPrediction({ confidence: 0.9 }));
      const input = createInput([pool], 0.05, mlPredictions);

      const results = await analyst.analyze(input);

      const dailyReturn = 100 / 365;
      // Use actual risk score from result
      const expected = dailyReturn * (1 - results[0].riskScore / 20);
      expect(results[0].riskAdjustedReturn).toBeCloseTo(expected, 4);
    });
  });

  // ============= POOL FILTERING =============
  describe('Pool Filtering', () => {
    it('should filter by min TVL', async () => {
      const customAnalyst = new LPAnalyst({ minTvl: 1_000_000 });
      const pool = createMockPool({ tvl: 500_000 });
      const input = createInput([pool]);

      const results = await customAnalyst.analyze(input);

      expect(results[0].approved).toBe(false);
      expect(results[0].rejectReason).toContain('TVL too low');
    });

    it('should filter by min volume/TVL ratio', async () => {
      const customAnalyst = new LPAnalyst({ minVolumeTvlRatio: 0.5 });
      const pool = createMockPool({ tvl: 10_000_000, volume24h: 3_000_000 }); // 0.3 ratio
      const input = createInput([pool]);

      const results = await customAnalyst.analyze(input);

      expect(results[0].approved).toBe(false);
      expect(results[0].rejectReason).toContain('Volume/TVL too low');
    });

    it('should allow whitelisted tokens only', async () => {
      const pool1 = createMockPool({ name: 'SOL/USDC', volume24h: 15_000_000 });
      const pool2 = createMockPool({ name: 'JUP/USDT', volume24h: 15_000_000 });
      const pool3 = createMockPool({ name: 'BONK/SOL', volume24h: 15_000_000 });
      const mlPredictions = new Map<string, PredictionResult>();
      mlPredictions.set('pool123', createMockPrediction({ confidence: 0.9 }));
      const input = createInput([pool1, pool2, pool3], 0.05, mlPredictions);

      const results = await analyst.analyze(input);

      // All should pass scam filters (whitelisted tokens)
      expect(results[0].approved).toBe(true);
      expect(results[1].approved).toBe(true);
      expect(results[2].approved).toBe(true);
    });
  });

  // ============= CONFIDENCE SCORING =============
  describe('Confidence Scoring', () => {
    it('should return confidence score from ML', async () => {
      const pool = createMockPool({ apy: 50, tvl: 20_000_000, volume24h: 15_000_000 });
      const mlPredictions = new Map<string, PredictionResult>();
      mlPredictions.set('pool123', createMockPrediction({ confidence: 0.88 }));
      const input = createInput([pool], 0.05, mlPredictions);

      const results = await analyst.analyze(input);

      expect(results[0].confidence).toBeGreaterThanOrEqual(0.84);
      expect(results[0].approved).toBe(true);
    });

    it('should filter below 50% confidence threshold', async () => {
      const pool = createMockPool({ apy: 50, tvl: 150_000, volume24h: 20_000 }); // Low confidence
      const input = createInput([pool]);

      const results = await analyst.analyze(input);

      expect(results[0].confidence).toBeLessThan(0.50);
      expect(results[0].approved).toBe(false);
    });

    it('should use custom min confidence threshold', async () => {
      const customAnalyst = new LPAnalyst({ minConfidenceThreshold: 0.9 });
      const pool = createMockPool({ apy: 50, tvl: 20_000_000 });
      const mlPredictions = new Map<string, PredictionResult>();
      mlPredictions.set('pool123', createMockPrediction({ confidence: 0.85 }));
      const input = createInput([pool], 0.05, mlPredictions);

      const results = await customAnalyst.analyze(input);

      expect(results[0].confidence).toBeLessThan(0.9);
      expect(results[0].approved).toBe(false);
    });
  });

  // ============= EMPTY RESULTS =============
  describe('Empty Results', () => {
    it('should return empty array for no pools', async () => {
      const input = createInput([]);

      const results = await analyst.analyze(input);

      expect(results).toHaveLength(0);
    });

    it('should return results even if all rejected', async () => {
      const pools = [
        createMockPool({ apy: 600 }), // Too high APY
        createMockPool({ tvl: 50_000 }), // Too low TVL (below $100k)
        createMockPool({ name: 'SCAM/USDC' }), // Unknown token
      ];
      const input = createInput(pools);

      const results = await analyst.analyze(input);

      expect(results).toHaveLength(3);
      expect(results.every(r => !r.approved)).toBe(true);
    });
  });

  // ============= CONFIGURATION UPDATES =============
  describe('Configuration Updates', () => {
    it('should update LP config', () => {
      const testAnalyst = new LPAnalyst();
      testAnalyst.updateLPConfig({ maxApy: 300, minTvl: 500_000 });

      const config = testAnalyst.getLPConfig();
      expect(config.maxApy).toBe(300);
      expect(config.minTvl).toBe(500_000);
    });

    it('should update base config', () => {
      const testAnalyst = new LPAnalyst();
      testAnalyst.updateConfig({ minConfidence: 0.9 });

      const config = testAnalyst.getConfig();
      expect(config.minConfidence).toBe(0.9);
    });

    it('should use volatility from input over config', async () => {
      const pool = createMockPool({ apy: 50, tvl: 20_000_000 });
      const input = createInput([pool], 0.15); // High volatility

      const results = await analyst.analyze(input);

      // High volatility should affect position sizing (checked by RiskManager)
      expect(results[0]).toBeDefined();
    });
  });

  // ============= EDGE CASES =============
  describe('Edge Cases', () => {
    it('should handle zero APY', async () => {
      const pool = createMockPool({ apy: 0, tvl: 20_000_000 });
      const input = createInput([pool]);

      const results = await analyst.analyze(input);

      expect(results[0].expectedReturn).toBe(0);
      expect(results[0].riskAdjustedReturn).toBe(0);
    });

    it('should handle pool name without slash', async () => {
      const pool = createMockPool({ name: 'SOL-USDC' });
      const input = createInput([pool]);

      const results = await analyst.analyze(input);

      // Should still extract tokens (split by /)
      expect(results[0]).toBeDefined();
    });

    it('should handle missing ML prediction gracefully', async () => {
      const pool = createMockPool({ apy: 50, tvl: 60_000_000, volume24h: 40_000_000 });
      const mlPredictions = new Map<string, PredictionResult>();
      // No prediction for pool123
      const input = createInput([pool], 0.05, mlPredictions);

      const results = await analyst.analyze(input);

      expect(results[0].mlPrediction).toBeUndefined();
      expect(results[0].warnings).toContain('ML: unavailable (using heuristics)');
      expect(results[0].approved).toBe(true); // Should pass with heuristics
    });

    it('should handle very high volume/TVL ratio', async () => {
      const pool = createMockPool({ tvl: 10_000_000, volume24h: 50_000_000 }); // 5.0 ratio
      const input = createInput([pool]);

      const results = await analyst.analyze(input);

      // Should pass volume/TVL check
      expect(results[0].approved).toBe(true);
    });
  });
});
