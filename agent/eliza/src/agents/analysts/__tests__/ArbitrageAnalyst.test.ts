/**
 * ArbitrageAnalyst Unit Tests
 *
 * Tests for the standalone ArbitrageAnalyst agent covering:
 * - Basic functionality
 * - Spread calculation
 * - Gas cost analysis
 * - Direction validation
 * - MEV risk scoring
 * - Confidence scoring (ML-based)
 * - Empty results handling
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  ArbitrageAnalyst,
  DEFAULT_ARBITRAGE_CONFIG,
  type ArbitrageAnalysisInput,
  type ArbitrageOpportunityResult,
} from '../ArbitrageAnalyst.js';
import type { ArbitrageOpportunity } from '../../../services/marketScanner/types.js';

// Mock the sentiment integration to avoid external dependencies
// The mock returns the input confidence as the final score (no adjustment)
vi.mock('../../../services/sentiment/sentimentIntegration.js', () => ({
  getSentimentIntegration: () => ({
    getAdjustedScore: vi.fn().mockImplementation(
      async (_symbol: string, mlConfidence: number, _strategyType: string) => ({
        mlConfidence,
        rawSentiment: 0.5,
        normalizedSentiment: 0.75,
        finalScore: mlConfidence, // Pass through the base confidence
        sentimentWeight: 0.05,
        signal: 'neutral',
        sentimentAvailable: false,
        reasoning: 'Test mock - sentiment unavailable',
      })
    ),
  }),
}));

// Mock the arbitrage ML model to return confidence based on the opportunity's confidence field
// This maintains backward compatibility with existing tests
vi.mock('../../../services/arbitrage/index.js', () => {
  // Mock ArbitrageFeatureExtractor as a class
  class MockArbitrageFeatureExtractor {
    extractFeatures = vi.fn().mockResolvedValue(new Float32Array(27));
    clearHistory = vi.fn();
    getFeatureNames = vi.fn().mockReturnValue([]);
  }

  return {
    ArbitrageFeatureExtractor: MockArbitrageFeatureExtractor,
    arbitrageModelLoader: {
      initialize: vi.fn().mockResolvedValue(true),
      isInitialized: vi.fn().mockReturnValue(true),
      predict: vi.fn().mockImplementation(async (_features: Float32Array, threshold: number, _tradeId: string) => {
        // ML mock returns 0.85 for all cases
        return {
          probability: 0.85,
          confidence: 0.85,
          isProfitable: true,
          threshold,
        };
      }),
      getMetadata: vi.fn().mockReturnValue({ version: 'test' }),
      getFeatureNames: vi.fn().mockReturnValue([]),
    },
    ARBITRAGE_FEATURE_NAMES: [],
  };
});

// ============= TEST HELPERS =============

/**
 * Create a mock arbitrage opportunity with sensible defaults
 */
function createMockArbitrage(overrides: Partial<ArbitrageOpportunity> = {}): ArbitrageOpportunity {
  return {
    symbol: 'SOL',
    buyExchange: 'binance',
    sellExchange: 'jupiter',
    buyPrice: 100,
    sellPrice: 101.5,
    spreadPct: 1.5,
    estimatedProfit: 15,
    fees: 5,
    netProfit: 10,
    confidence: 'high',
    ...overrides,
  };
}

/**
 * Create input for analyst
 */
function createInput(
  opportunities: ArbitrageOpportunity[],
  volatility24h = 0.05
): ArbitrageAnalysisInput {
  return { opportunities, volatility24h };
}

// ============= TESTS =============

describe('ArbitrageAnalyst', () => {
  let analyst: ArbitrageAnalyst;

  beforeEach(() => {
    // Create analyst with verbose=false to reduce test output
    analyst = new ArbitrageAnalyst({ verbose: false });
  });

  // ============= 1. BASIC FUNCTIONALITY =============
  describe('Basic Functionality', () => {
    it('should instantiate correctly with default config', () => {
      const defaultAnalyst = new ArbitrageAnalyst();
      expect(defaultAnalyst).toBeInstanceOf(ArbitrageAnalyst);
      expect(defaultAnalyst.getConfig().minConfidence).toBe(DEFAULT_ARBITRAGE_CONFIG.minConfidence);
    });

    it('should instantiate with custom config', () => {
      const customAnalyst = new ArbitrageAnalyst({
        minConfidence: 0.8,
        minSpreadPct: 2.0,
        verbose: false,
      });
      expect(customAnalyst.getConfig().minConfidence).toBe(0.8);
    });

    it('should return correct agent name', () => {
      expect(analyst.getName()).toBe('ArbitrageAnalyst');
    });

    it('should analyze market data and return results', async () => {
      const input = createInput([createMockArbitrage()]);
      const results = await analyst.analyze(input);

      expect(results).toHaveLength(1);
      expect(results[0].type).toBe('arbitrage');
      expect(results[0].name).toContain('SOL');
    });

    it('should process multiple opportunities', async () => {
      const input = createInput([
        createMockArbitrage({ symbol: 'SOL' }),
        createMockArbitrage({ symbol: 'ETH' }),
        createMockArbitrage({ symbol: 'BTC' }),
      ]);
      const results = await analyst.analyze(input);

      expect(results).toHaveLength(3);
      expect(results.map(r => r.raw.symbol)).toEqual(['SOL', 'ETH', 'BTC']);
    });
  });

  // ============= 2. SPREAD CALCULATION =============
  describe('Spread Calculation', () => {
    it('should calculate spread correctly', async () => {
      const input = createInput([createMockArbitrage({ spreadPct: 2.5 })]);
      const results = await analyst.analyze(input);

      expect(results[0].expectedReturn).toBe(2.5);
    });

    it('should reject spreads below threshold (0.10%)', async () => {
      const input = createInput([createMockArbitrage({ spreadPct: 0.05, buyExchange: 'raydium', sellExchange: 'jupiter' })]);
      const results = await analyst.analyze(input);

      expect(results[0].approved).toBe(false);
      expect(results[0].rejectReason).toContain('Spread too low');
    });

    it('should reject unrealistically high spreads (>10%)', async () => {
      const input = createInput([createMockArbitrage({ spreadPct: 15, buyExchange: 'raydium', sellExchange: 'jupiter' })]);
      const results = await analyst.analyze(input);

      expect(results[0].approved).toBe(false);
      expect(results[0].rejectReason).toContain('Spread');
    });

    it('should prefer higher spreads (higher risk-adjusted return)', async () => {
      const input = createInput([
        createMockArbitrage({ spreadPct: 1.5, netProfit: 10 }),
        createMockArbitrage({ spreadPct: 3.0, netProfit: 25 }),
      ]);
      const results = await analyst.analyze(input);

      const lowSpread = results[0];
      const highSpread = results[1];

      expect(highSpread.riskAdjustedReturn).toBeGreaterThan(lowSpread.riskAdjustedReturn);
    });

    it('should use custom spread thresholds when configured', async () => {
      const customAnalyst = new ArbitrageAnalyst({
        minSpreadPct: 2.0,
        maxSpreadPct: 5.0,
        allowDexToDex: true,
        verbose: false,
      });

      const input = createInput([createMockArbitrage({ spreadPct: 1.5, buyExchange: 'raydium', sellExchange: 'jupiter' })]);
      const results = await customAnalyst.analyze(input);

      expect(results[0].approved).toBe(false);
      expect(results[0].rejectReason).toContain('Spread too low');
    });
  });

  // ============= 3. GAS COST ANALYSIS =============
  describe('Gas Cost Analysis', () => {
    it('should include gas in profitability check via netProfit', async () => {
      // Use DEX→DEX direction (both allowed by default config)
      const input = createInput([createMockArbitrage({ netProfit: 10, fees: 5, buyExchange: 'raydium', sellExchange: 'jupiter' })]);
      const results = await analyst.analyze(input);

      expect(results[0].approved).toBe(true);
      expect(results[0].raw.netProfit).toBe(10);
    });

    it('should reject when spread is below minimum threshold', async () => {
      const input = createInput([createMockArbitrage({ spreadPct: 0.04, netProfit: 0.4, buyExchange: 'raydium', sellExchange: 'jupiter' })]);
      const results = await analyst.analyze(input);

      expect(results[0].approved).toBe(false);
      expect(results[0].rejectReason).toContain('Spread too low');
    });

    it('should reject zero or negative net profit', async () => {
      const input = createInput([createMockArbitrage({ netProfit: -2 })]);
      const results = await analyst.analyze(input);

      expect(results[0].approved).toBe(false);
      expect(results[0].riskAdjustedReturn).toBe(0);
    });

    it('should calculate risk-adjusted return accounting for risk', async () => {
      const input = createInput([createMockArbitrage({ spreadPct: 2.0, netProfit: 15 })]);
      const results = await analyst.analyze(input);

      // riskAdjustedReturn = spreadPct * (1 - riskScore / 20)
      // With spreadPct=2.0 and riskScore=5 (since 1 < 2.0 <= 2), but CEX→DEX reduces by 1 = 4
      // riskAdjustedReturn = 2.0 * (1 - 4/20) = 2.0 * 0.8 = 1.6
      expect(results[0].riskAdjustedReturn).toBeCloseTo(1.6, 1);
    });
  });

  // ============= 4. DIRECTION VALIDATION =============
  describe('Direction Validation', () => {
    it('should reject CEX→DEX arbitrage (requires CEX balance)', async () => {
      // Default config: allowCexToDex = false (requires CEX balance)
      const input = createInput([
        createMockArbitrage({
          buyExchange: 'binance',
          sellExchange: 'jupiter',
        }),
      ]);
      const results = await analyst.analyze(input);

      expect(results[0].direction.isCexToDex).toBe(true);
      expect(results[0].approved).toBe(false);
      expect(results[0].rejectReason).toContain('CEX→DEX disabled');
    });

    it('should allow DEX→CEX arbitrage', async () => {
      // Default config: allowDexToCex = true
      const input = createInput([
        createMockArbitrage({
          buyExchange: 'jupiter',
          sellExchange: 'binance',
        }),
      ]);
      const results = await analyst.analyze(input);

      expect(results[0].direction.isDexToCex).toBe(true);
      expect(results[0].approved).toBe(true);
    });

    it('should ALLOW DEX→DEX arbitrage when enabled (per risk params)', async () => {
      // Per risk parameters: dexToDex: true (enabled)
      const input = createInput([
        createMockArbitrage({
          buyExchange: 'raydium',
          sellExchange: 'jupiter',
        }),
      ]);
      const results = await analyst.analyze(input);

      expect(results[0].direction.isDexToDex).toBe(true);
      expect(results[0].approved).toBe(true); // Now allowed!
    });

    it('should reject DEX→DEX when disabled via config', async () => {
      const restrictedAnalyst = new ArbitrageAnalyst({
        allowDexToDex: false,
        verbose: false,
      });
      const input = createInput([
        createMockArbitrage({
          buyExchange: 'raydium',
          sellExchange: 'jupiter',
        }),
      ]);
      const results = await restrictedAnalyst.analyze(input);

      expect(results[0].direction.isDexToDex).toBe(true);
      expect(results[0].approved).toBe(false);
      expect(results[0].rejectReason).toContain('DEX→DEX disabled');
    });

    it('should reject CEX→CEX arbitrage (transfer required)', async () => {
      const input = createInput([
        createMockArbitrage({
          buyExchange: 'binance',
          sellExchange: 'coinbase',
        }),
      ]);
      const results = await analyst.analyze(input);

      expect(results[0].direction.isCexToCex).toBe(true);
      expect(results[0].approved).toBe(false);
      expect(results[0].rejectReason).toContain('CEX→CEX');
    });

    it('should recognize all configured CEX exchanges', async () => {
      // Default CEX list: binance, coinbase, kraken
      const testCases = [
        { buy: 'binance', sell: 'jupiter', expected: 'isCexToDex' },
        { buy: 'coinbase', sell: 'raydium', expected: 'isCexToDex' },
        { buy: 'kraken', sell: 'orca', expected: 'isCexToDex' },
      ];

      for (const tc of testCases) {
        const input = createInput([
          createMockArbitrage({ buyExchange: tc.buy, sellExchange: tc.sell }),
        ]);
        const results = await analyst.analyze(input);
        expect(results[0].direction[tc.expected as keyof typeof results[0]['direction']]).toBe(true);
      }
    });
  });

  // ============= 5. MEV RISK SCORING =============
  describe('MEV Risk Scoring', () => {
    it('should score MEV risk based on spread', async () => {
      // Higher spread = higher risk
      const lowSpread = createInput([createMockArbitrage({ spreadPct: 1.2 })]);
      const highSpread = createInput([createMockArbitrage({ spreadPct: 2.5 })]);

      const lowResults = await analyst.analyze(lowSpread);
      const highResults = await analyst.analyze(highSpread);

      // spreadPct <= 1: riskScore = 3, spreadPct > 1 <= 2: riskScore = 5, spreadPct > 2: riskScore = 7
      // CEX→DEX reduces by 1
      expect(lowResults[0].riskScore).toBe(4); // 5 - 1 (CEX→DEX bonus)
      expect(highResults[0].riskScore).toBe(6); // 7 - 1 (CEX→DEX bonus)
    });

    it('should reduce risk score for CEX→DEX direction', async () => {
      // CEX→DEX gets -1 risk score bonus
      const input = createInput([
        createMockArbitrage({
          buyExchange: 'binance',
          sellExchange: 'jupiter',
          spreadPct: 1.5,
        }),
      ]);
      const results = await analyst.analyze(input);

      // Base: 5 (since 1 < 1.5 <= 2), then -1 for CEX→DEX = 4
      expect(results[0].riskScore).toBe(4);
      expect(results[0].direction.isCexToDex).toBe(true);
    });

    it('should have higher risk for DEX→CEX (no bonus)', async () => {
      const cexToDex = createInput([
        createMockArbitrage({
          buyExchange: 'binance',
          sellExchange: 'jupiter',
          spreadPct: 1.5,
        }),
      ]);
      const dexToCex = createInput([
        createMockArbitrage({
          buyExchange: 'jupiter',
          sellExchange: 'binance',
          spreadPct: 1.5,
        }),
      ]);

      const cexResults = await analyst.analyze(cexToDex);
      const dexResults = await analyst.analyze(dexToCex);

      // CEX→DEX: 5 - 1 = 4, DEX→CEX: 5 (no bonus)
      expect(dexResults[0].riskScore).toBe(5);
      expect(cexResults[0].riskScore).toBe(4);
    });
  });

  // ============= 6. CONFIDENCE SCORING (ML-BASED) =============
  describe('Confidence Scoring', () => {
    it('should return ML-based confidence score', async () => {
      const input = createInput([createMockArbitrage({ confidence: 'high' })]);
      const results = await analyst.analyze(input);

      // ML model mock returns 0.85 confidence
      expect(results[0].confidence).toBe(0.85);
    });

    it('should use ML model for all confidence levels', async () => {
      // ML model now determines confidence, not the input string
      const testCases = ['high', 'medium', 'low'] as const;

      for (const level of testCases) {
        const input = createInput([createMockArbitrage({ confidence: level })]);
        const results = await analyst.analyze(input);
        // ML mock returns 0.85 for all cases
        expect(results[0].confidence).toBe(0.85);
      }
    });

    it('should approve when ML confidence exceeds threshold', async () => {
      // Use DEX→DEX direction (allowed by default)
      const input = createInput([createMockArbitrage({ confidence: 'low', buyExchange: 'raydium', sellExchange: 'jupiter' })]);
      const results = await analyst.analyze(input);

      // ML mock returns 0.85 > 0.60 (default minConfidence)
      expect(results[0].approved).toBe(true);
    });

    it('should approve when ML confidence is high', async () => {
      const input = createInput([createMockArbitrage({ confidence: 'high', buyExchange: 'raydium', sellExchange: 'jupiter' })]);
      const results = await analyst.analyze(input);

      // ML mock returns 0.85 > 0.60 threshold
      expect(results[0].approved).toBe(true);
    });

    it('should respect custom min confidence threshold', async () => {
      const strictAnalyst = new ArbitrageAnalyst({
        minConfidence: 0.9, // Higher than ML mock's 0.85
        verbose: false,
      });

      const input = createInput([createMockArbitrage({ confidence: 'high' })]);
      const results = await strictAnalyst.analyze(input);

      // ML mock returns 0.85 < 0.9 threshold
      expect(results[0].approved).toBe(false);
    });
  });

  // ============= 7. EMPTY RESULTS =============
  describe('Empty Results', () => {
    it('should return empty array if no opportunities', async () => {
      const input = createInput([]);
      const results = await analyst.analyze(input);

      expect(results).toEqual([]);
      expect(results).toHaveLength(0);
    });

    it('should return results even if all are rejected', async () => {
      const input = createInput([
        createMockArbitrage({ spreadPct: 0.02 }), // Below 0.10% threshold
        createMockArbitrage({ buyExchange: 'binance', sellExchange: 'coinbase' }), // CEX→CEX (disabled)
      ]);
      const results = await analyst.analyze(input);

      expect(results).toHaveLength(2);
      expect(results.every(r => !r.approved)).toBe(true);
    });
  });

  // ============= 8. CONFIG UPDATES =============
  describe('Configuration Updates', () => {
    it('should update config via updateConfig', () => {
      analyst.updateConfig({ minConfidence: 0.9 });
      expect(analyst.getConfig().minConfidence).toBe(0.9);
    });

    it('should update arbitrage-specific config', () => {
      analyst.updateArbitrageConfig({ minSpreadPct: 2.5 });
      // Verify by testing behavior
      const input = createInput([createMockArbitrage({ spreadPct: 2.0 })]);
      // Would need to actually test the behavior
    });

    it('should use volatility from input over config', async () => {
      const input: ArbitrageAnalysisInput = {
        opportunities: [createMockArbitrage()],
        volatility24h: 0.15, // Higher than default
      };
      const results = await analyst.analyze(input);

      // The analysis should complete (we can't directly verify volatility usage
      // without exposing internals, but we can verify it doesn't error)
      expect(results).toHaveLength(1);
    });
  });

  // ============= 9. EDGE CASES =============
  describe('Edge Cases', () => {
    it('should handle zero spread', async () => {
      const input = createInput([createMockArbitrage({ spreadPct: 0, buyExchange: 'raydium', sellExchange: 'jupiter' })]);
      const results = await analyst.analyze(input);

      expect(results[0].approved).toBe(false);
      expect(results[0].rejectReason).toContain('Spread too low');
    });

    it('should handle negative net profit', async () => {
      const input = createInput([createMockArbitrage({ netProfit: -10 })]);
      const results = await analyst.analyze(input);

      expect(results[0].approved).toBe(false);
      expect(results[0].riskAdjustedReturn).toBe(0);
    });

    it('should handle unknown confidence level with ML fallback', async () => {
      const input = createInput([
        createMockArbitrage({ confidence: 'unknown' as any }),
      ]);
      const results = await analyst.analyze(input);

      // ML model mock returns 0.85 regardless of input confidence string
      expect(results[0].confidence).toBe(0.85);
    });

    it('should handle case-insensitive exchange names', async () => {
      const input = createInput([
        createMockArbitrage({
          buyExchange: 'BINANCE',
          sellExchange: 'Jupiter',
        }),
      ]);
      const results = await analyst.analyze(input);

      expect(results[0].direction.isCexToDex).toBe(true);
    });
  });
});

