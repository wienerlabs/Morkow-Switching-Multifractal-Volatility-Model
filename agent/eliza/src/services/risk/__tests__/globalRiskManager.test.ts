/**
 * Global Risk Manager Tests
 * 
 * Comprehensive tests for all 6 risk management features:
 * 1. Drawdown Circuit Breakers (Daily 5%, Weekly 10%, Monthly 15%)
 * 2. Correlation Risk Tracking (Asset/Protocol exposure limits)
 * 3. Dynamic Stop Loss (ATR-based or asset class)
 * 4. Protocol Concentration (Max 50% per protocol)
 * 5. Oracle Staleness (30s reject, 60s emergency)
 * 6. Emergency Gas Budget (Real-time gas monitoring)
 */
import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { 
  GlobalRiskManager,
  resetGlobalRiskManager,
  DEFAULT_DRAWDOWN_LIMITS,
  DEFAULT_EXPOSURE_LIMITS,
  DEFAULT_STOP_LOSS_CONFIG,
} from '../globalRiskManager.js';
import type { TrackedPosition } from '../types.js';

// ============= MOCKS =============

// Mock DB
const mockDbStatements = new Map<string, any>();
const mockDb = {
  prepare: (sql: string) => ({
    get: (..._args: any[]) => {
      // Return null for risk_state (no persisted state)
      if (sql.includes('risk_state')) return null;
      return null;
    },
    all: (..._args: any[]) => {
      // Return empty array for risk_alerts
      return [];
    },
    run: (..._args: any[]) => ({ changes: 1 }),
  }),
  pragma: () => {},
  exec: () => {},
  transaction: (fn: () => void) => fn,
};

vi.mock('../../db/index.js', () => ({
  getDb: () => mockDb,
}));

// Mock Solana failover connection — returns a mock Connection-like object
vi.mock('../../solana/connection.js', () => ({
  getSolanaConnection: () => ({
    getRecentPrioritizationFees: () => Promise.resolve([
      { slot: 1, prioritizationFee: 1000 },
      { slot: 2, prioritizationFee: 2000 },
    ]),
    getBalance: () => Promise.resolve(5_000_000_000),
  }),
  recordRpcFailure: () => {},
  recordRpcSuccess: () => {},
  getActiveRpcUrl: () => 'https://fake',
}));

// Mock portfolioManager
const mockPortfolioState = {
  totalValueUsd: 10000,
  lpPositions: new Map(),
  perpsPositions: new Map(),
  trades: [],
};

vi.mock('../../portfolioManager.js', () => ({
  getPortfolioManager: () => ({
    getTotalValueUsd: () => mockPortfolioState.totalValueUsd,
    getState: () => mockPortfolioState,
    getOpenLPPositions: () => Array.from(mockPortfolioState.lpPositions.values()),
  }),
}));

// Mock logger
vi.mock('../../logger.js', () => ({
  logger: {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
  },
}));

// Mock Solana Connection - factory must not reference outside variables
vi.mock('@solana/web3.js', () => {
  return {
    Connection: class {
      getRecentPrioritizationFees() {
        return Promise.resolve([
          { slot: 1, prioritizationFee: 1000 },
          { slot: 2, prioritizationFee: 2000 },
          { slot: 3, prioritizationFee: 1500 },
        ]);
      }
    },
    PublicKey: class {
      key: string;
      constructor(key: string) { this.key = key; }
      toBase58() { return this.key; }
    },
  };
});

// Mock fetch for Jupiter/Birdeye APIs
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('GlobalRiskManager', () => {
  let manager: GlobalRiskManager;
  
  beforeEach(() => {
    vi.clearAllMocks();
    resetGlobalRiskManager();
    mockPortfolioState.totalValueUsd = 10000;
    mockPortfolioState.lpPositions = new Map();
    mockPortfolioState.perpsPositions = new Map();
    
    // Default mock for Jupiter price API
    mockFetch.mockResolvedValue({
      json: () => Promise.resolve({
        data: {
          'So11111111111111111111111111111111111111112': {
            price: 100,
            extraInfo: { lastSwappedPrice: { lastJupiterSellAt: Date.now() / 1000 } },
          },
        },
      }),
    });
    
    manager = new GlobalRiskManager();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // ============= DEFAULT CONFIG TESTS =============

  describe('Default Configuration', () => {
    it('should have correct drawdown limits', () => {
      expect(DEFAULT_DRAWDOWN_LIMITS.daily).toBe(5.0);
      expect(DEFAULT_DRAWDOWN_LIMITS.weekly).toBe(10.0);
      expect(DEFAULT_DRAWDOWN_LIMITS.monthly).toBe(15.0);
    });

    it('should have correct exposure limits', () => {
      expect(DEFAULT_EXPOSURE_LIMITS.maxBaseAssetPct).toBe(40);
      expect(DEFAULT_EXPOSURE_LIMITS.maxQuoteAssetPct).toBe(60);
      expect(DEFAULT_EXPOSURE_LIMITS.maxProtocolPct).toBe(50);
    });

    it('should have correct stop loss config', () => {
      expect(DEFAULT_STOP_LOSS_CONFIG.majorStopPct).toBe(3.0);
      expect(DEFAULT_STOP_LOSS_CONFIG.midcapStopPct).toBe(5.0);
      expect(DEFAULT_STOP_LOSS_CONFIG.altStopPct).toBe(7.0);
      expect(DEFAULT_STOP_LOSS_CONFIG.useATR).toBe(true);
      expect(DEFAULT_STOP_LOSS_CONFIG.atrMultiplier).toBe(2.0);
    });
  });

  // ============= 1. DRAWDOWN CIRCUIT BREAKERS =============

  describe('1. Drawdown Circuit Breakers', () => {
    it('should start in ACTIVE state', () => {
      expect(manager.getCircuitBreakerState()).toBe('ACTIVE');
      expect(manager.isCircuitBreakerActive()).toBe(false);
    });

    it('should calculate zero drawdown when value equals start', () => {
      const status = manager.calculateDrawdownStatus();
      expect(status.dailyDrawdownPct).toBe(0);
      expect(status.weeklyDrawdownPct).toBe(0);
      expect(status.monthlyDrawdownPct).toBe(0);
      expect(status.circuitBreakerState).toBe('ACTIVE');
    });

    it('should trigger PAUSED state when daily drawdown >= 5%', () => {
      // Simulate 6% daily loss (10000 -> 9400)
      mockPortfolioState.totalValueUsd = 9400;
      
      const status = manager.calculateDrawdownStatus();
      expect(status.dailyDrawdownPct).toBeCloseTo(6, 0);
      expect(status.circuitBreakerState).toBe('PAUSED');
      expect(manager.isCircuitBreakerActive()).toBe(true);
    });

    it('should trigger STOPPED state when weekly drawdown >= 10%', () => {
      // Simulate 12% weekly loss
      mockPortfolioState.totalValueUsd = 8800;
      
      const status = manager.calculateDrawdownStatus();
      expect(status.weeklyDrawdownPct).toBeCloseTo(12, 0);
      expect(status.circuitBreakerState).toBe('STOPPED');
    });

    it('should trigger LOCKDOWN state when monthly drawdown >= 15%', () => {
      // Simulate 18% monthly loss
      mockPortfolioState.totalValueUsd = 8200;

      const status = manager.calculateDrawdownStatus();
      expect(status.monthlyDrawdownPct).toBeCloseTo(18, 0);
      expect(status.circuitBreakerState).toBe('LOCKDOWN');
    });

    it('should track peak value correctly', () => {
      // Value goes up
      mockPortfolioState.totalValueUsd = 12000;
      let status = manager.calculateDrawdownStatus();
      expect(status.peakValueUsd).toBe(12000);

      // Value goes down - peak should remain
      mockPortfolioState.totalValueUsd = 11000;
      status = manager.calculateDrawdownStatus();
      expect(status.peakValueUsd).toBe(12000);
    });

    it('should reject invalid reset confirmation', () => {
      // Trigger circuit breaker
      mockPortfolioState.totalValueUsd = 9000;
      manager.calculateDrawdownStatus();

      // Try to reset with wrong phrase
      const result = manager.resetCircuitBreaker('wrong_phrase');
      expect(result).toBe(false);
      expect(manager.getCircuitBreakerState()).not.toBe('ACTIVE');
    });

    it('should accept valid reset confirmation', () => {
      // Trigger circuit breaker
      mockPortfolioState.totalValueUsd = 9000;
      manager.calculateDrawdownStatus();

      // Reset with correct phrase
      const result = manager.resetCircuitBreaker('CONFIRM_RESET');
      expect(result).toBe(true);
      expect(manager.getCircuitBreakerState()).toBe('ACTIVE');
    });
  });

  // ============= 2. CORRELATION RISK TRACKING =============

  describe('2. Correlation Risk Tracking', () => {
    it('should return compliant status with no positions', () => {
      const risk = manager.calculateCorrelationRisk();
      expect(risk.isCompliant).toBe(true);
      expect(risk.violations).toHaveLength(0);
    });

    it('should calculate base asset exposure correctly', () => {
      // Add LP position SOL/USDC worth $5000 (50% of portfolio)
      mockPortfolioState.lpPositions.set('lp_1', {
        id: 'lp_1',
        poolName: 'SOL/USDC',
        dex: 'raydium',
        currentValueUsd: 5000,
        entryPriceUsd: 5000,
        capitalUsd: 5000,
        entryTime: Date.now(),
      });

      const risk = manager.calculateCorrelationRisk();

      // SOL exposure should be 25% (50% of $5000 = $2500 out of $10000)
      const solExposure = risk.baseAssetExposures.find(e => e.asset === 'SOL');
      expect(solExposure).toBeDefined();
      expect(solExposure!.exposurePct).toBeCloseTo(25, 0);
    });

    it('should detect base asset exposure violation (>40%)', () => {
      // Add large SOL position - $9000 in SOL/USDC (90% of portfolio)
      mockPortfolioState.lpPositions.set('lp_1', {
        id: 'lp_1',
        poolName: 'SOL/USDC',
        dex: 'raydium',
        currentValueUsd: 9000,
        entryPriceUsd: 9000,
        capitalUsd: 9000,
        entryTime: Date.now(),
      });

      const risk = manager.calculateCorrelationRisk();

      // SOL exposure = 45% (50% of $9000 = $4500 out of $10000) > 40% limit
      expect(risk.isCompliant).toBe(false);
      expect(risk.violations.some(v => v.includes('Base asset SOL'))).toBe(true);
    });

    it('should detect protocol concentration violation (>50%)', () => {
      // Add two large positions on same protocol
      mockPortfolioState.lpPositions.set('lp_1', {
        id: 'lp_1',
        poolName: 'SOL/USDC',
        dex: 'raydium',
        currentValueUsd: 3000,
        entryPriceUsd: 3000,
        capitalUsd: 3000,
        entryTime: Date.now(),
      });
      mockPortfolioState.lpPositions.set('lp_2', {
        id: 'lp_2',
        poolName: 'ETH/USDC',
        dex: 'raydium', // Same protocol
        currentValueUsd: 3000,
        entryPriceUsd: 3000,
        capitalUsd: 3000,
        entryTime: Date.now(),
      });

      const risk = manager.calculateCorrelationRisk();

      // Protocol exposure = 60% ($6000 out of $10000) > 50% limit
      expect(risk.isCompliant).toBe(false);
      expect(risk.violations.some(v => v.includes('Protocol raydium'))).toBe(true);
    });

    it('should check if new position would violate limits', () => {
      // Current: $3000 in SOL/USDC on raydium
      mockPortfolioState.lpPositions.set('lp_1', {
        id: 'lp_1',
        poolName: 'SOL/USDC',
        dex: 'raydium',
        currentValueUsd: 3000,
        entryPriceUsd: 3000,
        capitalUsd: 3000,
        entryTime: Date.now(),
      });

      // Try to add $5000 more to raydium (would be 8000/13000 = 61.5% > 50%)
      const check = manager.wouldViolateExposureLimits({
        baseAsset: 'ETH',
        quoteAsset: 'USDC',
        protocol: 'raydium',
        sizeUsd: 5000,
      });

      expect(check.allowed).toBe(false);
      expect(check.violations.some(v => v.includes('raydium'))).toBe(true);
    });
  });

  // ============= 3. DYNAMIC STOP LOSS =============

  describe('3. Dynamic Stop Loss', () => {
    it('should classify major assets correctly', () => {
      expect(manager.classifyAsset('BTC')).toBe('major');
      expect(manager.classifyAsset('ETH')).toBe('major');
      expect(manager.classifyAsset('SOL')).toBe('major');
      expect(manager.classifyAsset('USDC')).toBe('major');
    });

    it('should classify midcap assets correctly', () => {
      expect(manager.classifyAsset('JUP')).toBe('midcap');
      expect(manager.classifyAsset('RAY')).toBe('midcap');
      expect(manager.classifyAsset('ORCA')).toBe('midcap');
    });

    it('should classify alt assets correctly', () => {
      expect(manager.classifyAsset('BONK')).toBe('alt');
      expect(manager.classifyAsset('WIF')).toBe('alt');
      expect(manager.classifyAsset('RANDOMTOKEN')).toBe('alt');
    });

    it('should recommend 3% stop for major assets', () => {
      const result = manager.getRecommendedStopLoss('SOL');
      expect(result.classification).toBe('major');
      expect(result.recommendedStopPct).toBe(3.0);
    });

    it('should recommend 5% stop for midcap assets', () => {
      const result = manager.getRecommendedStopLoss('JUP');
      expect(result.classification).toBe('midcap');
      expect(result.recommendedStopPct).toBe(5.0);
    });

    it('should recommend 7% stop for alt assets', () => {
      const result = manager.getRecommendedStopLoss('BONK');
      expect(result.classification).toBe('alt');
      expect(result.recommendedStopPct).toBe(7.0);
    });

    it('should use ATR-based stop when ATR is higher', () => {
      // ATR = 10, price = 100, so ATR% = 10%
      // ATR-based stop = 10% * 2 (multiplier) = 20%
      // This should be used instead of 3% major stop
      const result = manager.getRecommendedStopLoss('SOL', 10, 100);
      expect(result.recommendedStopPct).toBe(20);
    });

    it('should use class-based stop when ATR is lower', () => {
      // ATR = 1, price = 100, so ATR% = 1%
      // ATR-based stop = 1% * 2 = 2%
      // Should use 3% major stop instead
      const result = manager.getRecommendedStopLoss('SOL', 1, 100);
      expect(result.recommendedStopPct).toBe(3.0);
    });

    it('should detect when position should be stopped out', () => {
      const position: TrackedPosition = {
        id: 'test_1',
        type: 'lp',
        protocol: 'raydium',
        baseAsset: 'SOL',
        quoteAsset: 'USDC',
        sizeUsd: 1000,
        entryPrice: 100,
        currentPrice: 95, // -5% loss
        unrealizedPnlUsd: -50,
        entryTime: new Date(),
        stopLossPct: 3, // 3% stop
      };

      const result = manager.shouldStopOut(position);
      expect(result.shouldStop).toBe(true);
      expect(result.reason).toContain('hit stop loss');
    });

    it('should not stop out position within threshold', () => {
      const position: TrackedPosition = {
        id: 'test_1',
        type: 'lp',
        protocol: 'raydium',
        baseAsset: 'SOL',
        quoteAsset: 'USDC',
        sizeUsd: 1000,
        entryPrice: 100,
        currentPrice: 98, // -2% loss
        unrealizedPnlUsd: -20,
        entryTime: new Date(),
        stopLossPct: 3, // 3% stop
      };

      const result = manager.shouldStopOut(position);
      expect(result.shouldStop).toBe(false);
    });
  });

  // ============= 4. PROTOCOL CONCENTRATION (Already covered in #2) =============

  describe('4. Protocol Concentration Limit', () => {
    it('should enforce 50% max protocol concentration', () => {
      // Add position worth 55% of portfolio
      mockPortfolioState.lpPositions.set('lp_1', {
        id: 'lp_1',
        poolName: 'SOL/USDC',
        dex: 'drift',
        currentValueUsd: 5500,
        entryPriceUsd: 5500,
        capitalUsd: 5500,
        entryTime: Date.now(),
      });

      const risk = manager.calculateCorrelationRisk();
      expect(risk.isCompliant).toBe(false);
      expect(risk.violations.some(v => v.includes('drift'))).toBe(true);
    });

    it('should allow positions within 50% limit', () => {
      // Add position worth 45% of portfolio
      mockPortfolioState.lpPositions.set('lp_1', {
        id: 'lp_1',
        poolName: 'SOL/USDC',
        dex: 'drift',
        currentValueUsd: 4500,
        entryPriceUsd: 4500,
        capitalUsd: 4500,
        entryTime: Date.now(),
      });

      const risk = manager.calculateCorrelationRisk();

      // Check that protocol is within limit
      const driftExposure = risk.protocolExposures.find(e => e.protocol === 'drift');
      expect(driftExposure!.exposurePct).toBe(45);
      expect(risk.violations.some(v => v.includes('drift'))).toBe(false);
    });
  });

  // ============= 5. ORACLE STALENESS PROTECTION =============

  describe('5. Oracle Staleness Protection', () => {
    it('should accept fresh oracle price (<30s)', async () => {
      // Mock Jupiter response with fresh timestamp
      mockFetch.mockResolvedValueOnce({
        json: () => Promise.resolve({
          data: {
            'So11111111111111111111111111111111111111112': {
              price: 100,
              extraInfo: {
                lastSwappedPrice: {
                  lastJupiterSellAt: Date.now() / 1000 - 10 // 10 seconds ago
                }
              },
            },
          },
        }),
      });

      const result = await manager.validateOracleForTrade('SOL');
      expect(result.valid).toBe(true);
      expect(result.price).toBe(100);
    });

    it('should reject stale oracle price (>30s)', async () => {
      // Mock Jupiter response with stale timestamp
      mockFetch.mockResolvedValueOnce({
        json: () => Promise.resolve({
          data: {
            'So11111111111111111111111111111111111111112': {
              price: 100,
              extraInfo: {
                lastSwappedPrice: {
                  lastJupiterSellAt: Date.now() / 1000 - 45 // 45 seconds ago
                }
              },
            },
          },
        }),
      });

      const result = await manager.validateOracleForTrade('SOL');
      expect(result.valid).toBe(false);
      expect(result.reason).toContain('stale');
    });

    it('should trigger emergency on very stale price (>60s)', async () => {
      // Mock Jupiter response with very stale timestamp
      mockFetch.mockResolvedValueOnce({
        json: () => Promise.resolve({
          data: {
            'So11111111111111111111111111111111111111112': {
              price: 100,
              extraInfo: {
                lastSwappedPrice: {
                  lastJupiterSellAt: Date.now() / 1000 - 120 // 120 seconds ago
                }
              },
            },
          },
        }),
      });

      const status = await manager.checkOraclePrice('SOL');
      expect(status.isEmergency).toBe(true);
      expect(status.isStale).toBe(true);
    });

    it('should handle oracle API failure gracefully', async () => {
      // Mock API failure
      mockFetch.mockRejectedValueOnce(new Error('API unavailable'));

      const result = await manager.validateOracleForTrade('SOL');
      expect(result.valid).toBe(false);
      expect(result.reason).toContain('unavailable');
    });

    it('should cache price for exposure calculations', async () => {
      mockFetch.mockResolvedValueOnce({
        json: () => Promise.resolve({
          data: {
            'So11111111111111111111111111111111111111112': {
              price: 150,
              extraInfo: {
                lastSwappedPrice: {
                  lastJupiterSellAt: Date.now() / 1000
                }
              },
            },
          },
        }),
      });

      await manager.checkOraclePrice('SOL');
      // Price should be cached internally for use in calculations
      // (We can't directly test the private cache, but this ensures no errors)
    });
  });

  // ============= 6. EMERGENCY GAS BUDGET =============

  describe('6. Emergency Gas Budget', () => {
    it('should allow trading when gas reserve is sufficient', async () => {
      // With 1 SOL balance and SOL at $100, we have $100 reserve
      const result = await manager.getGasBudgetStatus(1);
      expect(result.canTrade).toBe(true);
      expect(result.canEmergencyExit).toBe(true);
    });

    it('should return recommended priority fee', async () => {
      const result = await manager.getGasBudgetStatus(1);
      expect(result.recommendedPriorityFee).toBeDefined();
      expect(typeof result.recommendedPriorityFee).toBe('number');
    });

    it('should report reserve balance based on wallet', async () => {
      // The gas budget check depends on SOL price being set
      // When SOL price is 0 (not set), gas costs are 0, so any balance is sufficient
      // This tests that the function runs without error and returns expected structure
      const result = await manager.getGasBudgetStatus(0.001);

      // Result should have the expected structure
      expect(typeof result.canTrade).toBe('boolean');
      expect(typeof result.canEmergencyExit).toBe('boolean');
      expect(typeof result.recommendedPriorityFee).toBe('number');
    });
  });

  // ============= GLOBAL RISK CHECK =============

  describe('Global Risk Check (Integration)', () => {
    it('should allow trade when all checks pass', async () => {
      // Fresh oracle
      mockFetch.mockResolvedValueOnce({
        json: () => Promise.resolve({
          data: {
            'So11111111111111111111111111111111111111112': {
              price: 100,
              extraInfo: {
                lastSwappedPrice: {
                  lastJupiterSellAt: Date.now() / 1000
                }
              },
            },
          },
        }),
      });

      const result = await manager.performGlobalRiskCheck({
        symbol: 'SOL',
        protocol: 'raydium',
        sizeUsd: 1000,
        walletBalanceSol: 1,
      });

      expect(result.canTrade).toBe(true);
      expect(result.blockReasons).toHaveLength(0);
    });

    it('should block trade when circuit breaker is active', async () => {
      // Trigger circuit breaker
      mockPortfolioState.totalValueUsd = 9000;
      manager.calculateDrawdownStatus();

      // Fresh oracle
      mockFetch.mockResolvedValueOnce({
        json: () => Promise.resolve({
          data: {
            'So11111111111111111111111111111111111111112': {
              price: 100,
              extraInfo: {
                lastSwappedPrice: {
                  lastJupiterSellAt: Date.now() / 1000
                }
              },
            },
          },
        }),
      });

      const result = await manager.performGlobalRiskCheck({
        symbol: 'SOL',
        protocol: 'raydium',
        sizeUsd: 1000,
        walletBalanceSol: 1,
      });

      expect(result.canTrade).toBe(false);
      expect(result.blockReasons.some(r => r.includes('Circuit breaker'))).toBe(true);
    });

    it('should block trade when oracle is stale', async () => {
      // Stale oracle
      mockFetch.mockResolvedValueOnce({
        json: () => Promise.resolve({
          data: {
            'So11111111111111111111111111111111111111112': {
              price: 100,
              extraInfo: {
                lastSwappedPrice: {
                  lastJupiterSellAt: Date.now() / 1000 - 60
                }
              },
            },
          },
        }),
      });

      const result = await manager.performGlobalRiskCheck({
        symbol: 'SOL',
        protocol: 'raydium',
        sizeUsd: 1000,
        walletBalanceSol: 1,
      });

      expect(result.canTrade).toBe(false);
      expect(result.blockReasons.some(r => r.includes('Oracle stale') || r.includes('stale'))).toBe(true);
    });

    it('should return full risk summary', async () => {
      const summary = await manager.getRiskSummary(1);

      expect(summary.circuitBreaker).toBe('ACTIVE');
      expect(summary.drawdown).toBeDefined();
      expect(summary.correlationRisk).toBeDefined();
      expect(summary.gasBudget).toBeDefined();
      expect(summary.alerts).toBeDefined();
      expect(typeof summary.canTrade).toBe('boolean');
    });
  });

  // ============= ALERT SYSTEM =============

  describe('Alert System', () => {
    it('should start with no alerts', () => {
      const alerts = manager.getAlerts();
      expect(alerts).toHaveLength(0);
    });

    it('should generate alert on circuit breaker trigger', () => {
      mockPortfolioState.totalValueUsd = 9000;
      manager.calculateDrawdownStatus();

      const alerts = manager.getAlerts();
      expect(alerts.length).toBeGreaterThan(0);
      expect(alerts[0].category).toBe('drawdown');
    });

    it('should limit alerts to requested count', () => {
      // Trigger multiple alerts
      mockPortfolioState.totalValueUsd = 9000;
      manager.calculateDrawdownStatus();
      mockPortfolioState.totalValueUsd = 8500;
      manager.calculateDrawdownStatus();

      const alerts = manager.getAlerts(1);
      expect(alerts).toHaveLength(1);
    });
  });
});

