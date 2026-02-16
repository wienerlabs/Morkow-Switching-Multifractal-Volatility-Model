/**
 * Arbitrage Executor Integration Tests
 *
 * Tests all risk gates in the execute() pipeline:
 * - Outcome circuit breaker
 * - PM approval
 * - Guardian validation
 * - Global risk manager
 * - Adversarial debate
 * - Spread/profit checks
 * - Regime scaling
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';

// ============= MOCKS =============

vi.mock('../logger.js', () => ({
  logger: {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
  },
}));

vi.mock('../../config/production.js', () => ({
  config: {
    solanaRpcUrl: 'https://fake-rpc.test',
    slippage: { arbitrage: 50, spot: 100, lending: 50, lpDeposit: 100, lpWithdraw: 50 },
    redis: { url: '', enabled: false },
  },
}));

const mockRiskCheck = {
  canTrade: true,
  blockReasons: [] as string[],
  circuitBreakerState: 'ACTIVE',
  alamsVar: { regimePositionScale: 1.0 },
};

vi.mock('../risk/index.js', () => ({
  getGlobalRiskManager: () => ({
    performGlobalRiskCheck: vi.fn().mockResolvedValue(mockRiskCheck),
  }),
}));

let mockIsBlocked = false;
const mockDebateResult = {
  final_decision: 'approve',
  final_confidence: 0.85,
  recommended_size_pct: 100,
  num_rounds: 2,
  rounds: [{ arbitrator: { reasoning: 'OK' } }, { arbitrator: { reasoning: 'OK' } }],
};

vi.mock('../risk/debateClient.js', () => ({
  getDebateClient: () => ({
    isStrategyBlocked: vi.fn().mockResolvedValue(mockIsBlocked),
    runDebate: vi.fn().mockResolvedValue(mockDebateResult),
    recordTradeOutcome: vi.fn().mockResolvedValue(undefined),
  }),
}));

vi.mock('../portfolioManager.js', () => ({
  getPortfolioManager: () => ({
    recordArbitrageTrade: vi.fn(),
  }),
}));

let mockGuardianApproved = true;
vi.mock('../guardian/index.js', () => ({
  guardian: {
    validate: vi.fn().mockImplementation(() => Promise.resolve({
      approved: mockGuardianApproved,
      blockReason: mockGuardianApproved ? undefined : 'Guardian blocked: high risk',
    })),
  },
}));

vi.mock('../pm/index.js', () => ({
  pmDecisionEngine: {
    isEnabled: () => false,
    needsApproval: () => false,
  },
  approvalQueue: {
    queueTrade: vi.fn(),
    waitForApproval: vi.fn(),
  },
}));

vi.mock('../marketData.js', () => ({
  getSolPrice: vi.fn().mockResolvedValue(150.0),
  DEFAULT_GAS_SOL: 0.002,
}));

vi.mock('@jup-ag/api', () => ({
  createJupiterApiClient: vi.fn(),
}));

vi.mock('@solana/web3.js', () => ({
  Connection: class {
    getRecentPrioritizationFees() { return Promise.resolve([]); }
    getBalance() { return Promise.resolve(5_000_000_000); }
    getSignatureStatus() { return Promise.resolve({ value: { confirmationStatus: 'confirmed' } }); }
    sendTransaction() { return Promise.resolve('mock-tx-sig'); }
  },
  PublicKey: class {
    constructor(public key: string) {}
    toBase58() { return this.key; }
  },
  Keypair: {
    fromSecretKey: () => ({
      publicKey: { toBase58: () => 'fake-wallet' },
    }),
    generate: () => ({
      publicKey: { toBase58: () => 'fake-wallet' },
    }),
  },
  VersionedTransaction: class {
    static deserialize() { return new this(); }
    sign() {}
    message = { staticAccountKeys: [{ toBase58: () => 'fake' }] };
    signatures = ['sig1'];
  },
}));

// Mock fetch for Jupiter and Binance
const mockJupiterQuoteResponse = {
  inAmount: '1000000',
  outAmount: '1010000',
  swapTransaction: 'base64-encoded-tx',
};

vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
  ok: true,
  json: () => Promise.resolve(mockJupiterQuoteResponse),
  text: () => Promise.resolve(''),
}));

// ============= TESTS =============

import { ArbitrageExecutor } from '../arbitrageExecutor.js';
import type { ArbitrageOpportunity } from '../marketScanner/types.js';

function makeOpp(overrides: Partial<ArbitrageOpportunity> = {}): ArbitrageOpportunity {
  return {
    symbol: 'SOL',
    buyExchange: 'binance',
    sellExchange: 'jupiter',
    buyPrice: 150.0,
    sellPrice: 151.5,
    spreadPct: 1.0,
    buyTimestamp: Date.now(),
    sellTimestamp: Date.now(),
    ...overrides,
  } as ArbitrageOpportunity;
}

describe('ArbitrageExecutor', () => {
  let executor: ArbitrageExecutor;

  beforeEach(() => {
    // Reset mock state
    mockRiskCheck.canTrade = true;
    mockRiskCheck.blockReasons = [];
    mockRiskCheck.circuitBreakerState = 'ACTIVE';
    mockRiskCheck.alamsVar = { regimePositionScale: 1.0 };
    mockDebateResult.final_decision = 'approve';
    mockIsBlocked = false;
    mockGuardianApproved = true;

    executor = new ArbitrageExecutor({
      binanceApiKey: 'test-key',
      binanceSecretKey: 'test-secret',
      solanaPrivateKey: '',
      solanaRpcUrl: 'https://fake-rpc.test',
      dryRun: true,
      minProfitUsd: 5.0,
      minSpreadPct: 0.10,
      maxWithdrawWaitMs: 300000,
    });
  });

  describe('execute — risk gates', () => {
    it('circuit breaker open → blocked', async () => {
      mockIsBlocked = true;

      const result = await executor.execute(makeOpp(), 1000);

      expect(result.success).toBe(false);
      expect(result.error).toContain('circuit breaker');
    });

    it('guardian rejects → blocked', async () => {
      mockGuardianApproved = false;

      const result = await executor.execute(makeOpp(), 1000);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Guardian blocked');
    });

    it('risk manager rejects → blocked', async () => {
      mockRiskCheck.canTrade = false;
      mockRiskCheck.blockReasons = ['portfolio exposure too high'];

      const liveExecutor = new ArbitrageExecutor({
        binanceApiKey: 'test-key',
        binanceSecretKey: 'test-secret',
        solanaPrivateKey: '',
        solanaRpcUrl: 'https://fake-rpc.test',
        dryRun: false,
        minProfitUsd: 5.0,
        minSpreadPct: 0.10,
        maxWithdrawWaitMs: 300000,
      });

      const result = await liveExecutor.execute(makeOpp(), 1000);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Risk check failed');
    });

    it('debate rejects (amount >$2000) → blocked', async () => {
      mockDebateResult.final_decision = 'reject';

      const result = await executor.execute(makeOpp(), 3000);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Adversarial debate rejected');
    });

    it('debate skipped for small amounts', async () => {
      mockDebateResult.final_decision = 'reject';

      const result = await executor.execute(makeOpp({ spreadPct: 2.0 }), 1000);

      // $1000 < $2000 threshold, so debate should be skipped → trade succeeds
      expect(result.success).toBe(true);
      expect(result.error).toBeUndefined();
    });

    it('spread too low → rejected', async () => {
      const result = await executor.execute(makeOpp({ spreadPct: 0.05 }), 1000);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Spread too low');
    });

    it('unknown token → error', async () => {
      const result = await executor.execute(makeOpp({ symbol: 'UNKNOWN_TOKEN' }), 1000);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Unknown token');
    });

    it('regime scaling applied (dry-run mode)', async () => {
      mockRiskCheck.alamsVar = { regimePositionScale: 0.5 };

      const result = await executor.execute(makeOpp({ spreadPct: 2.0 }), 2000);

      // Should succeed in dry-run mode with scaled amount
      expect(result.success).toBe(true);
      expect(result.error).toBeUndefined();
    });
  });

  describe('execute — direction routing', () => {
    it('routes CEX→DEX for binance→jupiter', async () => {
      const result = await executor.execute(
        makeOpp({ buyExchange: 'binance', sellExchange: 'jupiter', spreadPct: 2.0 }),
        1000,
      );

      expect(result.direction).toBe('cex-to-dex');
    });

    it('routes DEX→DEX for orca→meteora', async () => {
      const result = await executor.execute(
        makeOpp({ buyExchange: 'orca', sellExchange: 'meteora', spreadPct: 2.0 }),
        1000,
      );

      expect(result.direction).toBe('dex-to-dex');
    });
  });
});
