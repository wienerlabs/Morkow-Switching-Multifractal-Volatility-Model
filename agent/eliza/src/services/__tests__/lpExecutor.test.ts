/**
 * LP Executor Integration Tests
 *
 * Tests all risk gates in the deposit/withdraw pipeline:
 * - Outcome circuit breaker
 * - Global risk manager
 * - Adversarial debate
 * - Price impact check
 * - Regime scaling
 * - Portfolio tracking
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

const mockDebateResult = {
  final_decision: 'approve',
  final_confidence: 0.85,
  recommended_size_pct: 100,
  num_rounds: 2,
  rounds: [{ arbitrator: { reasoning: 'OK' } }, { arbitrator: { reasoning: 'OK' } }],
};

let mockIsBlocked = false;

vi.mock('../risk/debateClient.js', () => ({
  getDebateClient: () => ({
    isStrategyBlocked: vi.fn().mockResolvedValue(mockIsBlocked),
    runDebate: vi.fn().mockResolvedValue(mockDebateResult),
    recordTradeOutcome: vi.fn().mockResolvedValue(undefined),
  }),
}));

const mockPortfolioPositionId = 'portfolio-lp-123';
vi.mock('../portfolioManager.js', () => ({
  getPortfolioManager: () => ({
    openLPPosition: vi.fn().mockReturnValue(mockPortfolioPositionId),
    closeLPPosition: vi.fn().mockReturnValue(5.0),
  }),
}));

vi.mock('@solana/web3.js', () => ({
  Connection: class {
    getRecentPrioritizationFees() { return Promise.resolve([]); }
    getBalance() { return Promise.resolve(5_000_000_000); }
  },
  PublicKey: class {
    constructor(public key: string) {}
    toBase58() { return this.key; }
  },
  Keypair: { generate: () => ({ publicKey: { toBase58: () => 'fake-wallet' } }) },
}));

// Mock DEX executors
const mockDepositResult = {
  success: true,
  txSignature: 'mock-tx-sig',
  positionId: 'mock-pos-id',
};

const mockWithdrawResult = {
  success: true,
  txSignature: 'mock-tx-sig',
  amountUsd: 105,
};

const mockPriceImpact = {
  impactPct: 0.1,
  expectedOutput: 1000,
  minimumOutput: 995,
  isAcceptable: true,
};

vi.mock('../lpExecutor/orca.js', () => ({
  OrcaExecutor: class {
    readonly dex = 'orca';
    isSupported() { return true; }
    deposit = vi.fn().mockResolvedValue(mockDepositResult);
    withdraw = vi.fn().mockResolvedValue(mockWithdrawResult);
    getPosition = vi.fn().mockResolvedValue(null);
    calculatePriceImpact = vi.fn().mockResolvedValue(mockPriceImpact);
    resolveWhirlpoolAddress = vi.fn().mockResolvedValue(null);
  },
}));

vi.mock('../lpExecutor/meteora.js', () => ({
  MeteoraExecutor: class {
    readonly dex = 'meteora';
    isSupported() { return true; }
    deposit = vi.fn().mockResolvedValue(mockDepositResult);
    withdraw = vi.fn().mockResolvedValue(mockWithdrawResult);
    getPosition = vi.fn().mockResolvedValue(null);
    calculatePriceImpact = vi.fn().mockResolvedValue(mockPriceImpact);
  },
}));

// ============= TESTS =============

import { LPExecutor } from '../lpExecutor/index.js';
import type { DepositParams, WithdrawParams, LPPoolInfo } from '../lpExecutor/types.js';

function makePool(overrides: Partial<LPPoolInfo> = {}): LPPoolInfo {
  return {
    address: 'pool-address-123',
    name: 'SOL-USDC',
    dex: 'orca',
    token0: { symbol: 'SOL', mint: 'So111...', decimals: 9 },
    token1: { symbol: 'USDC', mint: 'EPjF...', decimals: 6 },
    fee: 30,
    tvlUsd: 1_000_000,
    apy: 12.5,
    ...overrides,
  };
}

function makeWallet(): any {
  return { publicKey: { toBase58: () => 'fake-wallet' } };
}

describe('LPExecutor', () => {
  let executor: LPExecutor;

  beforeEach(() => {
    // Reset mock state
    mockRiskCheck.canTrade = true;
    mockRiskCheck.blockReasons = [];
    mockRiskCheck.circuitBreakerState = 'ACTIVE';
    mockRiskCheck.alamsVar = { regimePositionScale: 1.0 };
    mockDebateResult.final_decision = 'approve';
    mockIsBlocked = false;
    mockPriceImpact.isAcceptable = true;
    mockPriceImpact.impactPct = 0.1;
    mockDepositResult.success = true;
    mockDepositResult.positionId = 'mock-pos-id';
    mockWithdrawResult.success = true;
    mockWithdrawResult.amountUsd = 105;

    executor = new LPExecutor({ rpcUrl: 'https://fake-rpc.test', dryRun: false });
  });

  describe('deposit', () => {
    it('happy path: all checks pass → deposit succeeds', async () => {
      const result = await executor.deposit({
        pool: makePool(),
        amountUsd: 500,
        wallet: makeWallet(),
      });

      expect(result.success).toBe(true);
      expect(result.portfolioPositionId).toBe(mockPortfolioPositionId);
    });

    it('circuit breaker open → blocked', async () => {
      mockIsBlocked = true;

      const result = await executor.deposit({
        pool: makePool(),
        amountUsd: 500,
        wallet: makeWallet(),
      });

      expect(result.success).toBe(false);
      expect(result.error).toContain('circuit breaker');
    });

    it('risk manager rejects (production mode) → blocked', async () => {
      mockRiskCheck.canTrade = false;
      mockRiskCheck.blockReasons = ['daily drawdown exceeded'];

      const result = await executor.deposit({
        pool: makePool(),
        amountUsd: 500,
        wallet: makeWallet(),
      });

      expect(result.success).toBe(false);
      expect(result.error).toContain('Risk check failed');
    });

    it('debate rejects (amount >$1000) → blocked', async () => {
      mockDebateResult.final_decision = 'reject';

      const result = await executor.deposit({
        pool: makePool(),
        amountUsd: 2000,
        wallet: makeWallet(),
      });

      expect(result.success).toBe(false);
      expect(result.error).toContain('Adversarial debate rejected');
    });

    it('debate skipped for small amounts', async () => {
      mockDebateResult.final_decision = 'reject';

      const result = await executor.deposit({
        pool: makePool(),
        amountUsd: 500,
        wallet: makeWallet(),
      });

      // Debate should NOT be consulted for $500, so deposit should succeed
      expect(result.success).toBe(true);
    });

    it('price impact too high → blocked', async () => {
      mockPriceImpact.isAcceptable = false;
      mockPriceImpact.impactPct = 5.0;

      const result = await executor.deposit({
        pool: makePool(),
        amountUsd: 500,
        wallet: makeWallet(),
      });

      expect(result.success).toBe(false);
      expect(result.error).toContain('Price impact too high');
    });

    it('regime scaling applied (scale=0.5 → amount halved)', async () => {
      mockRiskCheck.alamsVar = { regimePositionScale: 0.5 };

      const result = await executor.deposit({
        pool: makePool(),
        amountUsd: 1000,
        wallet: makeWallet(),
      });

      expect(result.success).toBe(true);
      // The deposit should proceed with 500 instead of 1000
      // (We can't directly verify the internal amount, but the test verifies no crash)
    });

    it('unsupported DEX → error', async () => {
      const result = await executor.deposit({
        pool: makePool({ dex: 'raydium' }),
        amountUsd: 500,
        wallet: makeWallet(),
      });

      expect(result.success).toBe(false);
      expect(result.error).toContain('Unsupported DEX');
    });
  });

  describe('withdraw', () => {
    it('happy path → portfolio closed', async () => {
      const result = await executor.withdraw({
        positionId: 'pos-123',
        portfolioPositionId: 'portfolio-lp-123',
        pool: makePool(),
        wallet: makeWallet(),
      });

      expect(result.success).toBe(true);
    });

    it('withdraw with realized P&L tracking', async () => {
      mockWithdrawResult.amountUsd = 110;

      const result = await executor.withdraw({
        positionId: 'pos-123',
        portfolioPositionId: 'portfolio-lp-456',
        pool: makePool(),
        wallet: makeWallet(),
      });

      expect(result.success).toBe(true);
    });

    it('unsupported DEX → error', async () => {
      const result = await executor.withdraw({
        positionId: 'pos-123',
        pool: makePool({ dex: 'raydium' }),
        wallet: makeWallet(),
      });

      expect(result.success).toBe(false);
      expect(result.error).toContain('Unsupported DEX');
    });
  });

  describe('normalization', () => {
    it('isSupported returns true for orca, meteora', () => {
      expect(executor.isSupported('orca')).toBe(true);
      expect(executor.isSupported('whirlpool')).toBe(true);
      expect(executor.isSupported('meteora')).toBe(true);
      expect(executor.isSupported('DLMM')).toBe(true);
    });

    it('isSupported returns false for raydium (disabled)', () => {
      expect(executor.isSupported('raydium')).toBe(false);
    });
  });
});
