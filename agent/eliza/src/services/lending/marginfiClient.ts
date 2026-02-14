/**
 * MarginFi Client
 *
 * Production client for MarginFi lending protocol on Solana.
 * Supports deposits, withdrawals, borrows, and repayments.
 * Includes Guardian pre-execution validation.
 */

import { Connection, Keypair, PublicKey } from '@solana/web3.js';
import { MarginfiClient, getConfig } from '@mrgnlabs/marginfi-client-v2';
import type { MarginfiAccountWrapper } from '@mrgnlabs/marginfi-client-v2';
import { Wallet } from '@mrgnlabs/mrgn-common';
import bs58 from 'bs58';
import BigNumber from 'bignumber.js';

import { logger } from '../logger.js';
import { guardian } from '../guardian/index.js';
import type { GuardianTradeParams } from '../guardian/types.js';
import { pmDecisionEngine, approvalQueue } from '../pm/index.js';
import type { QueueTradeParams } from '../pm/types.js';
import type {
  DepositParams,
  WithdrawParams,
  BorrowParams,
  RepayParams,
  LendingPosition,
  LendingAPY,
  LendingResult,
} from './types.js';

export interface MarginFiClientConfig {
  rpcUrl: string;
  privateKey: string;
  environment?: 'production' | 'staging' | 'dev';
}

export class MarginFiLendingClient {
  private connection: Connection;
  private keypair: Keypair;
  private wallet: Wallet;
  private client: MarginfiClient | null = null;
  private account: MarginfiAccountWrapper | null = null;
  private initialized = false;

  constructor(private config: MarginFiClientConfig) {
    this.connection = new Connection(config.rpcUrl, 'confirmed');
    this.keypair = Keypair.fromSecretKey(bs58.decode(config.privateKey));
    this.wallet = {
      publicKey: this.keypair.publicKey,
      signTransaction: async <T extends { sign: (keypair: Keypair) => void }>(tx: T): Promise<T> => {
        tx.sign(this.keypair);
        return tx;
      },
      signAllTransactions: async <T extends { sign: (keypair: Keypair) => void }>(txs: T[]): Promise<T[]> => {
        txs.forEach((tx: T) => tx.sign(this.keypair));
        return txs;
      },
    } as Wallet;
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;

    // More aggressive retry strategy for RPC rate limiting
    const maxRetries = 5;
    const baseDelayMs = 5000; // 5 seconds base delay

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        logger.info('[MARGINFI] Initializing client...', { attempt, maxRetries });

        // Get config for production
        const marginfiConfig = getConfig(this.config.environment || 'production');

        // Initialize MarginFi client with retry-aware connection
        this.client = await MarginfiClient.fetch(
          marginfiConfig,
          this.wallet,
          this.connection,
          { confirmOpts: { commitment: 'confirmed' } }
        );

        // Get or create marginfi account
        const accounts = await this.client.getMarginfiAccountsForAuthority(
          this.keypair.publicKey
        );

        if (accounts.length > 0) {
          this.account = accounts[0];
          logger.info('[MARGINFI] Using existing account', {
            address: this.account.address.toBase58(),
          });
        } else {
          // Create new account
          this.account = await this.client.createMarginfiAccount();
          logger.info('[MARGINFI] Created new account', {
            address: this.account.address.toBase58(),
          });
        }

        this.initialized = true;
        logger.info('[MARGINFI] Client initialized successfully');
        return; // Success - exit retry loop
      } catch (error: any) {
        const isRetryable = error?.message?.includes('rate limit') ||
                           error?.message?.includes('429') ||
                           error?.message?.includes('fetch') ||
                           error?.message?.includes('retries') ||
                           error?.message?.includes('timeout') ||
                           error?.message?.includes('ECONNRESET');

        const delayMs = baseDelayMs * attempt; // 5s, 10s, 15s, 20s, 25s

        if (attempt < maxRetries && isRetryable) {
          logger.warn('[MARGINFI] Initialization failed, retrying...', {
            attempt,
            maxRetries,
            error: error?.message,
            retryInMs: delayMs,
          });
          await new Promise(resolve => setTimeout(resolve, delayMs));
          continue;
        }

        // On final failure, log but don't throw - allow system to continue without MarginFi
        logger.error('[MARGINFI] Failed to initialize after all retries - MarginFi will be unavailable', {
          errorMessage: error?.message || 'Unknown error',
          errorName: error?.name,
          errorCode: error?.code,
          rpcUrl: this.config.rpcUrl?.substring(0, 50) + '...',
          environment: this.config.environment,
          totalAttempts: attempt,
          recommendation: 'Consider using a dedicated RPC endpoint for MarginFi',
        });
        // Don't throw - allow the system to continue without MarginFi
        // The initialized flag remains false, so operations will fail gracefully
        return;
      }
    }
  }

  private ensureInitialized(): void {
    if (!this.initialized || !this.client || !this.account) {
      throw new Error('MarginFi client not initialized. Call initialize() first.');
    }
  }

  async deposit(params: DepositParams): Promise<LendingResult> {
    try {
      this.ensureInitialized();
      logger.info('[MARGINFI] Depositing', { asset: params.asset, amount: params.amount });

      // ========== PM APPROVAL CHECK (before Guardian) ==========
      if (pmDecisionEngine.isEnabled()) {
        const pmParams: QueueTradeParams = {
          strategy: 'lending',
          action: 'DEPOSIT',
          asset: params.asset,
          amount: params.amount,
          amountUsd: params.amount,
          confidence: 0.7,
          risk: {
            volatility: 0,
            liquidityScore: 80,
            riskScore: 20,
          },
          reasoning: `Deposit ${params.amount} ${params.asset} to MarginFi`,
          protocol: 'marginfi',
        };

        const portfolioValueUsd = 10000;
        const needsApproval = pmDecisionEngine.needsApproval(pmParams, portfolioValueUsd);

        if (needsApproval) {
          logger.info('[MARGINFI] Deposit requires PM approval', {
            asset: params.asset,
            amount: params.amount,
          });

          const tradeId = approvalQueue.queueTrade(pmParams);
          const approvalResult = await approvalQueue.waitForApproval(tradeId);

          if (!approvalResult.approved) {
            logger.warn('[MARGINFI] PM rejected deposit', {
              tradeId,
              status: approvalResult.status,
              reason: approvalResult.rejectionReason,
            });
            return { success: false, error: `PM rejected: ${approvalResult.rejectionReason || approvalResult.status}` };
          }
        }
      }
      // ========================================

      // ========== GUARDIAN PRE-EXECUTION VALIDATION ==========
      const guardianParams: GuardianTradeParams = {
        inputMint: params.asset, // Asset being deposited
        outputMint: params.asset, // Same for lending deposits
        amountIn: params.amount,
        amountInUsd: params.amount, // Assumes USD-pegged asset or needs price conversion
        slippageBps: 50, // Minimal for lending
        strategy: 'lending',
        protocol: 'marginfi',
        walletAddress: this.keypair.publicKey.toBase58(),
      };

      const guardianResult = await guardian.validate(guardianParams);
      if (!guardianResult.approved) {
        logger.warn('[MARGINFI] Guardian blocked deposit', {
          reason: guardianResult.blockReason,
          asset: params.asset,
          amount: params.amount,
        });
        return { success: false, error: `Guardian blocked: ${guardianResult.blockReason}` };
      }
      // ========================================

      const bank = this.client!.getBankByTokenSymbol(params.asset);
      if (!bank) {
        throw new Error(`Bank not found for asset: ${params.asset}`);
      }

      // Deposit using UI amount (SDK handles conversion)
      const signature = await this.account!.deposit(
        params.amount,
        bank.address
      );

      logger.info('[MARGINFI] Deposit successful', {
        asset: params.asset,
        amount: params.amount,
        signature,
      });

      return { success: true, signature };
    } catch (error: any) {
      logger.error('[MARGINFI] Deposit failed', { error: error.message, asset: params.asset, amount: params.amount });
      return { success: false, error: error.message };
    }
  }

  async withdraw(params: WithdrawParams): Promise<LendingResult> {
    try {
      this.ensureInitialized();
      logger.info('[MARGINFI] Withdrawing', { asset: params.asset, amount: params.amount });

      // ========== GUARDIAN PRE-EXECUTION VALIDATION ==========
      const guardianParams: GuardianTradeParams = {
        inputMint: params.asset,
        outputMint: params.asset,
        amountIn: params.amount,
        amountInUsd: params.amount,
        slippageBps: 50,
        strategy: 'lending',
        protocol: 'marginfi',
        walletAddress: this.keypair.publicKey.toBase58(),
      };

      const guardianResult = await guardian.validate(guardianParams);
      if (!guardianResult.approved) {
        logger.warn('[MARGINFI] Guardian blocked withdrawal', {
          reason: guardianResult.blockReason,
          asset: params.asset,
          amount: params.amount,
        });
        return { success: false, error: `Guardian blocked: ${guardianResult.blockReason}` };
      }
      // ========================================

      const bank = this.client!.getBankByTokenSymbol(params.asset);
      if (!bank) {
        throw new Error(`Bank not found for asset: ${params.asset}`);
      }

      // Withdraw using UI amount
      const signatures = await this.account!.withdraw(
        params.amount,
        bank.address,
        params.withdrawAll ?? false
      );

      const signature = Array.isArray(signatures) ? signatures[0] : signatures;

      logger.info('[MARGINFI] Withdraw successful', {
        asset: params.asset,
        amount: params.amount,
        signature,
      });

      return { success: true, signature };
    } catch (error: any) {
      logger.error('[MARGINFI] Withdraw failed', { error: error.message, asset: params.asset, amount: params.amount });
      return { success: false, error: error.message };
    }
  }

  async borrow(params: BorrowParams): Promise<LendingResult> {
    try {
      this.ensureInitialized();
      logger.info('[MARGINFI] Borrowing', { asset: params.asset, amount: params.amount });

      // ========== PM APPROVAL CHECK (before Guardian) ==========
      if (pmDecisionEngine.isEnabled()) {
        const pmParams: QueueTradeParams = {
          strategy: 'lending',
          action: 'BORROW',
          asset: params.asset,
          amount: params.amount,
          amountUsd: params.amount,
          confidence: 0.7,
          risk: {
            volatility: 0,
            liquidityScore: 70,
            riskScore: 40,
          },
          reasoning: `Borrow ${params.amount} ${params.asset} from MarginFi`,
          protocol: 'marginfi',
        };

        const portfolioValueUsd = 10000;
        const needsApproval = pmDecisionEngine.needsApproval(pmParams, portfolioValueUsd);

        if (needsApproval) {
          logger.info('[MARGINFI] Borrow requires PM approval', {
            asset: params.asset,
            amount: params.amount,
          });

          const tradeId = approvalQueue.queueTrade(pmParams);
          const approvalResult = await approvalQueue.waitForApproval(tradeId);

          if (!approvalResult.approved) {
            logger.warn('[MARGINFI] PM rejected borrow', {
              tradeId,
              status: approvalResult.status,
              reason: approvalResult.rejectionReason,
            });
            return { success: false, error: `PM rejected: ${approvalResult.rejectionReason || approvalResult.status}` };
          }
        }
      }
      // ========================================

      const guardianParams: GuardianTradeParams = {
        inputMint: params.asset,
        outputMint: params.asset,
        amountIn: params.amount,
        amountInUsd: params.amount,
        slippageBps: 50,
        strategy: 'lending',
        protocol: 'marginfi',
        walletAddress: this.keypair.publicKey.toBase58(),
      };

      const guardianResult = await guardian.validate(guardianParams);
      if (!guardianResult.approved) {
        logger.warn('[MARGINFI] Guardian blocked borrow', {
          reason: guardianResult.blockReason,
          asset: params.asset,
          amount: params.amount,
        });
        return { success: false, error: `Guardian blocked: ${guardianResult.blockReason}` };
      }

      const bank = this.client!.getBankByTokenSymbol(params.asset);
      if (!bank) {
        throw new Error(`Bank not found for asset: ${params.asset}`);
      }

      const signatures = await this.account!.borrow(params.amount, bank.address);
      const signature = Array.isArray(signatures) ? signatures[0] : signatures;

      logger.info('[MARGINFI] Borrow successful', { asset: params.asset, amount: params.amount, signature });
      return { success: true, signature };
    } catch (error: any) {
      logger.error('[MARGINFI] Borrow failed', { error: error.message, asset: params.asset, amount: params.amount });
      return { success: false, error: error.message };
    }
  }

  async repay(params: RepayParams): Promise<LendingResult> {
    try {
      this.ensureInitialized();
      logger.info('[MARGINFI] Repaying', { asset: params.asset, amount: params.amount });

      // ========== GUARDIAN PRE-EXECUTION VALIDATION ==========
      const guardianParams: GuardianTradeParams = {
        inputMint: params.asset,
        outputMint: params.asset,
        amountIn: params.amount,
        amountInUsd: params.amount,
        slippageBps: 50,
        strategy: 'lending',
        protocol: 'marginfi',
        walletAddress: this.keypair.publicKey.toBase58(),
      };

      const guardianResult = await guardian.validate(guardianParams);
      if (!guardianResult.approved) {
        logger.warn('[MARGINFI] Guardian blocked repay', {
          reason: guardianResult.blockReason,
          asset: params.asset,
          amount: params.amount,
        });
        return { success: false, error: `Guardian blocked: ${guardianResult.blockReason}` };
      }
      // ========================================

      const bank = this.client!.getBankByTokenSymbol(params.asset);
      if (!bank) {
        throw new Error(`Bank not found for asset: ${params.asset}`);
      }

      const signature = await this.account!.repay(params.amount, bank.address, params.repayAll ?? false);
      logger.info('[MARGINFI] Repay successful', { asset: params.asset, amount: params.amount, signature });
      return { success: true, signature };
    } catch (error: any) {
      logger.error('[MARGINFI] Repay failed', { error: error.message, asset: params.asset, amount: params.amount });
      return { success: false, error: error.message };
    }
  }

  async getPositions(): Promise<LendingPosition[]> {
    try {
      this.ensureInitialized();
      const positions: LendingPosition[] = [];
      const balances = this.account!.activeBalances;

      for (const balance of balances) {
        const bank = this.client!.getBankByPk(balance.bankPk);
        if (!bank) continue;

        const oraclePrice = this.client!.getOraclePriceByBank(balance.bankPk);
        const price = oraclePrice ? parseFloat(oraclePrice.priceRealtime.price.toString()) : 0;

        const deposited = bank.getAssetQuantity(balance.assetShares).toNumber();
        const borrowed = bank.getLiabilityQuantity(balance.liabilityShares).toNumber();
        const rates = bank.computeInterestRates();
        const supplyAPY = rates.lendingRate.times(100).toNumber();
        const borrowAPY = rates.borrowingRate.times(100).toNumber();

        const depositValue = deposited * price;
        const borrowValue = borrowed * price;
        const totalValue = depositValue + borrowValue;
        const netAPY = totalValue > 0 ? ((depositValue * supplyAPY) - (borrowValue * borrowAPY)) / totalValue : 0;

        positions.push({
          protocol: 'marginfi',
          asset: bank.tokenSymbol || 'UNKNOWN',
          deposited, borrowed, depositedUsd: depositValue, borrowedUsd: borrowValue,
          supplyAPY, borrowAPY, netAPY,
          healthFactor: this.getHealthFactor(),
        });
      }
      return positions;
    } catch (error) {
      logger.error('[MARGINFI] Failed to get positions', { error });
      return [];
    }
  }

  async getAPYs(): Promise<LendingAPY[]> {
    try {
      this.ensureInitialized();
      const apys: LendingAPY[] = [];

      for (const [, bank] of this.client!.banks) {
        const rates = bank.computeInterestRates();
        const utilization = bank.computeUtilizationRate().times(100).toNumber();
        const capacity = bank.computeRemainingCapacity();

        apys.push({
          asset: bank.tokenSymbol || bank.mint.toBase58().slice(0, 8),
          supplyAPY: rates.lendingRate.times(100).toNumber(),
          borrowAPY: rates.borrowingRate.times(100).toNumber(),
          utilization,
          totalDeposits: bank.getTotalAssetQuantity().toNumber(),
          totalBorrows: bank.getTotalLiabilityQuantity().toNumber(),
          depositCapacity: capacity.depositCapacity.toNumber(),
          borrowCapacity: capacity.borrowCapacity.toNumber(),
        });
      }
      return apys;
    } catch (error) {
      logger.error('[MARGINFI] Failed to get APYs', { error });
      return [];
    }
  }

  getHealthFactor(): number {
    if (!this.account) return 0;
    try {
      const health = this.account.computeHealthComponents(0); // 0 = Maintenance
      if (health.liabilities.isZero()) return 999;
      return health.assets.dividedBy(health.liabilities).toNumber();
    } catch { return 0; }
  }

  getMaxBorrow(asset: string): number {
    try {
      this.ensureInitialized();
      const bank = this.client!.getBankByTokenSymbol(asset);
      if (!bank) return 0;
      return this.account!.computeMaxBorrowForBank(bank.address).toNumber();
    } catch { return 0; }
  }

  getMaxWithdraw(asset: string): number {
    try {
      this.ensureInitialized();
      const bank = this.client!.getBankByTokenSymbol(asset);
      if (!bank) return 0;
      return this.account!.computeMaxWithdrawForBank(bank.address).toNumber();
    } catch { return 0; }
  }

  get publicKey(): PublicKey { return this.keypair.publicKey; }
  get accountAddress(): PublicKey | null { return this.account?.address || null; }
  isInitialized(): boolean { return this.initialized; }
}
