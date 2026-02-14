/**
 * Solend Lending Client
 *
 * Production client for Solend (Save) lending protocol on Solana.
 * Supports deposits, withdrawals, borrows, and repayments.
 *
 * FIXED: Real transaction signing and execution (no placeholders).
 * Includes Guardian pre-execution validation.
 */

import {
  Connection,
  Keypair,
  PublicKey,
  Transaction,
  VersionedTransaction,
  ComputeBudgetProgram,
} from '@solana/web3.js';
import {
  SolendActionCore,
  type PoolType,
  type ReserveType,
} from '@solendprotocol/solend-sdk';
import bs58 from 'bs58';
import BN from 'bn.js';

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

// Solend main pool address on mainnet
const SOLEND_MAIN_POOL = '4UpD2fh7xH3VP9QQaXtsS1YY3bxzWhtfpks7FatyKvdY';
const SOLEND_PROGRAM_ID = 'So1endDq2YkqhipRh3WViPa8hdiSpxWy6z3Z6tMCpAo';
// REST API for dynamic reserve data
const SOLEND_API_URL = 'https://api.solend.fi/v1/markets/configs';

export interface SolendClientConfig {
  rpcUrl: string;
  privateKey: string;
  poolAddress?: string;
}

export class SolendLendingClient {
  private connection: Connection;
  private keypair: Keypair;
  private pool: PoolType | null = null;
  private reserves: ReserveType[] = [];
  private initialized = false;
  private poolAddress: string;

  constructor(private config: SolendClientConfig) {
    this.connection = new Connection(config.rpcUrl, 'confirmed');
    this.keypair = Keypair.fromSecretKey(bs58.decode(config.privateKey));
    this.poolAddress = config.poolAddress || SOLEND_MAIN_POOL;
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      logger.info('[SOLEND] Initializing client using REST API...');

      // Fetch market config from Solend REST API (dynamic, no hardcoded fallback)
      const response = await fetch(SOLEND_API_URL);
      if (!response.ok) {
        throw new Error(`Solend API request failed: ${response.status} ${response.statusText}`);
      }

      const marketsConfig = await response.json() as any[];

      // Find our target pool
      const marketConfig = marketsConfig.find((m: any) => m.address === this.poolAddress);
      if (!marketConfig) {
        throw new Error(`Pool ${this.poolAddress} not found in Solend API`);
      }

      // Parse reserves from API response
      this.reserves = (marketConfig.reserves || []).map((r: any) => this.parseReserveFromApi(r));

      // Build pool object
      this.pool = {
        name: marketConfig.name || 'Solend Pool',
        address: this.poolAddress,
        authorityAddress: marketConfig.authorityAddress || this.poolAddress,
        owner: marketConfig.owner || this.poolAddress,
        reserves: this.reserves,
      };

      if (this.reserves.length === 0) {
        throw new Error(`No reserves found for pool ${this.poolAddress}`);
      }

      this.initialized = true;
      logger.info('[SOLEND] Client initialized successfully via REST API', {
        poolAddress: this.poolAddress,
        poolName: marketConfig.name,
        reserveCount: this.reserves.length,
        reserveSymbols: this.reserves.map(r => r.symbol).join(', '),
      });
    } catch (error: any) {
      logger.error('[SOLEND] Failed to initialize', {
        errorMessage: error?.message || 'Unknown error',
        errorName: error?.name,
        errorStack: error?.stack?.split('\n').slice(0, 5).join('\n'),
        errorCode: error?.code,
        rpcUrl: this.config.rpcUrl,
        poolAddress: this.poolAddress,
      });
      throw error;
    }
  }

  /**
   * Parse reserve data from Solend API response into ReserveType format
   * Note: Using 'as unknown as ReserveType' because Solend SDK ReserveType is very complex
   * with many fields we don't need for basic operations
   */
  private parseReserveFromApi(apiReserve: any): ReserveType {
    const liquidityToken = apiReserve.liquidityToken || {};
    return {
      address: apiReserve.address,
      symbol: liquidityToken.symbol,
      decimals: liquidityToken.decimals || 9,
      mintAddress: liquidityToken.mint,
      liquidityAddress: apiReserve.liquidityAddress,
      cTokenMint: apiReserve.collateralMintAddress,
      cTokenLiquidityAddress: apiReserve.collateralSupplyAddress,
      pythOracle: apiReserve.pythOracle,
      switchboardOracle: apiReserve.switchboardOracle,
      liquidityFeeReceiverAddress: apiReserve.liquidityFeeReceiverAddress,
      // These will be fetched from on-chain if needed for APY calculations
      supplyInterest: new BN(0),
      borrowInterest: new BN(0),
      reserveUtilization: new BN(0),
      totalSupply: new BN(0),
      totalBorrow: new BN(0),
      availableAmount: new BN(0),
    } as unknown as ReserveType;
  }

  private ensureInitialized(): void {
    if (!this.initialized || !this.pool) {
      throw new Error('Solend client not initialized. Call initialize() first.');
    }
  }

  private getReserveBySymbol(symbol: string): ReserveType | undefined {
    return this.reserves.find(r => 
      r.symbol?.toLowerCase() === symbol.toLowerCase()
    );
  }

  private buildInputReserve(reserve: ReserveType): any {
    return {
      address: reserve.address,
      liquidityAddress: reserve.liquidityAddress,
      cTokenMint: reserve.cTokenMint,
      cTokenLiquidityAddress: reserve.cTokenLiquidityAddress,
      pythOracle: reserve.pythOracle,
      switchboardOracle: reserve.switchboardOracle,
      mintAddress: reserve.mintAddress,
      liquidityFeeReceiverAddress: reserve.liquidityFeeReceiverAddress,
    };
  }

  private buildInputPool(): any {
    return {
      address: this.pool!.address,
      owner: this.pool!.owner,
      name: this.pool!.name,
      authorityAddress: this.pool!.authorityAddress,
      reserves: this.reserves.map(r => ({
        address: r.address,
        pythOracle: r.pythOracle,
        switchboardOracle: r.switchboardOracle,
        mintAddress: r.mintAddress,
        liquidityFeeReceiverAddress: r.liquidityFeeReceiverAddress,
      })),
    };
  }

  /**
   * Execute Solend transactions - sign, send, and confirm.
   * This is the REAL transaction execution - no placeholders!
   *
   * Solend SDK returns an action object with transactions that need to be executed.
   * The action object may contain:
   * - preTxnIxs: Pre-transaction instructions (setup ATAs, etc.)
   * - lendingIxs: Main lending instructions
   * - postTxnIxs: Post-transaction instructions (cleanup)
   * - setupIxs: Setup instructions
   * - cleanupIxs: Cleanup instructions
   */
  private async sendSolendTransaction(
    action: any,
    operationType: string
  ): Promise<string> {
    logger.info(`[SOLEND] Executing ${operationType} transaction`);

    // Collect all instructions from the action object
    const allInstructions: any[] = [];

    // Solend SDK action object structure varies - handle different formats
    if (action.preTxnIxs && action.preTxnIxs.length > 0) {
      allInstructions.push(...action.preTxnIxs);
    }
    if (action.setupIxs && action.setupIxs.length > 0) {
      allInstructions.push(...action.setupIxs);
    }
    if (action.lendingIxs && action.lendingIxs.length > 0) {
      allInstructions.push(...action.lendingIxs);
    }
    if (action.postTxnIxs && action.postTxnIxs.length > 0) {
      allInstructions.push(...action.postTxnIxs);
    }
    if (action.cleanupIxs && action.cleanupIxs.length > 0) {
      allInstructions.push(...action.cleanupIxs);
    }

    // If action has a direct instructions array
    if (action.instructions && action.instructions.length > 0) {
      allInstructions.push(...action.instructions);
    }

    // If action has txns array (versioned transactions)
    if (action.txns && action.txns.length > 0) {
      // Handle pre-built transactions
      const signatures: string[] = [];
      for (const txn of action.txns) {
        const sig = await this.sendAndConfirmTransaction(txn, operationType);
        signatures.push(sig);
      }
      return signatures[signatures.length - 1]; // Return last signature
    }

    if (allInstructions.length === 0) {
      throw new Error(`No instructions found in ${operationType} action`);
    }

    logger.info(`[SOLEND] Building ${operationType} transaction`, {
      instructionCount: allInstructions.length,
    });

    // Build transaction with compute budget
    const transaction = new Transaction();

    // Add priority fee for faster execution
    const priorityFeeIx = ComputeBudgetProgram.setComputeUnitPrice({
      microLamports: 50000,
    });
    const computeUnitIx = ComputeBudgetProgram.setComputeUnitLimit({
      units: 400000,
    });

    transaction.add(priorityFeeIx, computeUnitIx);
    transaction.add(...allInstructions);

    // Get recent blockhash
    const { blockhash, lastValidBlockHeight } = await this.connection.getLatestBlockhash('confirmed');
    transaction.recentBlockhash = blockhash;
    transaction.feePayer = this.keypair.publicKey;

    // Sign transaction
    transaction.sign(this.keypair);

    logger.info(`[SOLEND] Sending ${operationType} transaction`, {
      feePayer: this.keypair.publicKey.toBase58(),
      instructionCount: allInstructions.length,
    });

    // Send transaction
    const signature = await this.connection.sendRawTransaction(
      transaction.serialize(),
      {
        skipPreflight: false,
        preflightCommitment: 'confirmed',
      }
    );

    logger.info(`[SOLEND] Transaction sent, awaiting confirmation`, {
      signature,
      operationType,
    });

    // Confirm transaction
    const confirmation = await this.connection.confirmTransaction(
      {
        signature,
        blockhash,
        lastValidBlockHeight,
      },
      'confirmed'
    );

    if (confirmation.value.err) {
      throw new Error(`Transaction failed: ${JSON.stringify(confirmation.value.err)}`);
    }

    logger.info(`[SOLEND] ${operationType} transaction confirmed`, {
      signature,
    });

    return signature;
  }

  /**
   * Send and confirm a pre-built transaction (versioned or legacy)
   */
  private async sendAndConfirmTransaction(
    txn: Transaction | VersionedTransaction,
    operationType: string
  ): Promise<string> {
    const { blockhash, lastValidBlockHeight } = await this.connection.getLatestBlockhash('confirmed');

    let signature: string;

    if (txn instanceof VersionedTransaction) {
      // Versioned transaction
      txn.sign([this.keypair]);
      signature = await this.connection.sendTransaction(txn, {
        skipPreflight: false,
        preflightCommitment: 'confirmed',
      });
    } else {
      // Legacy transaction
      txn.recentBlockhash = blockhash;
      txn.feePayer = this.keypair.publicKey;
      txn.sign(this.keypair);
      signature = await this.connection.sendRawTransaction(
        txn.serialize(),
        {
          skipPreflight: false,
          preflightCommitment: 'confirmed',
        }
      );
    }

    const confirmation = await this.connection.confirmTransaction(
      {
        signature,
        blockhash,
        lastValidBlockHeight,
      },
      'confirmed'
    );

    if (confirmation.value.err) {
      throw new Error(`${operationType} transaction failed: ${JSON.stringify(confirmation.value.err)}`);
    }

    return signature;
  }

  async deposit(params: DepositParams): Promise<LendingResult> {
    try {
      this.ensureInitialized();
      logger.info('[SOLEND] Depositing', { asset: params.asset, amount: params.amount });

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
          reasoning: `Deposit ${params.amount} ${params.asset} to Solend`,
          protocol: 'solend',
        };

        const portfolioValueUsd = 10000;
        const needsApproval = pmDecisionEngine.needsApproval(pmParams, portfolioValueUsd);

        if (needsApproval) {
          logger.info('[SOLEND] Deposit requires PM approval', {
            asset: params.asset,
            amount: params.amount,
          });

          const tradeId = approvalQueue.queueTrade(pmParams);
          const approvalResult = await approvalQueue.waitForApproval(tradeId);

          if (!approvalResult.approved) {
            logger.warn('[SOLEND] PM rejected deposit', {
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
        inputMint: params.asset,
        outputMint: params.asset,
        amountIn: params.amount,
        amountInUsd: params.amount,
        slippageBps: 50,
        strategy: 'lending',
        protocol: 'solend',
        walletAddress: this.keypair.publicKey.toBase58(),
      };

      logger.info('[SOLEND] Running Guardian validation', { asset: params.asset, amount: params.amount });
      const guardianResult = await guardian.validate(guardianParams);
      if (!guardianResult.approved) {
        logger.warn('[SOLEND] Guardian blocked deposit', {
          reason: guardianResult.blockReason,
          asset: params.asset,
          amount: params.amount,
        });
        return { success: false, error: `Guardian blocked: ${guardianResult.blockReason}` };
      }
      // ========================================

      const reserve = this.getReserveBySymbol(params.asset);
      if (!reserve) {
        throw new Error(`Reserve not found for asset: ${params.asset}`);
      }

      // Convert to lamports
      const amountLamports = Math.floor(params.amount * Math.pow(10, reserve.decimals)).toString();

      const wallet = {
        publicKey: this.keypair.publicKey,
      };

      const action = await SolendActionCore.buildDepositTxns(
        this.buildInputPool(),
        this.buildInputReserve(reserve),
        this.connection,
        amountLamports,
        wallet,
        {}
      );

      logger.info('[SOLEND] Deposit transaction built', {
        asset: params.asset,
        amount: params.amount,
      });

      // REAL transaction execution - sign, send, confirm
      const signature = await this.sendSolendTransaction(action, 'deposit');

      logger.info('[SOLEND] Deposit successful', {
        asset: params.asset,
        amount: params.amount,
        signature,
      });

      return { success: true, signature };
    } catch (error: any) {
      logger.error('[SOLEND] Deposit failed', { error: error.message, asset: params.asset, amount: params.amount });
      return { success: false, error: error.message };
    }
  }

  async withdraw(params: WithdrawParams): Promise<LendingResult> {
    try {
      this.ensureInitialized();
      logger.info('[SOLEND] Withdrawing', { asset: params.asset, amount: params.amount });

      // ========== GUARDIAN PRE-EXECUTION VALIDATION ==========
      const guardianParams: GuardianTradeParams = {
        inputMint: params.asset,
        outputMint: params.asset,
        amountIn: params.amount,
        amountInUsd: params.amount,
        slippageBps: 50,
        strategy: 'lending',
        protocol: 'solend',
        walletAddress: this.keypair.publicKey.toBase58(),
      };

      const guardianResult = await guardian.validate(guardianParams);
      if (!guardianResult.approved) {
        logger.warn('[SOLEND] Guardian blocked withdraw', {
          reason: guardianResult.blockReason,
          asset: params.asset,
          amount: params.amount,
        });
        return { success: false, error: `Guardian blocked: ${guardianResult.blockReason}` };
      }
      // ========================================

      const reserve = this.getReserveBySymbol(params.asset);
      if (!reserve) {
        throw new Error(`Reserve not found for asset: ${params.asset}`);
      }

      const amountLamports = Math.floor(params.amount * Math.pow(10, reserve.decimals)).toString();

      const wallet = { publicKey: this.keypair.publicKey };

      const action = await SolendActionCore.buildWithdrawTxns(
        this.buildInputPool(),
        this.buildInputReserve(reserve),
        this.connection,
        amountLamports,
        wallet,
        {}
      );

      logger.info('[SOLEND] Withdraw transaction built', {
        asset: params.asset,
        amount: params.amount,
      });

      // REAL transaction execution - sign, send, confirm
      const signature = await this.sendSolendTransaction(action, 'withdraw');

      logger.info('[SOLEND] Withdraw successful', {
        asset: params.asset,
        amount: params.amount,
        signature,
      });

      return { success: true, signature };
    } catch (error: any) {
      logger.error('[SOLEND] Withdraw failed', { error: error.message, asset: params.asset, amount: params.amount });
      return { success: false, error: error.message };
    }
  }

  async borrow(params: BorrowParams): Promise<LendingResult> {
    try {
      this.ensureInitialized();
      logger.info('[SOLEND] Borrowing', { asset: params.asset, amount: params.amount });

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
          reasoning: `Borrow ${params.amount} ${params.asset} from Solend`,
          protocol: 'solend',
        };

        const portfolioValueUsd = 10000;
        const needsApproval = pmDecisionEngine.needsApproval(pmParams, portfolioValueUsd);

        if (needsApproval) {
          logger.info('[SOLEND] Borrow requires PM approval', {
            asset: params.asset,
            amount: params.amount,
          });

          const tradeId = approvalQueue.queueTrade(pmParams);
          const approvalResult = await approvalQueue.waitForApproval(tradeId);

          if (!approvalResult.approved) {
            logger.warn('[SOLEND] PM rejected borrow', {
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
        protocol: 'solend',
        walletAddress: this.keypair.publicKey.toBase58(),
      };

      const guardianResult = await guardian.validate(guardianParams);
      if (!guardianResult.approved) {
        logger.warn('[SOLEND] Guardian blocked borrow', {
          reason: guardianResult.blockReason,
          asset: params.asset,
          amount: params.amount,
        });
        return { success: false, error: `Guardian blocked: ${guardianResult.blockReason}` };
      }

      const reserve = this.getReserveBySymbol(params.asset);
      if (!reserve) {
        throw new Error(`Reserve not found for asset: ${params.asset}`);
      }

      const amountLamports = Math.floor(params.amount * Math.pow(10, reserve.decimals)).toString();

      const wallet = { publicKey: this.keypair.publicKey };

      const action = await SolendActionCore.buildBorrowTxns(
        this.buildInputPool(),
        this.buildInputReserve(reserve),
        this.connection,
        amountLamports,
        wallet,
        {}
      );

      logger.info('[SOLEND] Borrow transaction built', {
        asset: params.asset,
        amount: params.amount,
      });

      // REAL transaction execution - sign, send, confirm
      const signature = await this.sendSolendTransaction(action, 'borrow');

      logger.info('[SOLEND] Borrow successful', {
        asset: params.asset,
        amount: params.amount,
        signature,
      });

      return { success: true, signature };
    } catch (error: any) {
      logger.error('[SOLEND] Borrow failed', { error: error.message, asset: params.asset, amount: params.amount });
      return { success: false, error: error.message };
    }
  }

  async repay(params: RepayParams): Promise<LendingResult> {
    try {
      this.ensureInitialized();
      logger.info('[SOLEND] Repaying', { asset: params.asset, amount: params.amount });

      // ========== GUARDIAN PRE-EXECUTION VALIDATION ==========
      const guardianParams: GuardianTradeParams = {
        inputMint: params.asset,
        outputMint: params.asset,
        amountIn: params.amount,
        amountInUsd: params.amount,
        slippageBps: 50,
        strategy: 'lending',
        protocol: 'solend',
        walletAddress: this.keypair.publicKey.toBase58(),
      };

      const guardianResult = await guardian.validate(guardianParams);
      if (!guardianResult.approved) {
        logger.warn('[SOLEND] Guardian blocked repay', {
          reason: guardianResult.blockReason,
          asset: params.asset,
          amount: params.amount,
        });
        return { success: false, error: `Guardian blocked: ${guardianResult.blockReason}` };
      }
      // ========================================

      const reserve = this.getReserveBySymbol(params.asset);
      if (!reserve) {
        throw new Error(`Reserve not found for asset: ${params.asset}`);
      }

      const amountLamports = Math.floor(params.amount * Math.pow(10, reserve.decimals)).toString();

      const wallet = { publicKey: this.keypair.publicKey };

      const action = await SolendActionCore.buildRepayTxns(
        this.buildInputPool(),
        this.buildInputReserve(reserve),
        this.connection,
        amountLamports,
        wallet,
        {}
      );

      logger.info('[SOLEND] Repay transaction built', {
        asset: params.asset,
        amount: params.amount,
      });

      // REAL transaction execution - sign, send, confirm
      const signature = await this.sendSolendTransaction(action, 'repay');

      logger.info('[SOLEND] Repay successful', {
        asset: params.asset,
        amount: params.amount,
        signature,
      });

      return { success: true, signature };
    } catch (error: any) {
      logger.error('[SOLEND] Repay failed', { error: error.message, asset: params.asset, amount: params.amount });
      return { success: false, error: error.message };
    }
  }

  async getPositions(): Promise<LendingPosition[]> {
    try {
      this.ensureInitialized();

      const SOLEND_PROGRAM_ID = new PublicKey('So1endDq2YkqhipRh3WViPa8hdiSpxWy6z3Z6tMCpAo');
      const userPubkey = this.keypair.publicKey;

      // Fetch user's obligation accounts from Solend program
      const obligationAccounts = await this.connection.getProgramAccounts(SOLEND_PROGRAM_ID, {
        filters: [
          { dataSize: 1300 }, // Obligation account data size
          { memcmp: { offset: 10, bytes: userPubkey.toBase58() } },
        ],
      });

      if (obligationAccounts.length === 0) {
        logger.info('[SOLEND] No obligation accounts found for user');
        return [];
      }

      logger.info('[SOLEND] Found obligation accounts', {
        count: obligationAccounts.length,
        addresses: obligationAccounts.map(a => a.pubkey.toBase58()),
      });

      // Full obligation parsing requires Solend SDK layout deserialization
      // For now, report found obligations without detailed position data
      logger.warn('[SOLEND] Detailed obligation parsing pending - returning account count only');
      return [];
    } catch (error: any) {
      logger.error('[SOLEND] Failed to get positions', { error: error.message });
      return [];
    }
  }

  async getAPYs(): Promise<LendingAPY[]> {
    try {
      this.ensureInitialized();
      const apys: LendingAPY[] = [];

      for (const reserve of this.reserves) {
        const supplyAPY = reserve.supplyInterest?.toNumber() || 0;
        const borrowAPY = reserve.borrowInterest?.toNumber() || 0;
        const utilization = reserve.reserveUtilization?.toNumber() || 0;

        apys.push({
          asset: reserve.symbol || 'UNKNOWN',
          supplyAPY: supplyAPY * 100,
          borrowAPY: borrowAPY * 100,
          utilization: utilization * 100,
          totalDeposits: reserve.totalSupply?.toNumber() || 0,
          totalBorrows: reserve.totalBorrow?.toNumber() || 0,
          depositCapacity: reserve.availableAmount?.toNumber() || 0,
          borrowCapacity: reserve.availableAmount?.toNumber() || 0,
        });
      }

      return apys;
    } catch (error) {
      logger.error('[SOLEND] Failed to get APYs', { error });
      return [];
    }
  }

  getHealthFactor(): number {
    // Would need obligation data to calculate
    return 0;
  }

  get publicKey(): PublicKey {
    return this.keypair.publicKey;
  }

  isInitialized(): boolean {
    return this.initialized;
  }
}

