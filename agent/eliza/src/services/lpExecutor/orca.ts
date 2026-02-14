/**
 * Orca Whirlpools Executor
 * Uses @orca-so/whirlpools-sdk v0.17.x
 */

import { Connection, Keypair, PublicKey, Transaction, VersionedTransaction } from '@solana/web3.js';
import { logger } from '../logger.js';
import { guardian } from '../guardian/index.js';
import type { GuardianTradeParams } from '../guardian/types.js';
import { pmDecisionEngine, approvalQueue } from '../pm/index.js';
import type { QueueTradeParams } from '../pm/types.js';
import {
  WhirlpoolContext,
  buildWhirlpoolClient,
  PriceMath,
  TickUtil,
  PoolUtil,
  increaseLiquidityQuoteByInputToken,
  IGNORE_CACHE,
  NO_TOKEN_EXTENSION_CONTEXT,
} from '@orca-so/whirlpools-sdk';
import { Percentage, type Wallet } from '@orca-so/common-sdk';
import type {
  SupportedDex,
  LPPoolInfo,
  DepositParams,
  DepositResult,
  WithdrawParams,
  WithdrawResult,
  PositionInfo,
  PriceImpactResult,
  ExecutorConfig,
  IDexExecutor,
} from './types.js';
import { Decimal } from 'decimal.js';

export class OrcaExecutor implements IDexExecutor {
  readonly dex: SupportedDex = 'orca';
  private connection: Connection;
  private config: ExecutorConfig;

  constructor(connection: Connection, config: ExecutorConfig) {
    this.connection = connection;
    this.config = config;
  }

  isSupported(pool: LPPoolInfo): boolean {
    const dex = pool.dex.toLowerCase();
    return dex === 'orca' || dex.includes('whirlpool');
  }

  private createWalletAdapter(keypair: Keypair): Wallet {
    return {
      publicKey: keypair.publicKey,
      signTransaction: async <T extends Transaction | VersionedTransaction>(tx: T): Promise<T> => {
        if (tx instanceof VersionedTransaction) {
          tx.sign([keypair]);
        } else {
          tx.partialSign(keypair);
        }
        return tx;
      },
      signAllTransactions: async <T extends Transaction | VersionedTransaction>(txs: T[]): Promise<T[]> => {
        for (const tx of txs) {
          if (tx instanceof VersionedTransaction) {
            tx.sign([keypair]);
          } else {
            tx.partialSign(keypair);
          }
        }
        return txs;
      },
    };
  }

  private createReadOnlyWallet(pubkey: PublicKey): Wallet {
    return {
      publicKey: pubkey,
      signTransaction: async <T extends Transaction | VersionedTransaction>(tx: T): Promise<T> => tx,
      signAllTransactions: async <T extends Transaction | VersionedTransaction>(txs: T[]): Promise<T[]> => txs,
    };
  }

  async deposit(params: DepositParams): Promise<DepositResult> {
    const { pool, amountUsd, slippageBps = 50, wallet } = params;

    try {
      logger.info('[Orca] Starting deposit', { pool: pool.name, amountUsd, slippageBps });

      // PM Approval check (before Guardian)
      if (pmDecisionEngine.isEnabled()) {
        const pmParams: QueueTradeParams = {
          strategy: 'lp',
          action: 'DEPOSIT',
          asset: pool.name,
          assetMint: pool.address,
          amount: amountUsd,
          amountUsd,
          confidence: 0.7,
          risk: {
            volatility: 0,
            liquidityScore: 70,
            riskScore: 30,
          },
          reasoning: `Deposit to ${pool.name} Whirlpool`,
          protocol: 'orca',
        };

        const portfolioValueUsd = params.portfolioValueUsd || 10000;
        const needsApproval = pmDecisionEngine.needsApproval(pmParams, portfolioValueUsd);

        if (needsApproval) {
          logger.info('[Orca] Deposit requires PM approval', { pool: pool.name, amountUsd });

          const tradeId = approvalQueue.queueTrade(pmParams);
          const approvalResult = await approvalQueue.waitForApproval(tradeId);

          if (!approvalResult.approved) {
            logger.warn('[Orca] PM rejected deposit', {
              tradeId,
              status: approvalResult.status,
              reason: approvalResult.rejectionReason,
            });
            return {
              success: false,
              error: `PM rejected: ${approvalResult.rejectionReason || approvalResult.status}`,
            };
          }
        }
      }

      // Guardian pre-execution validation
      const guardianParams: GuardianTradeParams = {
        inputMint: pool.token0.mint,
        outputMint: pool.token1.mint,
        amountIn: amountUsd,
        amountInUsd: amountUsd,
        slippageBps,
        strategy: 'lp',
        protocol: 'orca',
        walletAddress: wallet.publicKey.toBase58(),
      };

      const guardianResult = await guardian.validate(guardianParams);
      if (!guardianResult.approved) {
        logger.warn('[Orca] Guardian blocked transaction', {
          reason: guardianResult.blockReason,
          pool: pool.name,
          amountUsd,
        });
        return {
          success: false,
          error: `Guardian blocked: ${guardianResult.blockReason}`,
        };
      }

      // Validate pool address format
      if (!pool.address || pool.address.length < 32) {
        throw new Error(`Invalid pool address: ${pool.address}`);
      }

      // Create Whirlpool context and client
      const ctx = WhirlpoolContext.from(this.connection, this.createWalletAdapter(wallet));
      const client = buildWhirlpoolClient(ctx);

      // Fetch whirlpool
      let whirlpoolPubkey: PublicKey;
      try {
        logger.info('[Orca] Parsing pool address', { address: pool.address, length: pool.address?.length });
        whirlpoolPubkey = new PublicKey(pool.address);
        logger.info('[Orca] Pool address parsed successfully', { pubkey: whirlpoolPubkey.toBase58() });
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Unknown error';
        logger.error('[Orca] Failed to parse pool address', { address: pool.address, error: errorMsg });
        throw new Error(`Invalid Solana address format: ${pool.address} - ${errorMsg}`);
      }

      const whirlpool = await client.getPool(whirlpoolPubkey, IGNORE_CACHE);
      const whirlpoolData = whirlpool.getData();

      // Calculate full range tick indexes
      const tickSpacing = whirlpoolData.tickSpacing;
      const tickLowerIndex = TickUtil.getInitializableTickIndex(-443636, tickSpacing);
      const tickUpperIndex = TickUtil.getInitializableTickIndex(443636, tickSpacing);

      // Calculate token amounts
      const tokenAPrice = await this.getTokenPrice(pool.token0.mint);
      const amountPerSide = amountUsd / 2;
      const tokenAAmountDecimal = new Decimal(amountPerSide / tokenAPrice);

      // Get quote for adding liquidity (use NO_TOKEN_EXTENSION_CONTEXT for standard tokens)
      const quote = increaseLiquidityQuoteByInputToken(
        whirlpoolData.tokenMintA,
        tokenAAmountDecimal,
        tickLowerIndex,
        tickUpperIndex,
        Percentage.fromFraction(slippageBps, 10000),
        whirlpool,
        NO_TOKEN_EXTENSION_CONTEXT
      );

      // Open position and add liquidity
      const { positionMint, tx: openPositionTx } = await whirlpool.openPositionWithMetadata(
        tickLowerIndex,
        tickUpperIndex,
        quote
      );

      // Build and sign transaction
      openPositionTx.addSigner(wallet);
      const signature = await openPositionTx.buildAndExecute();

      // Wait for confirmation and check if transaction succeeded
      logger.info('[Orca] Deposit transaction sent, waiting for confirmation...', { signature });
      const txResult = await this.connection.confirmTransaction(signature, 'confirmed');

      if (txResult.value.err) {
        const errorDetails = JSON.stringify(txResult.value.err);
        logger.error('[Orca] Deposit transaction FAILED on-chain', {
          signature,
          error: errorDetails
        });
        throw new Error(`Transaction failed on-chain: ${errorDetails}`);
      }

      logger.info('[Orca] Deposit successful and confirmed', { signature, positionMint: positionMint.toBase58() });

      return {
        success: true,
        txSignature: signature,
        positionId: positionMint.toBase58(),
        lpTokenMint: positionMint.toBase58(),
        lpTokenBalance: 1,
        amountToken0: quote.tokenEstA.toNumber() / 10 ** pool.token0.decimals,
        amountToken1: quote.tokenEstB.toNumber() / 10 ** pool.token1.decimals,
        priceImpactPct: 0,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      logger.error('[Orca] Deposit failed', { error: errorMessage });
      throw error;
    }
  }

  async withdraw(params: WithdrawParams): Promise<WithdrawResult> {
    const { positionId, percentage = 100, slippageBps = 50, wallet } = params;

    try {
      logger.info('[Orca] Starting withdrawal', { positionId, percentage });

      const ctx = WhirlpoolContext.from(this.connection, this.createWalletAdapter(wallet));
      const client = buildWhirlpoolClient(ctx);

      // Get position and whirlpool
      const positionPubkey = new PublicKey(positionId);
      const position = await client.getPosition(positionPubkey, IGNORE_CACHE);
      const positionData = position.getData();
      const whirlpool = await client.getPool(positionData.whirlpool, IGNORE_CACHE);

      // Close position (handles decrease liquidity, collect fees, and close)
      const closeTxs = await whirlpool.closePosition(
        positionPubkey,
        Percentage.fromFraction(slippageBps, 10000),
        wallet.publicKey,
        wallet.publicKey
      );

      // Execute all transactions
      let lastSignature = '';
      for (const tx of closeTxs) {
        tx.addSigner(wallet);
        lastSignature = await tx.buildAndExecute();
      }

      logger.info('[Orca] Withdrawal successful', { signature: lastSignature });

      return { success: true, txSignature: lastSignature };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      logger.error('[Orca] Withdrawal failed', { error: errorMessage });
      return { success: false, error: errorMessage };
    }
  }

  async getPosition(positionId: string, wallet: PublicKey): Promise<PositionInfo | null> {
    try {
      const ctx = WhirlpoolContext.from(this.connection, this.createReadOnlyWallet(wallet));
      const client = buildWhirlpoolClient(ctx);

      const positionPubkey = new PublicKey(positionId);
      const position = await client.getPosition(positionPubkey, IGNORE_CACHE);
      const positionData = position.getData();
      const whirlpool = await client.getPool(positionData.whirlpool, IGNORE_CACHE);
      const whirlpoolData = whirlpool.getData();

      // Get token amounts in position
      const amounts = PoolUtil.getTokenAmountsFromLiquidity(
        positionData.liquidity,
        whirlpoolData.sqrtPrice,
        PriceMath.tickIndexToSqrtPriceX64(positionData.tickLowerIndex),
        PriceMath.tickIndexToSqrtPriceX64(positionData.tickUpperIndex),
        true
      );

      const tokenAMint = whirlpoolData.tokenMintA.toBase58();
      const tokenBMint = whirlpoolData.tokenMintB.toBase58();
      const priceA = await this.getTokenPrice(tokenAMint);
      const priceB = await this.getTokenPrice(tokenBMint);

      const token0Decimals = whirlpool.getTokenAInfo().decimals;
      const token1Decimals = whirlpool.getTokenBInfo().decimals;

      const token0Amount = amounts.tokenA.toNumber() / 10 ** token0Decimals;
      const token1Amount = amounts.tokenB.toNumber() / 10 ** token1Decimals;

      return {
        positionId,
        pool: {
          address: positionData.whirlpool.toBase58(),
          name: 'Orca Whirlpool Position',
          dex: 'orca',
          token0: { symbol: 'TokenA', mint: tokenAMint, decimals: token0Decimals },
          token1: { symbol: 'TokenB', mint: tokenBMint, decimals: token1Decimals },
          fee: whirlpoolData.tickSpacing,
          tvlUsd: 0,
          apy: 0,
        },
        token0Amount,
        token1Amount,
        valueUsd: token0Amount * priceA + token1Amount * priceB,
        feesEarnedUsd: positionData.feeOwedA.toNumber() + positionData.feeOwedB.toNumber(),
        unrealizedPnlUsd: 0,
        entryTime: Date.now(),
        priceLower: PriceMath.tickIndexToPrice(positionData.tickLowerIndex, token0Decimals, token1Decimals).toNumber(),
        priceUpper: PriceMath.tickIndexToPrice(positionData.tickUpperIndex, token0Decimals, token1Decimals).toNumber(),
        inRange: whirlpoolData.tickCurrentIndex >= positionData.tickLowerIndex &&
                 whirlpoolData.tickCurrentIndex <= positionData.tickUpperIndex,
      };
    } catch (error) {
      logger.error('[Orca] Failed to fetch position', { error });
      return null;
    }
  }

  async calculatePriceImpact(pool: LPPoolInfo, amountUsd: number): Promise<PriceImpactResult> {
    const estimatedImpact = amountUsd / (pool.tvlUsd || 1_000_000) * 100;

    return {
      impactPct: Math.min(estimatedImpact, 10),
      expectedOutput: amountUsd * 0.997,
      minimumOutput: amountUsd * 0.99,
      isAcceptable: estimatedImpact < this.config.maxPriceImpactPct,
    };
  }

  private async getTokenPrice(mint: string): Promise<number> {
    try {
      const response = await fetch(`https://api.jup.ag/price/v2?ids=${mint}`);
      const data = await response.json() as { data?: { [key: string]: { price?: number } } };
      return data.data?.[mint]?.price || 0;
    } catch {
      return 0;
    }
  }
}

