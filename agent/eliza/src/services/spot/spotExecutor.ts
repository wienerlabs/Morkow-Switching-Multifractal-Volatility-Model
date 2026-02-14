/**
 * Spot Trading Executor
 *
 * Executes spot token swaps using Jupiter Ultra API.
 * Handles buy/sell operations with slippage protection.
 */

import { Connection, Keypair, VersionedTransaction, PublicKey } from '@solana/web3.js';
import { logger } from '../logger.js';
import { guardian } from '../guardian/index.js';
import type { GuardianTradeParams } from '../guardian/types.js';
import { pmDecisionEngine, approvalQueue } from '../pm/index.js';
import type { QueueTradeParams } from '../pm/types.js';

// Jupiter Ultra API key from environment (required)
const JUPITER_API_KEY = process.env.JUPITER_API_KEY || '';

export interface SpotExecutorConfig {
  rpcUrl: string;
  wallet: Keypair;
  slippageBps?: number; // Default 50 (0.5%)
}

export interface SpotBuyParams {
  inputMint: string;   // SOL or USDC token address
  outputMint: string;  // Token to buy address
  amountUsd: number;   // USD amount to spend
}

export interface SpotSellParams {
  inputMint: string;   // Token to sell address
  outputMint: string;  // USDC token address
  amountTokens: number; // Number of tokens to sell
}

export interface SpotExecutionResult {
  success: boolean;
  signature?: string;
  outputAmount?: number;
  usdReceived?: number;
  error?: string;
}

// Jupiter Ultra API response
interface JupiterUltraOrder {
  requestId: string;
  inputMint: string;
  inAmount: string;
  outputMint: string;
  outAmount: string;
  otherAmountThreshold: string;
  swapMode: string;
  slippageBps: number;
  priceImpactPct: string;
  routePlan: any[];
  transaction: string | null;
  inUsdValue: number;
  outUsdValue: number;
  error?: string;
}

/**
 * SpotExecutor - Jupiter Ultra API based spot trading
 */
export class SpotExecutor {
  private connection: Connection;
  private wallet: Keypair;
  private slippageBps: number;
  private decimalsCache: Map<string, number> = new Map();

  constructor(private config: SpotExecutorConfig) {
    if (!JUPITER_API_KEY) {
      throw new Error('JUPITER_API_KEY environment variable is required for spot trading');
    }
    this.connection = new Connection(config.rpcUrl, 'confirmed');
    this.wallet = config.wallet;
    this.slippageBps = config.slippageBps || 50; // 0.5% default (Ultra API recommends lower)
  }

  /**
   * Get Jupiter Ultra API order (quote + transaction)
   */
  private async getJupiterUltraOrder(
    inputMint: string,
    outputMint: string,
    amount: number
  ): Promise<JupiterUltraOrder> {
    const taker = this.wallet.publicKey.toBase58();
    const url = `https://api.jup.ag/ultra/v1/order?inputMint=${inputMint}&outputMint=${outputMint}&amount=${amount}&slippageBps=${this.slippageBps}&taker=${taker}`;

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'x-api-key': JUPITER_API_KEY,
      }
    });

    const order = await response.json() as JupiterUltraOrder;

    if (order.error) {
      throw new Error(`Jupiter Ultra order error: ${order.error}`);
    }

    logger.info('[SpotExecutor] Jupiter Ultra order received', {
      requestId: order.requestId,
      inAmount: order.inAmount,
      outAmount: order.outAmount,
      priceImpact: order.priceImpactPct,
      hasTransaction: !!order.transaction,
    });

    return order;
  }

  /**
   * Execute Jupiter Ultra swap transaction
   */
  private async executeJupiterUltraSwap(order: JupiterUltraOrder): Promise<string> {
    if (!order.transaction) {
      throw new Error('No transaction in Jupiter Ultra order');
    }

    // Deserialize and sign transaction
    const txBuffer = Buffer.from(order.transaction, 'base64');
    const transaction = VersionedTransaction.deserialize(txBuffer);
    transaction.sign([this.wallet]);

    logger.info('[SpotExecutor] Sending transaction...', { requestId: order.requestId });

    // Send transaction
    const txid = await this.connection.sendTransaction(transaction, {
      skipPreflight: false,
      maxRetries: 3,
    });

    // Confirm transaction
    const latestBlockhash = await this.connection.getLatestBlockhash();
    await this.connection.confirmTransaction({
      signature: txid,
      blockhash: latestBlockhash.blockhash,
      lastValidBlockHeight: latestBlockhash.lastValidBlockHeight
    }, 'confirmed');

    logger.info('[SpotExecutor] Transaction confirmed', { signature: txid, requestId: order.requestId });
    return txid;
  }

  /**
   * Get real-time price from Jupiter Ultra API
   * Uses a small quote to calculate the actual execution price
   * @param tokenMint - Token mint address to get price for
   * @returns Price in USD or null if failed
   */
  async getJupiterPrice(tokenMint: string): Promise<number | null> {
    const SOL_MINT = 'So11111111111111111111111111111111111111112';
    const USDC_MINT = 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v';

    try {
      // Get token decimals
      const tokenDecimals = await this.getTokenDecimals(tokenMint);

      // Request a quote for 1 token worth of the asset
      // Use 1 full token (adjusted for decimals) as the input amount
      const oneToken = Math.pow(10, tokenDecimals); // e.g., 1,000,000 for 6 decimals

      const taker = this.wallet.publicKey.toBase58();
      const url = `https://api.jup.ag/ultra/v1/order?inputMint=${tokenMint}&outputMint=${USDC_MINT}&amount=${oneToken}&slippageBps=100&taker=${taker}`;

      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'x-api-key': JUPITER_API_KEY,
        }
      });

      const order = await response.json() as JupiterUltraOrder;

      if (order.error) {
        logger.warn('[SpotExecutor] Jupiter price quote error', { tokenMint: tokenMint.slice(0, 8), error: order.error });
        return null;
      }

      // outAmount is in USDC lamports (6 decimals)
      // This represents how much USDC we'd get for 1 token
      const usdcAmount = parseInt(order.outAmount) / 1_000_000;

      logger.info('[SpotExecutor] Jupiter price fetched', {
        tokenMint: tokenMint.slice(0, 8),
        priceUsd: usdcAmount.toFixed(6),
        inAmount: order.inAmount,
        outAmount: order.outAmount,
      });

      return usdcAmount;
    } catch (error) {
      logger.error('[SpotExecutor] Failed to get Jupiter price', {
        tokenMint: tokenMint.slice(0, 8),
        error: error instanceof Error ? error.message : String(error),
      });
      return null;
    }
  }

  /**
   * Fetch token decimals from mint account
   */
  private async getTokenDecimals(mintAddress: string): Promise<number> {
    // Check cache first
    if (this.decimalsCache.has(mintAddress)) {
      return this.decimalsCache.get(mintAddress)!;
    }

    try {
      const mintPubkey = new PublicKey(mintAddress);
      const tokenSupply = await this.connection.getTokenSupply(mintPubkey);
      const decimals = tokenSupply.value.decimals;

      // Cache for future use
      this.decimalsCache.set(mintAddress, decimals);

      logger.info('[SpotExecutor] Fetched token decimals', {
        mint: mintAddress,
        decimals,
      });

      return decimals;
    } catch (error: any) {
      logger.error('[SpotExecutor] Failed to fetch token decimals, using default 9', {
        mint: mintAddress,
        error: error.message,
      });
      // Fallback to 9 decimals (most common on Solana)
      return 9;
    }
  }

  /**
   * Convert USD amount to input token lamports
   * SOL has 9 decimals, USDC has 6 decimals
   */
  private async usdToInputLamports(amountUsd: number, inputMint: string): Promise<number> {
    const SOL_MINT = 'So11111111111111111111111111111111111111112';

    if (inputMint === SOL_MINT) {
      // For SOL: fetch current SOL price and convert USD to lamports
      try {
        const response = await fetch('https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd');
        const data = await response.json() as any;
        const solPrice = data?.solana?.usd || 124; // Fallback to ~$124
        const solAmount = amountUsd / solPrice;
        return Math.floor(solAmount * 1_000_000_000); // SOL has 9 decimals
      } catch (e) {
        // Fallback: assume SOL is ~$124
        const solAmount = amountUsd / 124;
        return Math.floor(solAmount * 1_000_000_000);
      }
    } else {
      // For USDC/USDT: 6 decimals, 1:1 with USD
      return Math.floor(amountUsd * 1_000_000);
    }
  }

  /**
   * Buy tokens with SOL or USDC
   */
  async buy(params: SpotBuyParams, analysisContext?: { confidence: number; reasoning: string; portfolioValueUsd: number }): Promise<SpotExecutionResult> {
    try {
      logger.info('[SpotExecutor] Executing buy', {
        inputMint: params.inputMint,
        outputMint: params.outputMint,
        amountUsd: params.amountUsd,
      });

      // PM Approval check (before Guardian)
      if (pmDecisionEngine.isEnabled() && analysisContext) {
        const pmParams: QueueTradeParams = {
          strategy: 'spot',
          action: 'BUY',
          asset: params.outputMint,
          assetMint: params.outputMint,
          amount: params.amountUsd,
          amountUsd: params.amountUsd,
          confidence: analysisContext.confidence,
          risk: {
            volatility: 0, // Will be populated by caller
            liquidityScore: 50,
            riskScore: 30,
          },
          reasoning: analysisContext.reasoning,
          protocol: 'jupiter',
        };

        const needsApproval = pmDecisionEngine.needsApproval(pmParams, analysisContext.portfolioValueUsd);

        if (needsApproval) {
          logger.info('[SpotExecutor] Trade requires PM approval', {
            amountUsd: params.amountUsd,
            outputMint: params.outputMint,
          });

          const tradeId = approvalQueue.queueTrade(pmParams);
          const approvalResult = await approvalQueue.waitForApproval(tradeId);

          if (!approvalResult.approved) {
            logger.warn('[SpotExecutor] PM rejected buy transaction', {
              tradeId,
              status: approvalResult.status,
              reason: approvalResult.rejectionReason,
            });
            return {
              success: false,
              error: `PM rejected: ${approvalResult.rejectionReason || approvalResult.status}`,
            };
          }

          logger.info('[SpotExecutor] PM approved buy transaction', {
            tradeId,
            approver: approvalResult.approver,
            waitTimeMs: approvalResult.waitTimeMs,
          });
        }
      }

      // Guardian pre-execution validation
      const guardianParams: GuardianTradeParams = {
        inputMint: params.inputMint,
        outputMint: params.outputMint,
        amountIn: params.amountUsd,
        amountInUsd: params.amountUsd,
        slippageBps: this.slippageBps,
        strategy: 'spot',
        protocol: 'jupiter',
        walletAddress: this.wallet.publicKey.toBase58(),
      };

      const guardianResult = await guardian.validate(guardianParams);
      if (!guardianResult.approved) {
        logger.warn('[SpotExecutor] Guardian blocked buy transaction', {
          reason: guardianResult.blockReason,
          outputMint: params.outputMint,
          amountUsd: params.amountUsd,
        });
        return {
          success: false,
          error: `Guardian blocked: ${guardianResult.blockReason}`,
        };
      }

      // Convert USD to input token lamports (handles both SOL and USDC)
      const inputAmount = await this.usdToInputLamports(params.amountUsd, params.inputMint);

      logger.info('[SpotExecutor] Converted USD to input lamports', {
        amountUsd: params.amountUsd,
        inputLamports: inputAmount,
        inputMint: params.inputMint,
      });

      // Get Jupiter Ultra order (quote + transaction in one call)
      const order = await this.getJupiterUltraOrder(
        params.inputMint,
        params.outputMint,
        inputAmount
      );

      logger.info('[SpotExecutor] Got Jupiter Ultra order', {
        requestId: order.requestId,
        inputAmount: order.inAmount,
        outputAmount: order.outAmount,
        priceImpact: order.priceImpactPct,
      });

      // Execute swap
      const signature = await this.executeJupiterUltraSwap(order);
      const outputAmount = parseInt(order.outAmount);

      logger.info('[SpotExecutor] Buy successful', {
        signature,
        outputAmount,
      });

      return {
        success: true,
        signature,
        outputAmount,
      };
    } catch (error: any) {
      logger.error('[SpotExecutor] Buy failed', {
        error: error.message,
        outputMint: params.outputMint,
      });
      return {
        success: false,
        error: error.message,
      };
    }
  }

  /**
   * Sell tokens for USDC
   */
  async sell(params: SpotSellParams, analysisContext?: { confidence: number; reasoning: string; portfolioValueUsd: number }): Promise<SpotExecutionResult> {
    try {
      logger.info('[SpotExecutor] Executing sell', {
        inputMint: params.inputMint,
        amountTokens: params.amountTokens,
      });

      // Fetch real token decimals from mint account
      const tokenDecimals = await this.getTokenDecimals(params.inputMint);
      const tokenLamports = Math.floor(params.amountTokens * Math.pow(10, tokenDecimals));

      // Get token price for USD value estimation
      const tokenPrice = await this.getJupiterPrice(params.inputMint);
      const estimatedUsdValue = tokenPrice ? params.amountTokens * tokenPrice : params.amountTokens;

      // PM Approval check (before Guardian)
      if (pmDecisionEngine.isEnabled() && analysisContext) {
        const pmParams: QueueTradeParams = {
          strategy: 'spot',
          action: 'SELL',
          asset: params.inputMint,
          assetMint: params.inputMint,
          amount: params.amountTokens,
          amountUsd: estimatedUsdValue,
          confidence: analysisContext.confidence,
          risk: {
            volatility: 0,
            liquidityScore: 50,
            riskScore: 30,
          },
          reasoning: analysisContext.reasoning,
          protocol: 'jupiter',
        };

        const needsApproval = pmDecisionEngine.needsApproval(pmParams, analysisContext.portfolioValueUsd);

        if (needsApproval) {
          logger.info('[SpotExecutor] Trade requires PM approval', {
            amountUsd: estimatedUsdValue,
            inputMint: params.inputMint,
          });

          const tradeId = approvalQueue.queueTrade(pmParams);
          const approvalResult = await approvalQueue.waitForApproval(tradeId);

          if (!approvalResult.approved) {
            logger.warn('[SpotExecutor] PM rejected sell transaction', {
              tradeId,
              status: approvalResult.status,
              reason: approvalResult.rejectionReason,
            });
            return {
              success: false,
              error: `PM rejected: ${approvalResult.rejectionReason || approvalResult.status}`,
            };
          }

          logger.info('[SpotExecutor] PM approved sell transaction', {
            tradeId,
            approver: approvalResult.approver,
            waitTimeMs: approvalResult.waitTimeMs,
          });
        }
      }

      // Guardian pre-execution validation
      const guardianParams: GuardianTradeParams = {
        inputMint: params.inputMint,
        outputMint: params.outputMint,
        amountIn: params.amountTokens,
        amountInUsd: estimatedUsdValue,
        slippageBps: this.slippageBps,
        strategy: 'spot',
        protocol: 'jupiter',
        walletAddress: this.wallet.publicKey.toBase58(),
      };

      const guardianResult = await guardian.validate(guardianParams);
      if (!guardianResult.approved) {
        logger.warn('[SpotExecutor] Guardian blocked sell transaction', {
          reason: guardianResult.blockReason,
          inputMint: params.inputMint,
          amountTokens: params.amountTokens,
        });
        return {
          success: false,
          error: `Guardian blocked: ${guardianResult.blockReason}`,
        };
      }

      logger.info('[SpotExecutor] Token conversion', {
        inputMint: params.inputMint,
        decimals: tokenDecimals,
        amountTokens: params.amountTokens,
        tokenLamports,
      });

      // Get Jupiter Ultra order (quote + transaction in one call)
      const order = await this.getJupiterUltraOrder(
        params.inputMint,
        params.outputMint,
        tokenLamports
      );

      logger.info('[SpotExecutor] Got Jupiter Ultra order', {
        requestId: order.requestId,
        inputAmount: order.inAmount,
        outputAmount: order.outAmount,
        priceImpact: order.priceImpactPct,
      });

      // Execute swap
      const signature = await this.executeJupiterUltraSwap(order);
      const usdcReceived = parseInt(order.outAmount) / 1_000_000; // Convert USDC lamports to USD

      logger.info('[SpotExecutor] Sell successful', {
        signature,
        usdcReceived,
      });

      return {
        success: true,
        signature,
        usdReceived: usdcReceived,
      };
    } catch (error: any) {
      logger.error('[SpotExecutor] Sell failed', {
        error: error.message,
        inputMint: params.inputMint,
      });
      return {
        success: false,
        error: error.message,
      };
    }
  }
}

