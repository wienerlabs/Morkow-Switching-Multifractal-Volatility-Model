/**
 * Pump.fun API Client
 * Trading execution and data fetching for Pump.fun platform
 */

import { Connection, Keypair, PublicKey, VersionedTransaction, TransactionMessage, SystemProgram } from '@solana/web3.js';
import bs58 from 'bs58';
import { logger } from '../logger.js';
import { getSolanaConnection, recordRpcFailure, recordRpcSuccess } from '../solana/connection.js';
import { guardian } from '../guardian/index.js';
import type { GuardianTradeParams } from '../guardian/types.js';
import { sendWithJito } from '../jitoService.js';

// ============= TRADING TYPES =============

export interface TransactionResult {
  success: boolean;
  signature?: string;
  error?: string;
  tokensBought?: number;
  tokensSold?: number;
  solSpent?: number;
  solReceived?: number;
  pricePerToken?: number;
  usedJito?: boolean;
}

export interface SimulationResult {
  success: boolean;
  expectedTokens: number;
  expectedPrice: number;
  priceImpactPct: number;
  minTokensOut: number;
  error?: string;
}

export interface BuyParams {
  tokenMint: string;
  amountSol: number;
  slippageBps: number;
}

export interface SellParams {
  tokenMint: string;
  amountTokens: number;
  slippageBps: number;
}

export interface PumpFunToken {
  coinMint: string;
  dev: string;
  name: string;
  ticker: string;
  imageUrl: string;
  creationTime: number;
  numHolders: number;
  marketCap: number;
  volume: number;
  currentMarketPrice: number;
  bondingCurveProgress: number;
  sniperCount: number;
  graduationDate: number | null;
  holders: Array<{
    totalTokenAmountHeld: number;
    isSniper: boolean;
    ownedPercentage: number;
    holderId: string;
  }>;
  isMayhemMode: boolean;
  allTimeHighMarketCap: number;
  poolAddress: string | null;
  twitter?: string;
  telegram?: string;
  website?: string;
  hasTwitter: boolean;
  hasTelegram: boolean;
  hasWebsite: boolean;
  hasSocial: boolean;
  twitterReuseCount: number;
  devHoldingsPercentage: number;
  buyTransactions: number;
  sellTransactions: number;
  transactions: number;
  sniperOwnedPercentage: number;
  topHoldersPercentage: number;
  tokenProgram: string;
}

export interface PumpFunApiResponse {
  coins: PumpFunToken[];
  total: number;
}

// SOL mint constant
const SOL_MINT = 'So11111111111111111111111111111111111111112';

export class PumpFunClient {
  private readonly baseUrl = 'https://advanced-api-v2.pump.fun';
  private readonly tradeApiUrl = 'https://pumpportal.fun/api';
  private apiKey: string;
  private walletPublic: string;
  private walletPrivate: string;
  private keypair: Keypair | null = null;
  private connection: Connection | null = null;
  private useJito: boolean = true;
  private jitoTipLamports: number = 10000;

  constructor() {
    this.apiKey = process.env.PUMPFUN_API_KEY || '';
    this.walletPublic = process.env.PUMPFUN_WALLET_PUBLIC || '';
    this.walletPrivate = process.env.PUMPFUN_WALLET_PRIVATE || '';

    if (!this.apiKey) {
      logger.warn('[PumpFunClient] PUMPFUN_API_KEY not configured - API access will be limited');
    }

    if (this.walletPrivate) {
      try {
        this.keypair = Keypair.fromSecretKey(bs58.decode(this.walletPrivate));
        this.walletPublic = this.keypair.publicKey.toBase58();
        logger.info('[PumpFunClient] Wallet configured', { publicKey: this.walletPublic });
      } catch (e) {
        logger.error('[PumpFunClient] Invalid wallet private key');
      }
    } else {
      logger.info('[PumpFunClient] No wallet configured - read-only mode');
    }

    // Initialize Solana connection — uses failover connection
    this.connection = getSolanaConnection();

    this.useJito = process.env.PUMPFUN_USE_JITO !== 'false';
    this.jitoTipLamports = parseInt(process.env.PUMPFUN_JITO_TIP_LAMPORTS || '10000', 10);

    logger.info('[PumpFunClient] Initialized', {
      apiConfigured: !!this.apiKey,
      walletConfigured: !!this.keypair,
      useJito: this.useJito,
    });
  }

  /**
   * Check if Pump.fun client is properly configured for reading
   */
  isConfigured(): boolean {
    return !!this.apiKey;
  }

  /**
   * Check if trading is enabled (wallet configured)
   */
  isTradingEnabled(): boolean {
    return !!this.keypair && !!this.connection;
  }

  /**
   * Fetch tokens from Pump.fun API
   * @param limit Number of tokens to fetch (default: 50)
   * @param offset Pagination offset (default: 0)
   */
  async getTokens(limit: number = 50, offset: number = 0): Promise<PumpFunToken[]> {
    if (!this.isConfigured()) {
      logger.warn('[PumpFunClient] Cannot fetch tokens - API key not configured');
      return [];
    }

    try {
      const url = `${this.baseUrl}/coins/list?sortBy=creationTime&direction=desc&limit=${limit}&offset=${offset}`;

      logger.info('[PumpFunClient] Fetching tokens', { limit, offset, url });

      const response = await fetch(url, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Accept': 'application/json',
        },
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Pump.fun API error: ${response.status} ${response.statusText} - ${errorText}`);
      }

      const responseData = await response.json() as any;

      // Handle both array and object responses
      const data = Array.isArray(responseData) ? responseData : (responseData.data || responseData.coins || []);

      logger.info('[PumpFunClient] Tokens fetched successfully', {
        count: data.length,
        total: data.length,
      });

      return data as PumpFunToken[];
    } catch (error: any) {
      logger.error('[PumpFunClient] Failed to fetch tokens', {
        error: error.message,
        stack: error.stack,
      });
      return [];
    }
  }

  /**
   * Get token details by mint address
   */
  async getTokenByMint(mint: string): Promise<PumpFunToken | null> {
    if (!this.isConfigured()) {
      logger.warn('[PumpFunClient] Cannot fetch token - API key not configured');
      return null;
    }

    try {
      const url = `${this.baseUrl}/coins/${mint}`;

      const response = await fetch(url, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Accept': 'application/json',
        },
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Pump.fun API error: ${response.status} ${response.statusText} - ${errorText}`);
      }

      return await response.json() as PumpFunToken;
    } catch (error: any) {
      logger.error('[PumpFunClient] Failed to fetch token details', {
        mint,
        error: error.message,
      });
      return null;
    }
  }

  /**
   * Get wallet public key
   */
  getWalletPublic(): string {
    return this.walletPublic;
  }

  // ============= TRADING METHODS =============

  /**
   * Get current SOL price in USD from token data
   */
  private async getSolPrice(): Promise<number> {
    const envPrice = process.env.SOL_PRICE;
    if (envPrice) return parseFloat(envPrice);

    try {
      // Fetch from CoinGecko
      const response = await fetch('https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd');
      if (response.ok) {
        const data = await response.json() as { solana: { usd: number } };
        return data.solana.usd;
      }
    } catch {
      // Fallback
    }
    return 150; // Default fallback
  }

  /**
   * Get bonding curve price for a token
   */
  async getBondingCurvePrice(tokenMint: string): Promise<number> {
    const token = await this.getTokenByMint(tokenMint);
    if (!token) {
      throw new Error(`Token not found: ${tokenMint}`);
    }
    return token.currentMarketPrice;
  }

  /**
   * Simulate a buy to get expected output
   */
  async simulateBuy(tokenMint: string, amountSol: number): Promise<SimulationResult> {
    try {
      const token = await this.getTokenByMint(tokenMint);
      if (!token) {
        return { success: false, expectedTokens: 0, expectedPrice: 0, priceImpactPct: 0, minTokensOut: 0, error: 'Token not found' };
      }

      // Calculate expected tokens based on bonding curve
      const price = token.currentMarketPrice;
      const expectedTokens = amountSol / price;

      // Estimate price impact based on market cap and buy size
      const solPrice = await this.getSolPrice();
      const buyUsd = amountSol * solPrice;
      const priceImpactPct = (buyUsd / token.marketCap) * 100;

      // Calculate minimum tokens with default 5% slippage
      const minTokensOut = expectedTokens * 0.95;

      logger.info('[PumpFunClient] Buy simulation', {
        tokenMint,
        amountSol,
        expectedTokens,
        price,
        priceImpactPct,
      });

      return {
        success: true,
        expectedTokens,
        expectedPrice: price,
        priceImpactPct,
        minTokensOut,
      };
    } catch (error: any) {
      logger.error('[PumpFunClient] Simulation failed', { error: error.message });
      return { success: false, expectedTokens: 0, expectedPrice: 0, priceImpactPct: 0, minTokensOut: 0, error: error.message };
    }
  }

  /**
   * Buy tokens from PumpFun bonding curve
   */
  async buy(params: BuyParams): Promise<TransactionResult> {
    if (!this.isTradingEnabled()) {
      return { success: false, error: 'Trading not enabled - wallet not configured' };
    }

    const { tokenMint, amountSol, slippageBps } = params;

    try {
      // Fetch current token data
      const token = await this.getTokenByMint(tokenMint);
      if (!token) {
        return { success: false, error: `Token not found: ${tokenMint}` };
      }

      // Get SOL price for Guardian validation
      const solPrice = await this.getSolPrice();
      const amountInUsd = amountSol * solPrice;

      // Simulate to get price impact
      const simulation = await this.simulateBuy(tokenMint, amountSol);
      if (!simulation.success) {
        return { success: false, error: `Simulation failed: ${simulation.error}` };
      }

      // Guardian pre-execution validation
      const guardianParams: GuardianTradeParams = {
        inputMint: SOL_MINT,
        outputMint: tokenMint,
        amountIn: amountSol,
        amountInUsd,
        slippageBps,
        priceImpactPct: simulation.priceImpactPct,
        strategy: 'pumpfun',
        protocol: 'pump.fun',
        walletAddress: this.keypair!.publicKey.toBase58(),
      };

      const guardianResult = await guardian.validate(guardianParams);
      if (!guardianResult.approved) {
        logger.warn('[PumpFunClient] Guardian blocked buy', {
          reason: guardianResult.blockReason,
          tokenMint,
          amountSol,
        });
        return { success: false, error: `Guardian blocked: ${guardianResult.blockReason}` };
      }

      logger.info('[PumpFunClient] Executing buy', {
        tokenMint,
        amountSol,
        amountInUsd,
        slippageBps,
        priceImpactPct: simulation.priceImpactPct,
      });

      // Build and execute transaction via PumpPortal API
      const txResult = await this.executePumpPortalTrade({
        action: 'buy',
        mint: tokenMint,
        amount: amountSol,
        denominatedInSol: 'true',
        slippage: slippageBps / 100, // Convert bps to percent
        priorityFee: this.jitoTipLamports / 1e9, // Convert lamports to SOL
        pool: 'pump',
      });

      if (!txResult.success) {
        return txResult;
      }

      return {
        success: true,
        signature: txResult.signature,
        tokensBought: simulation.expectedTokens,
        solSpent: amountSol,
        pricePerToken: simulation.expectedPrice,
        usedJito: this.useJito,
      };

    } catch (error: any) {
      logger.error('[PumpFunClient] Buy failed', {
        tokenMint,
        amountSol,
        error: error.message,
      });
      return { success: false, error: error.message };
    }
  }

  /**
   * Sell tokens back to PumpFun bonding curve
   */
  async sell(params: SellParams): Promise<TransactionResult> {
    if (!this.isTradingEnabled()) {
      return { success: false, error: 'Trading not enabled - wallet not configured' };
    }

    const { tokenMint, amountTokens, slippageBps } = params;

    try {
      // Fetch current token data
      const token = await this.getTokenByMint(tokenMint);
      if (!token) {
        return { success: false, error: `Token not found: ${tokenMint}` };
      }

      // Calculate expected SOL output
      const price = token.currentMarketPrice;
      const expectedSol = amountTokens * price;

      // Get SOL price for Guardian validation
      const solPrice = await this.getSolPrice();
      const amountInUsd = expectedSol * solPrice;

      // Estimate price impact
      const priceImpactPct = (amountInUsd / token.marketCap) * 100;

      // Guardian pre-execution validation
      const guardianParams: GuardianTradeParams = {
        inputMint: tokenMint,
        outputMint: SOL_MINT,
        amountIn: amountTokens,
        amountInUsd,
        slippageBps,
        priceImpactPct,
        strategy: 'pumpfun',
        protocol: 'pump.fun',
        walletAddress: this.keypair!.publicKey.toBase58(),
      };

      const guardianResult = await guardian.validate(guardianParams);
      if (!guardianResult.approved) {
        logger.warn('[PumpFunClient] Guardian blocked sell', {
          reason: guardianResult.blockReason,
          tokenMint,
          amountTokens,
        });
        return { success: false, error: `Guardian blocked: ${guardianResult.blockReason}` };
      }

      logger.info('[PumpFunClient] Executing sell', {
        tokenMint,
        amountTokens,
        expectedSol,
        amountInUsd,
        slippageBps,
      });

      // Build and execute transaction via PumpPortal API
      const txResult = await this.executePumpPortalTrade({
        action: 'sell',
        mint: tokenMint,
        amount: amountTokens,
        denominatedInSol: 'false',
        slippage: slippageBps / 100,
        priorityFee: this.jitoTipLamports / 1e9,
        pool: 'pump',
      });

      if (!txResult.success) {
        return txResult;
      }

      return {
        success: true,
        signature: txResult.signature,
        tokensSold: amountTokens,
        solReceived: expectedSol,
        pricePerToken: price,
        usedJito: this.useJito,
      };

    } catch (error: any) {
      logger.error('[PumpFunClient] Sell failed', {
        tokenMint,
        amountTokens,
        error: error.message,
      });
      return { success: false, error: error.message };
    }
  }

  /**
   * Execute trade via PumpPortal API
   * Uses their transaction building API with optional Jito protection
   */
  private async executePumpPortalTrade(params: {
    action: 'buy' | 'sell';
    mint: string;
    amount: number;
    denominatedInSol: string;
    slippage: number;
    priorityFee: number;
    pool: string;
  }): Promise<TransactionResult> {
    try {
      // Request transaction from PumpPortal API
      const response = await fetch(`${this.tradeApiUrl}/trade-local`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          publicKey: this.keypair!.publicKey.toBase58(),
          action: params.action,
          mint: params.mint,
          amount: params.amount,
          denominatedInSol: params.denominatedInSol,
          slippage: params.slippage,
          priorityFee: params.priorityFee,
          pool: params.pool,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`PumpPortal API error: ${response.status} - ${errorText}`);
      }

      // Get serialized transaction
      const txData = await response.arrayBuffer();
      const transaction = VersionedTransaction.deserialize(new Uint8Array(txData));

      // Sign transaction
      transaction.sign([this.keypair!]);

      // Execute with Jito MEV protection if enabled
      if (this.useJito) {
        const jitoResult = await sendWithJito(
          this.connection!,
          this.keypair!,
          transaction,
          {
            tipLamports: this.jitoTipLamports,
            network: 'mainnet',
            fallbackToRpc: true,
          }
        );

        if (!jitoResult.success) {
          return { success: false, error: `Jito execution failed: ${jitoResult.error}` };
        }

        logger.info('[PumpFunClient] Trade executed with Jito', {
          signature: jitoResult.signature,
          bundleId: jitoResult.bundleId,
          tipPaid: jitoResult.tipPaid,
        });

        return {
          success: true,
          signature: jitoResult.signature,
          usedJito: true,
        };
      }

      // Standard execution without Jito
      const signature = await this.connection!.sendTransaction(transaction, {
        skipPreflight: false,
        maxRetries: 3,
        preflightCommitment: 'confirmed',
      });

      // Confirm transaction
      const latestBlockhash = await this.connection!.getLatestBlockhash();
      await this.connection!.confirmTransaction({
        signature,
        blockhash: latestBlockhash.blockhash,
        lastValidBlockHeight: latestBlockhash.lastValidBlockHeight,
      }, 'confirmed');

      recordRpcSuccess();
      logger.info('[PumpFunClient] Trade executed via RPC', { signature });

      return {
        success: true,
        signature,
        usedJito: false,
      };

    } catch (error: any) {
      recordRpcFailure();
      logger.error('[PumpFunClient] Trade execution failed', {
        action: params.action,
        mint: params.mint,
        error: error.message,
      });
      return { success: false, error: error.message };
    }
  }
}

