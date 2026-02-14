/**
 * Simple Lending Executor - Wrapper for CRTXAgent
 * 
 * Executes lending deposits/withdrawals across Kamino, MarginFi, and Solend.
 * Uses existing protocol clients from the lending service.
 */

import { Connection, Keypair } from '@solana/web3.js';
import { logger } from '../logger.js';
import { MarginFiLendingClient } from './marginfiClient.js';
import { KaminoLendingClient } from './kaminoClient.js';
import { SolendLendingClient } from './solendClient.js';
import type { LendingProtocol, LendingResult } from './types.js';
import { OracleService } from '../risk/oracleService.js';
import bs58 from 'bs58';

export interface SimpleLendingExecutorConfig {
  rpcUrl: string;
  wallet: Keypair;
}

export interface LendingDepositParams {
  protocol: LendingProtocol;
  asset: string;      // Token symbol (e.g., 'USDC', 'SOL')
  amountUsd: number;  // USD amount to deposit
}

export interface LendingWithdrawParams {
  protocol: LendingProtocol;
  asset: string;
  amountUsd: number;
}

/**
 * Simplified LendingExecutor for CRTXAgent
 * Wraps existing protocol clients and handles USD -> token conversion
 */
export class SimpleLendingExecutor {
  private connection: Connection;
  private wallet: Keypair;
  private privateKeyBase58: string;
  private oracleService: OracleService;

  private marginfiClient: MarginFiLendingClient | null = null;
  private kaminoClient: KaminoLendingClient | null = null;
  private solendClient: SolendLendingClient | null = null;

  private initialized = false;

  constructor(private config: SimpleLendingExecutorConfig) {
    this.connection = new Connection(config.rpcUrl, 'confirmed');
    this.wallet = config.wallet;
    this.privateKeyBase58 = bs58.encode(config.wallet.secretKey);
    this.oracleService = new OracleService(config.rpcUrl);
  }

  /**
   * Initialize protocol clients
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    logger.info('[SimpleLendingExecutor] Initializing protocol clients...');

    // Initialize MarginFi
    try {
      this.marginfiClient = new MarginFiLendingClient({
        rpcUrl: this.config.rpcUrl,
        privateKey: this.privateKeyBase58,
        environment: 'production',
      });
      await this.marginfiClient.initialize();
      logger.info('[SimpleLendingExecutor] MarginFi initialized');
    } catch (error: any) {
      logger.error('[SimpleLendingExecutor] Failed to initialize MarginFi', {
        errorMessage: error?.message || 'Unknown error',
        errorName: error?.name,
        errorStack: error?.stack,
        errorDetails: error?.toString(),
      });
    }

    // Initialize Kamino
    try {
      this.kaminoClient = new KaminoLendingClient({
        rpcUrl: this.config.rpcUrl,
        privateKey: this.privateKeyBase58,
      });
      await this.kaminoClient.initialize();
      logger.info('[SimpleLendingExecutor] Kamino initialized');
    } catch (error: any) {
      logger.error('[SimpleLendingExecutor] Failed to initialize Kamino', {
        errorMessage: error?.message || 'Unknown error',
        errorName: error?.name,
        errorStack: error?.stack,
        errorDetails: error?.toString(),
      });
    }

    // Initialize Solend
    try {
      this.solendClient = new SolendLendingClient({
        rpcUrl: this.config.rpcUrl,
        privateKey: this.privateKeyBase58,
      });
      await this.solendClient.initialize();
      logger.info('[SimpleLendingExecutor] Solend initialized');
    } catch (error: any) {
      logger.error('[SimpleLendingExecutor] Failed to initialize Solend', {
        errorMessage: error?.message || 'Unknown error',
        errorName: error?.name,
        errorStack: error?.stack,
        errorDetails: error?.toString(),
      });
    }

    this.initialized = true;
    logger.info('[SimpleLendingExecutor] All protocol clients initialized');
  }

  /**
   * Convert USD amount to token amount using real-time oracle prices
   */
  private async usdToTokenAmount(asset: string, amountUsd: number): Promise<number> {
    try {
      // Get real-time price from Jupiter oracle
      const oracleStatus = await this.oracleService.getJupiterPrice(asset);

      if (oracleStatus.isStale) {
        logger.warn('[SimpleLendingExecutor] Oracle price is stale', {
          asset,
          stalenessSeconds: oracleStatus.stalenessSeconds,
        });
      }

      const tokenAmount = amountUsd / oracleStatus.price;

      logger.info('[SimpleLendingExecutor] USD to token conversion', {
        asset,
        amountUsd,
        price: oracleStatus.price,
        tokenAmount,
        source: oracleStatus.source,
      });

      return tokenAmount;
    } catch (error: any) {
      // Fallback to stablecoin assumption if oracle fails
      logger.error('[SimpleLendingExecutor] Failed to get oracle price, using fallback', {
        asset,
        error: error.message,
      });

      // Try DexScreener as fallback
      try {
        const dexScreenerResponse = await fetch(`https://api.dexscreener.com/latest/dex/search?q=${asset}`);
        const dexScreenerData = await dexScreenerResponse.json() as { pairs?: Array<{ priceUsd?: string }> };
        if (dexScreenerData.pairs && dexScreenerData.pairs[0]?.priceUsd) {
          const price = parseFloat(dexScreenerData.pairs[0].priceUsd);
          logger.info('[SimpleLendingExecutor] Using DexScreener fallback price', { asset, price });
          return amountUsd / price;
        }
      } catch (e) {
        logger.warn('[SimpleLendingExecutor] DexScreener fallback also failed');
      }

      // Fallback prices for stablecoins
      const fallbackPrices: Record<string, number> = {
        'USDC': 1,
        'USDT': 1,
        'PYUSD': 1,
        'USDS': 1,
        'SOL': 150,  // Last-resort fallback only if DexScreener is also down
      };

      const price = fallbackPrices[asset.toUpperCase()];
      if (!price) {
        throw new Error(`Cannot get price for ${asset} and no fallback available`);
      }

      return amountUsd / price;
    }
  }

  /**
   * Check if we have enough balance of a token
   */
  private async hasTokenBalance(asset: string, requiredAmount: number): Promise<boolean> {
    try {
      const { getAssociatedTokenAddress, getAccount } = await import('@solana/spl-token');
      const { PublicKey } = await import('@solana/web3.js');
      const { LENDING_TOKEN_MINTS, TOKEN_DECIMALS } = await import('./types.js');

      // Native SOL check
      if (asset.toUpperCase() === 'SOL') {
        const balance = await this.connection.getBalance(this.wallet.publicKey);
        const balanceSOL = balance / 1e9;
        logger.info('[SimpleLendingExecutor] SOL balance check', { balance: balanceSOL, required: requiredAmount });
        return balanceSOL >= requiredAmount;
      }

      // SPL Token check
      const mintAddress = LENDING_TOKEN_MINTS[asset.toUpperCase()];
      if (!mintAddress) {
        logger.warn('[SimpleLendingExecutor] Unknown token for balance check', { asset });
        return false;
      }

      const mintPubkey = new PublicKey(mintAddress);
      const ata = await getAssociatedTokenAddress(mintPubkey, this.wallet.publicKey);

      try {
        const accountInfo = await getAccount(this.connection, ata);
        const decimals = TOKEN_DECIMALS[asset.toUpperCase()] || 6;
        const balance = Number(accountInfo.amount) / Math.pow(10, decimals);
        logger.info('[SimpleLendingExecutor] Token balance check', { asset, balance, required: requiredAmount });
        return balance >= requiredAmount;
      } catch (error) {
        // ATA doesn't exist = no balance
        logger.info('[SimpleLendingExecutor] No token account found', { asset });
        return false;
      }
    } catch (error: any) {
      logger.error('[SimpleLendingExecutor] Balance check failed', { asset, error: error.message });
      return false;
    }
  }

  /**
   * Buy token using Jupiter swap (USDC -> target token)
   */
  private async buyTokenWithJupiter(asset: string, amountUsd: number): Promise<number> {
    try {
      const { VersionedTransaction } = await import('@solana/web3.js');
      const { LENDING_TOKEN_MINTS, TOKEN_DECIMALS } = await import('./types.js');

      logger.info('[SimpleLendingExecutor] Buying token with Jupiter', { asset, amountUsd });

      const targetMint = LENDING_TOKEN_MINTS[asset.toUpperCase()];
      const usdcMint = LENDING_TOKEN_MINTS['USDC'];

      if (!targetMint) {
        throw new Error(`Unknown token: ${asset}`);
      }

      // Get Jupiter quote
      const apiKey = process.env.JUPITER_API_KEY;
      const walletAddress = this.wallet.publicKey.toBase58();

      if (!apiKey) {
        throw new Error('JUPITER_API_KEY not found in .env');
      }

      const inputLamports = Math.floor(amountUsd * 1e6); // USDC has 6 decimals
      const url = `https://api.jup.ag/ultra/v1/order?inputMint=${usdcMint}&outputMint=${targetMint}&amount=${inputLamports}&taker=${walletAddress}`;

      logger.info('[SimpleLendingExecutor] Fetching Jupiter quote...', { inputUsd: amountUsd });

      const response = await fetch(url, {
        method: 'GET',
        headers: { 'x-api-key': apiKey },
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Jupiter API error: ${response.status} - ${errorText}`);
      }

      const quote = await response.json() as any;

      if (!quote.swapTransaction && !quote.transaction) {
        throw new Error('No transaction in Jupiter quote');
      }

      // Execute swap
      const swapTransaction = (quote.swapTransaction || quote.transaction)!;
      const txBuffer = Buffer.from(swapTransaction, 'base64');
      const transaction = VersionedTransaction.deserialize(txBuffer);

      logger.info('[SimpleLendingExecutor] Signing and sending Jupiter swap...');
      transaction.sign([this.wallet]);

      const txid = await this.connection.sendTransaction(transaction, {
        skipPreflight: false,
        maxRetries: 3,
      });

      logger.info('[SimpleLendingExecutor] Jupiter swap sent', { txid });

      // Wait for confirmation
      const startTime = Date.now();
      const timeout = 60000;
      let confirmed = false;

      while (!confirmed && Date.now() - startTime < timeout) {
        const status = await this.connection.getSignatureStatus(txid);

        if (status?.value?.err) {
          throw new Error(`Jupiter swap failed: ${JSON.stringify(status.value.err)}`);
        }

        if (status?.value?.confirmationStatus === 'confirmed' || status?.value?.confirmationStatus === 'finalized') {
          confirmed = true;
          break;
        }
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      if (!confirmed) {
        throw new Error(`Jupiter swap timeout after 60 seconds`);
      }

      // Calculate received amount
      const decimals = TOKEN_DECIMALS[asset.toUpperCase()] || 6;
      const receivedAmount = parseInt(quote.outAmount) / Math.pow(10, decimals);

      logger.info('[SimpleLendingExecutor] Jupiter swap successful', {
        asset,
        receivedAmount,
        txid
      });

      return receivedAmount;
    } catch (error: any) {
      logger.error('[SimpleLendingExecutor] Jupiter swap failed', { asset, error: error.message });
      throw error;
    }
  }

  /**
   * Execute lending deposit
   */
  async deposit(params: LendingDepositParams): Promise<LendingResult> {
    if (!this.initialized) {
      await this.initialize();
    }

    logger.info('[SimpleLendingExecutor] Executing deposit', {
      protocol: params.protocol,
      asset: params.asset,
      amountUsd: params.amountUsd,
    });

    // Convert USD to token amount
    let tokenAmount = await this.usdToTokenAmount(params.asset, params.amountUsd);

    // Check if we have enough balance
    const hasBalance = await this.hasTokenBalance(params.asset, tokenAmount);

    if (!hasBalance) {
      logger.info('[SimpleLendingExecutor] Insufficient balance, buying token with Jupiter', {
        asset: params.asset,
        required: tokenAmount,
      });

      try {
        // Buy token with USDC via Jupiter
        tokenAmount = await this.buyTokenWithJupiter(params.asset, params.amountUsd);
        logger.info('[SimpleLendingExecutor] Token purchased successfully', {
          asset: params.asset,
          amount: tokenAmount,
        });
      } catch (error: any) {
        logger.error('[SimpleLendingExecutor] Failed to buy token', {
          asset: params.asset,
          error: error.message,
        });
        return {
          success: false,
          error: `Failed to buy ${params.asset}: ${error.message}`,
        };
      }
    }

    // Route to appropriate protocol client
    try {
      let result: LendingResult;

      switch (params.protocol) {
        case 'marginfi':
          if (!this.marginfiClient) {
            throw new Error('MarginFi client not initialized');
          }
          result = await this.marginfiClient.deposit({
            asset: params.asset,
            amount: tokenAmount,
          });
          break;

        case 'kamino':
          if (!this.kaminoClient) {
            throw new Error('Kamino client not initialized');
          }
          result = await this.kaminoClient.deposit({
            asset: params.asset,
            amount: tokenAmount,
          });
          break;

        case 'solend':
          if (!this.solendClient) {
            throw new Error('Solend client not initialized');
          }
          result = await this.solendClient.deposit({
            asset: params.asset,
            amount: tokenAmount,
          });
          break;

        default:
          throw new Error(`Unknown protocol: ${params.protocol}`);
      }

      logger.info('[SimpleLendingExecutor] Deposit successful', {
        protocol: params.protocol,
        asset: params.asset,
        signature: result.signature,
      });

      return result;
    } catch (error: any) {
      logger.error('[SimpleLendingExecutor] Deposit failed', {
        protocol: params.protocol,
        asset: params.asset,
        error: error.message,
      });
      return {
        success: false,
        error: error.message,
      };
    }
  }

  /**
   * Execute lending withdrawal
   */
  async withdraw(params: LendingWithdrawParams): Promise<LendingResult> {
    if (!this.initialized) {
      await this.initialize();
    }

    logger.info('[SimpleLendingExecutor] Executing withdrawal', {
      protocol: params.protocol,
      asset: params.asset,
      amountUsd: params.amountUsd,
    });

    // Convert USD to token amount
    const tokenAmount = await this.usdToTokenAmount(params.asset, params.amountUsd);

    // Route to appropriate protocol client
    try {
      let result: LendingResult;

      switch (params.protocol) {
        case 'marginfi':
          if (!this.marginfiClient) {
            throw new Error('MarginFi client not initialized');
          }
          result = await this.marginfiClient.withdraw({
            asset: params.asset,
            amount: tokenAmount,
          });
          break;

        case 'kamino':
          if (!this.kaminoClient) {
            throw new Error('Kamino client not initialized');
          }
          result = await this.kaminoClient.withdraw({
            asset: params.asset,
            amount: tokenAmount,
          });
          break;

        case 'solend':
          if (!this.solendClient) {
            throw new Error('Solend client not initialized');
          }
          result = await this.solendClient.withdraw({
            asset: params.asset,
            amount: tokenAmount,
          });
          break;

        default:
          throw new Error(`Unknown protocol: ${params.protocol}`);
      }

      logger.info('[SimpleLendingExecutor] Withdrawal successful', {
        protocol: params.protocol,
        asset: params.asset,
        signature: result.signature,
      });

      return result;
    } catch (error: any) {
      logger.error('[SimpleLendingExecutor] Withdrawal failed', {
        protocol: params.protocol,
        asset: params.asset,
        error: error.message,
      });
      return {
        success: false,
        error: error.message,
      };
    }
  }
}

