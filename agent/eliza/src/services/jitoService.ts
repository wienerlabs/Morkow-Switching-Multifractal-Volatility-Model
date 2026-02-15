/**
 * Jito MEV Protection Service
 *
 * Provides MEV protection for Jupiter swaps using Jito bundles.
 * Prevents frontrunning and sandwich attacks by submitting transactions
 * directly to Jito block engine.
 */
import {
  Connection,
  VersionedTransaction,
  Keypair,
  PublicKey,
  SystemProgram,
  TransactionMessage,
} from '@solana/web3.js';
import { JitoJsonRpcClient } from 'jito-js-rpc';
import { logger } from './logger.js';
import { resilientFetch } from './resilience.js';

// Jito endpoints
const JITO_ENDPOINTS = {
  mainnet: 'https://mainnet.block-engine.jito.wtf/api/v1/bundles',
  devnet: 'https://dallas.testnet.block-engine.jito.wtf/api/v1/bundles',
};

export interface JitoConfig {
  tipLamports: number;          // Base tip amount (default: 10000 = 0.00001 SOL)
  maxTipLamports: number;       // Max tip for high volatility (default: 100000)
  useBundle: boolean;           // Enable bundle submission
  fallbackToRpc: boolean;       // Fallback to normal RPC on failure
  network: 'mainnet' | 'devnet';
}

export interface JitoResult {
  success: boolean;
  signature?: string;
  bundleId?: string;
  usedJito: boolean;
  tipPaid?: number;
  error?: string;
}

const DEFAULT_CONFIG: JitoConfig = {
  tipLamports: 10000,           // 0.00001 SOL
  maxTipLamports: 100000,       // 0.0001 SOL
  useBundle: true,
  fallbackToRpc: true,
  network: 'mainnet',
};

// Cached Jito client instance
let jitoClient: JitoJsonRpcClient | null = null;

/**
 * Get or create Jito client
 */
function getJitoClient(network: 'mainnet' | 'devnet'): JitoJsonRpcClient {
  if (!jitoClient) {
    jitoClient = new JitoJsonRpcClient(JITO_ENDPOINTS[network]);
  }
  return jitoClient;
}

/**
 * Get a random Jito tip account from API
 */
async function getRandomTipAccount(client: JitoJsonRpcClient): Promise<PublicKey> {
  try {
    const tipAccount = await client.getRandomTipAccount();
    return new PublicKey(tipAccount);
  } catch {
    // Fallback to known tip account
    return new PublicKey('96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5');
  }
}

/**
 * Calculate dynamic tip based on volatility and trade size
 */
export function calculateDynamicTip(
  volatility24h: number,
  tradeSizeUsd: number,
  config: JitoConfig = DEFAULT_CONFIG
): number {
  // Base tip
  let tip = config.tipLamports;
  
  // Increase tip for high volatility (>5% = competitive)
  if (volatility24h > 0.05) {
    tip = Math.floor(tip * 2);
  }
  if (volatility24h > 0.10) {
    tip = Math.floor(tip * 3);
  }
  
  // Increase tip for larger trades (more MEV opportunity)
  if (tradeSizeUsd > 5000) {
    tip = Math.floor(tip * 1.5);
  }
  if (tradeSizeUsd > 10000) {
    tip = Math.floor(tip * 2);
  }
  
  // Cap at max
  return Math.min(tip, config.maxTipLamports);
}

/**
 * Create tip transaction for Jito bundle
 */
async function createTipTransaction(
  connection: Connection,
  payer: Keypair,
  tipLamports: number,
  jitoClient: JitoJsonRpcClient
): Promise<VersionedTransaction> {
  const tipAccount = await getRandomTipAccount(jitoClient);
  const latestBlockhash = await connection.getLatestBlockhash();

  const tipInstruction = SystemProgram.transfer({
    fromPubkey: payer.publicKey,
    toPubkey: tipAccount,
    lamports: tipLamports,
  });

  const messageV0 = new TransactionMessage({
    payerKey: payer.publicKey,
    recentBlockhash: latestBlockhash.blockhash,
    instructions: [tipInstruction],
  }).compileToV0Message();

  const tipTx = new VersionedTransaction(messageV0);
  tipTx.sign([payer]);

  return tipTx;
}

/**
 * Send transaction with Jito MEV protection
 * 
 * @param connection - Solana connection
 * @param keypair - Wallet keypair for signing
 * @param transaction - The swap transaction to protect
 * @param options - Jito configuration options
 */
export async function sendWithJito(
  connection: Connection,
  keypair: Keypair,
  transaction: VersionedTransaction,
  options: Partial<JitoConfig> = {}
): Promise<JitoResult> {
  const config = { ...DEFAULT_CONFIG, ...options };
  
  logger.info('Preparing Jito bundle submission', {
    tipLamports: config.tipLamports,
    network: config.network,
    fallbackEnabled: config.fallbackToRpc,
  });
  
  try {
    // Create Jito RPC client
    const jitoClient = getJitoClient(config.network);

    // Create tip transaction
    const tipTx = await createTipTransaction(connection, keypair, config.tipLamports, jitoClient);

    // Serialize transactions to base64 for bundle
    const swapTxBase64 = Buffer.from(transaction.serialize()).toString('base64');
    const tipTxBase64 = Buffer.from(tipTx.serialize()).toString('base64');

    logger.info('Submitting bundle to Jito', {
      transactionCount: 2,
      network: config.network,
    });

    // Submit bundle - SDK expects tuple: [transactions[], encoding?]
    const bundleResult = await jitoClient.sendBundle(
      [[swapTxBase64, tipTxBase64], { encoding: 'base64' }]
    );

    if (bundleResult.error) {
      throw new Error(`Bundle submission failed: ${JSON.stringify(bundleResult.error)}`);
    }

    const bundleId = bundleResult.result as string;
    logger.info('Bundle submitted successfully', { bundleId });

    // Wait for confirmation (30 second timeout)
    const confirmation = await jitoClient.confirmInflightBundle(bundleId, 30000);

    // Check if bundle landed successfully
    const status = 'status' in confirmation ? confirmation.status : null;

    if (status === 'Landed') {
      // Get transaction signature from confirmation
      const txs = 'transactions' in confirmation ? confirmation.transactions : undefined;
      const signature = Array.isArray(txs) && txs.length > 0 ? txs[0] : undefined;
      const slot = 'landed_slot' in confirmation ? confirmation.landed_slot :
                   'slot' in confirmation ? confirmation.slot : undefined;

      logger.info('Jito bundle landed', {
        bundleId,
        signature,
        slot,
      });

      return {
        success: true,
        signature,
        bundleId,
        usedJito: true,
        tipPaid: config.tipLamports,
      };
    }

    throw new Error(`Bundle failed: ${status || 'unknown'}`);

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    logger.warn('Jito bundle submission failed', { error: errorMessage });

    // Fallback to normal RPC
    if (config.fallbackToRpc) {
      logger.info('Falling back to normal RPC submission');

      try {
        const signature = await connection.sendTransaction(transaction, {
          skipPreflight: false,
          maxRetries: 3,
          preflightCommitment: 'confirmed',
        });

        const latestBlockhash = await connection.getLatestBlockhash();
        await connection.confirmTransaction({
          signature,
          blockhash: latestBlockhash.blockhash,
          lastValidBlockHeight: latestBlockhash.lastValidBlockHeight,
        }, 'confirmed');

        logger.info('Transaction confirmed via fallback RPC', { signature });

        return {
          success: true,
          signature,
          usedJito: false,
          error: `Jito failed (${errorMessage}), used RPC fallback`,
        };
      } catch (fallbackError) {
        const fallbackMessage = fallbackError instanceof Error
          ? fallbackError.message
          : 'Unknown error';

        logger.error('Fallback RPC also failed', { error: fallbackMessage });

        return {
          success: false,
          usedJito: false,
          error: `Both Jito and RPC failed: ${fallbackMessage}`,
        };
      }
    }

    return {
      success: false,
      usedJito: false,
      error: errorMessage,
    };
  }
}

/**
 * Check if Jito is available
 */
export async function checkJitoHealth(network: 'mainnet' | 'devnet' = 'mainnet'): Promise<boolean> {
  try {
    const endpoint = JITO_ENDPOINTS[network];
    const response = await fetch(`${endpoint}/health`);
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Get current Jito tip statistics from the Jito bundle tip API
 */
export async function getJitoTipStats(): Promise<{
  minTip: number;
  medianTip: number;
  maxTip: number;
} | null> {
  try {
    const resp = await resilientFetch(
      'https://bundles.jito.wtf/api/v1/bundles/tip_floor',
      undefined,
      { label: 'jito/tip_floor', retries: 2, fetchTimeout: 10000 },
    );
    const data = await resp.json() as Array<{
      landed_tips_25th_percentile: number;
      landed_tips_50th_percentile: number;
      landed_tips_75th_percentile: number;
      landed_tips_95th_percentile: number;
      landed_tips_99th_percentile: number;
    }>;

    if (!data || data.length === 0) {
      return null;
    }

    // Jito returns values in SOL — convert to lamports
    const stats = data[0];
    return {
      minTip: Math.floor(stats.landed_tips_25th_percentile * 1e9),
      medianTip: Math.floor(stats.landed_tips_50th_percentile * 1e9),
      maxTip: Math.floor(stats.landed_tips_95th_percentile * 1e9),
    };
  } catch (e) {
    logger.warn('[Jito] Failed to fetch tip stats', { error: String(e) });
    // Return safe defaults if API is down
    return {
      minTip: 1000,
      medianTip: 10000,
      maxTip: 100000,
    };
  }
}

export { DEFAULT_CONFIG as JITO_DEFAULT_CONFIG };

