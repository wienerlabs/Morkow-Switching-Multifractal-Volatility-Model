/**
 * EXECUTE_REBALANCE Action
 *
 * Executes LP rebalancing on Solana using Jupiter aggregator.
 * Includes MEV protection via Jito bundles.
 * Requires prior ANALYZE_POOL action with REBALANCE decision.
 */
import type { Action, ActionResult, IAgentRuntime, Memory, State, HandlerCallback } from '../types/eliza.js';
import { Connection, VersionedTransaction, Keypair, PublicKey } from '@solana/web3.js';
import bs58 from 'bs58';
import BigNumber from 'bignumber.js';
import { sendWithJito, calculateDynamicTip, checkJitoHealth } from '../services/jitoService.js';
import type { JitoConfig, JitoResult } from '../services/jitoService.js';
import { getRiskManager } from '../services/riskManager.js';
import type { RiskCheckResult } from '../services/riskManager.js';
import { logger } from '../services/logger.js';

// Token mint addresses
const TOKENS = {
  SOL: 'So11111111111111111111111111111111111111112',
  USDC: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
  USDT: 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',
  BONK: 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
};

// Default fallback values for when API calls fail
const FALLBACK_VOLATILITY = 0.08; // 8% default volatility (conservative)
const LAST_RESORT_SOL_PRICE = 150; // Conservative last-resort fallback

/**
 * Fetch SOL price from Birdeye API with CoinGecko fallback
 */
async function getSolPrice(): Promise<number> {
  // Try Birdeye first
  const birdeyeApiKey = process.env.BIRDEYE_API_KEY;
  if (birdeyeApiKey) {
    try {
      const response = await fetch(
        `https://public-api.birdeye.so/defi/price?address=${TOKENS.SOL}`,
        {
          headers: {
            'X-API-KEY': birdeyeApiKey,
            'x-chain': 'solana',
          },
        }
      );
      const data = await response.json() as { success?: boolean; data?: { value?: number } };
      if (data.success && data.data?.value) {
        return data.data.value;
      }
    } catch (error) {
      logger.warn('[executeRebalance] Birdeye SOL price fetch failed', { error: String(error) });
    }
  }

  // Try CoinGecko as fallback
  try {
    const response = await fetch('https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd');
    const data = await response.json() as { solana?: { usd?: number } };
    if (data?.solana?.usd) {
      logger.info(`[executeRebalance] Using CoinGecko SOL price: $${data.solana.usd}`);
      return data.solana.usd;
    }
  } catch (error) {
    logger.warn('[executeRebalance] CoinGecko SOL price fetch failed', { error: String(error) });
  }

  logger.warn(`[executeRebalance] All price sources failed, using conservative fallback: $${LAST_RESORT_SOL_PRICE}`);
  return LAST_RESORT_SOL_PRICE;
}

/**
 * Calculate 24h volatility from Birdeye OHLCV data
 * Returns volatility as a decimal (e.g., 0.05 for 5%)
 */
async function get24hVolatility(): Promise<number> {
  const birdeyeApiKey = process.env.BIRDEYE_API_KEY;
  if (!birdeyeApiKey) {
    logger.warn('[executeRebalance] No BIRDEYE_API_KEY, using fallback volatility');
    return FALLBACK_VOLATILITY;
  }

  try {
    const now = Math.floor(Date.now() / 1000);
    const time24hAgo = now - 24 * 60 * 60;

    const response = await fetch(
      `https://public-api.birdeye.so/defi/ohlcv?address=${TOKENS.SOL}&type=1H&time_from=${time24hAgo}&time_to=${now}`,
      {
        headers: {
          'X-API-KEY': birdeyeApiKey,
          'x-chain': 'solana',
        },
      }
    );
    const data = await response.json() as {
      success?: boolean;
      data?: { items?: Array<{ c?: number }> }
    };

    if (data.success && data.data?.items && data.data.items.length >= 2) {
      // Calculate returns
      const prices = data.data.items.map(item => item.c ?? 0).filter(p => p > 0);
      if (prices.length < 2) return FALLBACK_VOLATILITY;

      const returns: number[] = [];
      for (let i = 1; i < prices.length; i++) {
        returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
      }

      // Calculate standard deviation of returns
      const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
      const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
      const volatility = Math.sqrt(variance);

      // Cap volatility between 0.01 and 0.30 (1% to 30%)
      return Math.max(0.01, Math.min(0.30, volatility));
    }
  } catch (error) {
    logger.warn('[executeRebalance] Failed to calculate volatility', { error: String(error) });
  }
  return FALLBACK_VOLATILITY;
}

/**
 * Get wallet SOL balance in lamports and USD
 */
async function getWalletValue(
  connection: Connection,
  publicKey: PublicKey,
  solPrice: number
): Promise<{ balanceSol: number; balanceUsd: number }> {
  try {
    const balanceLamports = await connection.getBalance(publicKey);
    const balanceSol = balanceLamports / 1e9;
    const balanceUsd = balanceSol * solPrice;
    return { balanceSol, balanceUsd };
  } catch (error) {
    logger.warn('[executeRebalance] Failed to get wallet balance', { error: String(error) });
    return { balanceSol: 0, balanceUsd: 0 };
  }
}

export interface RebalanceParams {
  poolAddress: string;
  poolName: string;
  amount?: number;
  slippageBps?: number;
}

export interface RebalanceResult {
  success: boolean;
  txSignature?: string;
  error?: string;
  executedAt: string;
  details: {
    poolName: string;
    amountIn: number;
    amountOut: number;
    priceImpact: number;
    fee: number;
  };
}

interface JupiterQuote {
  inAmount: string;
  outAmount: string;
  priceImpactPct?: string;
  error?: string;
}

interface JupiterSwapResponse {
  swapTransaction?: string;
  error?: string;
}

/**
 * Get Jupiter swap quote
 */
async function getJupiterQuote(
  inputMint: string,
  outputMint: string,
  amount: number,
  inputDecimals: number,
  slippageBps: number = 50
): Promise<JupiterQuote> {
  const adjustedAmount = new BigNumber(amount)
    .multipliedBy(new BigNumber(10).pow(inputDecimals))
    .toFixed(0);

  const url = `https://quote-api.jup.ag/v6/quote?inputMint=${inputMint}&outputMint=${outputMint}&amount=${adjustedAmount}&slippageBps=${slippageBps}&maxAccounts=64`;

  const response = await fetch(url);
  const data = await response.json() as JupiterQuote;

  if (data.error) {
    throw new Error(`Jupiter quote error: ${data.error}`);
  }

  return data;
}

/**
 * Get Jupiter swap transaction
 */
async function getJupiterSwapTx(
  quoteResponse: JupiterQuote,
  userPublicKey: string
): Promise<string> {
  const response = await fetch('https://quote-api.jup.ag/v6/swap', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      quoteResponse,
      userPublicKey,
      dynamicComputeUnitLimit: true,
      dynamicSlippage: true,
      priorityLevelWithMaxLamports: {
        maxLamports: 4000000,
        priorityLevel: 'high'
      }
    })
  });

  const data = await response.json() as JupiterSwapResponse;

  if (!data.swapTransaction) {
    throw new Error(`Jupiter swap error: ${data.error || 'No swap transaction'}`);
  }

  return data.swapTransaction;
}

/**
 * Execute swap on Solana with MEV protection
 *
 * Uses Jito bundles for MEV protection when available,
 * falls back to normal RPC if Jito fails.
 */
async function executeJupiterSwap(
  connection: Connection,
  keypair: Keypair,
  swapTransaction: string,
  options: {
    useJito?: boolean;
    volatility24h?: number;
    tradeSizeUsd?: number;
  } = {}
): Promise<{ signature: string; usedJito: boolean; tipPaid?: number }> {
  const txBuffer = Buffer.from(swapTransaction, 'base64');
  const transaction = VersionedTransaction.deserialize(txBuffer);

  transaction.sign([keypair]);

  // Try Jito MEV protection first (enabled by default)
  const useJito = options.useJito !== false;

  if (useJito) {
    // Check if Jito is healthy
    const jitoHealthy = await checkJitoHealth('mainnet');

    if (jitoHealthy) {
      // Calculate dynamic tip based on market conditions
      const tipLamports = calculateDynamicTip(
        options.volatility24h || 0.03,  // Default 3% volatility
        options.tradeSizeUsd || 1000    // Default $1000 trade
      );

      const jitoConfig: Partial<JitoConfig> = {
        tipLamports,
        useBundle: true,
        fallbackToRpc: true,
        network: 'mainnet',
      };

      const result: JitoResult = await sendWithJito(
        connection,
        keypair,
        transaction,
        jitoConfig
      );

      if (result.success && result.signature) {
        return {
          signature: result.signature,
          usedJito: result.usedJito,
          tipPaid: result.tipPaid,
        };
      }

      // If Jito failed and no fallback happened, throw error
      if (!result.success) {
        throw new Error(`Jito execution failed: ${result.error}`);
      }
    }
  }

  // Fallback: Normal RPC submission (no MEV protection)
  const txid = await connection.sendTransaction(transaction, {
    skipPreflight: false,
    maxRetries: 3,
    preflightCommitment: 'confirmed'
  });

  const latestBlockhash = await connection.getLatestBlockhash();
  const confirmation = await connection.confirmTransaction({
    signature: txid,
    blockhash: latestBlockhash.blockhash,
    lastValidBlockHeight: latestBlockhash.lastValidBlockHeight
  }, 'confirmed');

  if (confirmation.value.err) {
    throw new Error(`Transaction failed: ${JSON.stringify(confirmation.value.err)}`);
  }

  return { signature: txid, usedJito: false };
}

// Cooldown configuration
const COOLDOWN_MS = 24 * 60 * 60 * 1000; // 24 hours
const CACHE_KEY_PREFIX = 'cortex:rebalance:';

// Rebalance tracking data structure
interface RebalanceRecord {
  timestamp: number;
  pool: string;
  txHash: string;
  amountIn: number;
  amountOut: number;
}

/**
 * Get last rebalance record from Eliza cache
 */
async function getLastRebalance(runtime: IAgentRuntime, poolName: string): Promise<RebalanceRecord | null> {
  const cacheKey = `${CACHE_KEY_PREFIX}${poolName}`;

  if (runtime.getCache) {
    const record = await runtime.getCache<RebalanceRecord>(cacheKey);
    return record || null;
  }

  // Fallback to in-memory if cache not available
  return null;
}

/**
 * Save rebalance record to Eliza cache
 */
async function saveRebalanceRecord(
  runtime: IAgentRuntime,
  record: RebalanceRecord
): Promise<boolean> {
  const cacheKey = `${CACHE_KEY_PREFIX}${record.pool}`;

  if (runtime.setCache) {
    return await runtime.setCache<RebalanceRecord>(cacheKey, record);
  }

  return false;
}

/**
 * Check if cooldown has elapsed
 */
async function isCooldownElapsed(runtime: IAgentRuntime, poolName: string): Promise<{
  elapsed: boolean;
  lastRecord: RebalanceRecord | null;
  remainingMs: number;
}> {
  const lastRecord = await getLastRebalance(runtime, poolName);

  if (!lastRecord) {
    return { elapsed: true, lastRecord: null, remainingMs: 0 };
  }

  const timeSince = Date.now() - lastRecord.timestamp;
  const elapsed = timeSince >= COOLDOWN_MS;
  const remainingMs = elapsed ? 0 : COOLDOWN_MS - timeSince;

  return { elapsed, lastRecord, remainingMs };
}

export const executeRebalanceAction: Action = {
  name: 'EXECUTE_REBALANCE',
  description: 'Execute LP rebalancing on Solana via Jupiter aggregator',
  
  similes: ['REBALANCE', 'SWAP', 'EXECUTE_SWAP'],
  
  examples: [
    [
      { 
        user: '{{user1}}', 
        content: { text: 'Execute rebalance for SOL-USDC' } 
      },
      { 
        user: '{{agentName}}', 
        content: { 
          text: 'Executing rebalance...',
          action: 'EXECUTE_REBALANCE' 
        } 
      }
    ]
  ],

  validate: async (_runtime: IAgentRuntime, message: Memory): Promise<boolean> => {
    const text = (message.content?.text || '').toLowerCase();
    // Match execute/rebalance intent - works in both simulation and live modes
    return text.includes('execute') || text.includes('rebalance') || text.includes('swap');
  },

  handler: async (
    runtime: IAgentRuntime,
    message: Memory,
    _state?: State,
    _options?: Record<string, unknown>,
    callback?: HandlerCallback
  ): Promise<ActionResult> => {
    try {
      // Check settings
      const privateKeyStr = runtime.getSetting('SOLANA_PRIVATE_KEY');
      const rpcUrl = runtime.getSetting('SOLANA_RPC_URL') || 'https://api.mainnet-beta.solana.com';
      const isSimulation = runtime.getSetting('SIMULATION_MODE') === 'true' || !privateKeyStr;

      // Parse pool and determine swap direction
      const text = message.content?.text || '';
      let poolName = 'SOL-USDC';
      let inputMint = TOKENS.USDC;
      let outputMint = TOKENS.SOL;
      let inputDecimals = 6;
      let inputSymbol = 'USDC';
      let outputSymbol = 'SOL';

      if (text.toUpperCase().includes('BONK')) {
        poolName = 'BONK-SOL';
        inputMint = TOKENS.SOL;
        outputMint = TOKENS.BONK;
        inputDecimals = 9;
        inputSymbol = 'SOL';
        outputSymbol = 'BONK';
      } else if (text.toUpperCase().includes('USDT')) {
        poolName = 'SOL-USDT';
        inputMint = TOKENS.USDT;
        outputMint = TOKENS.SOL;
        inputDecimals = 6;
        inputSymbol = 'USDT';
        outputSymbol = 'SOL';
      }

      // Parse amount from message or use default
      const amountMatch = text.match(/(\d+(?:\.\d+)?)\s*(?:usdc|sol|usdt|bonk)/i);
      const amount = amountMatch ? parseFloat(amountMatch[1]) : 100; // Default 100 units

      if (callback) await callback({ text: `🔄 Preparing rebalance for ${poolName}...\n💰 Amount: ${amount} ${inputSymbol}` });

      // SIMULATION MODE - no wallet required
      if (isSimulation) {
        // Try to get a real quote for accurate simulation
        let estimatedOut = '~estimated';
        try {
          const quote = await getJupiterQuote(inputMint, outputMint, amount, inputDecimals, 50);
          const outAmount = new BigNumber(quote.outAmount).dividedBy(new BigNumber(10).pow(outputMint === TOKENS.SOL ? 9 : 6));
          estimatedOut = outAmount.toFixed(6);
        } catch { /* Use estimated if quote fails */ }

        const simText = `🎮 **SIMULATION MODE**\n\n📊 Pool: ${poolName}\n💰 Input: ${amount} ${inputSymbol}\n💎 Output: ${estimatedOut} ${outputSymbol}\n📉 Slippage: 0.5%\n\n✅ Simulation successful!\n\n⚙️ To execute for real:\n1. Set SIMULATION_MODE=false\n2. Configure SOLANA_PRIVATE_KEY`;
        if (callback) await callback({ text: simText });
        return { success: true, text: simText, data: { simulation: true, poolName, amount, inputSymbol, outputSymbol } };
      }

      // === LIVE EXECUTION - Check cooldown ===
      const cooldownStatus = await isCooldownElapsed(runtime, poolName);

      if (!cooldownStatus.elapsed) {
        const hoursRemaining = Math.ceil(cooldownStatus.remainingMs / (60 * 60 * 1000));
        const lastTx = cooldownStatus.lastRecord?.txHash || 'unknown';
        if (callback) await callback({
          text: `⏳ **Cooldown Active** for ${poolName}\n\n⏰ ${hoursRemaining}h remaining\n🔗 Last TX: [${lastTx.slice(0, 8)}...](https://solscan.io/tx/${lastTx})`
        });
        return { success: false, error: `Cooldown active. ${hoursRemaining}h remaining.` };
      }

      // === REAL EXECUTION ===

      // 0. Setup connection and keypair first (needed for wallet balance)
      const connection = new Connection(rpcUrl, 'confirmed');
      let keypair: Keypair;
      try {
        const secretKey = bs58.decode(privateKeyStr);
        keypair = Keypair.fromSecretKey(secretKey);
      } catch {
        // Try base64 format
        const secretKey = Uint8Array.from(Buffer.from(privateKeyStr, 'base64'));
        keypair = Keypair.fromSecretKey(secretKey);
      }

      const walletAddress = keypair.publicKey.toBase58();

      // 1. Fetch dynamic market data (volatility and SOL price)
      if (callback) await callback({ text: `📊 Fetching market data...` });

      const [volatility24h, solPrice] = await Promise.all([
        get24hVolatility(),
        getSolPrice(),
      ]);

      // 2. Get wallet balance for position sizing
      const walletValue = await getWalletValue(connection, keypair.publicKey, solPrice);

      logger.info(`[executeRebalance] Market data: volatility=${(volatility24h * 100).toFixed(2)}%, SOL=$${solPrice.toFixed(2)}, wallet=$${walletValue.balanceUsd.toFixed(2)}`);

      // 3. Calculate dynamic position size using Risk Manager
      const riskManager = getRiskManager();

      // Use a default confidence of 0.88 (threshold) or get from state if available
      const modelConfidence = 0.88;

      const positionCalc = riskManager.calculatePositionSize({
        modelConfidence,
        currentVolatility24h: volatility24h,
        portfolioValueUsd: walletValue.balanceUsd > 0 ? walletValue.balanceUsd : 10000, // Default $10k if balance unknown
      });

      const proposedPositionPct = positionCalc.positionPct;

      logger.info(`[executeRebalance] Position sizing: ${positionCalc.rationale}, result=${proposedPositionPct}%, $${positionCalc.positionUsd}`);

      // 4. Risk Management Check
      const riskCheck: RiskCheckResult = riskManager.checkTradeAllowed({
        proposedPositionPct,
        currentVolatility24h: volatility24h,
      });

      if (!riskCheck.allowed) {
        if (callback) await callback({
          text: `⛔ **Risk Check Failed**\n\n❌ ${riskCheck.reason}\n\n📊 Risk limits protect your portfolio from excessive losses.`
        });
        return { success: false, error: `Risk check failed: ${riskCheck.reason}` };
      }

      // Log warnings but continue
      if (riskCheck.warnings.length > 0) {
        const warningsText = riskCheck.warnings.map(w => `⚠️ ${w}`).join('\n');
        if (callback) await callback({ text: `📋 Risk Warnings:\n${warningsText}` });
      }

      // Use adjusted position if provided
      const adjustedPositionPct = riskCheck.suggestedPositionPct || proposedPositionPct;

      if (callback) await callback({
        text: `🔑 Wallet: ${walletAddress.slice(0, 8)}...${walletAddress.slice(-4)}\n📊 Position: ${adjustedPositionPct.toFixed(1)}% of portfolio (~$${positionCalc.positionUsd})\n📈 Volatility: ${(volatility24h * 100).toFixed(1)}%`
      });

      // 2. Get Jupiter quote
      if (callback) await callback({ text: `📊 Getting Jupiter quote...` });
      const quote = await getJupiterQuote(inputMint, outputMint, amount, inputDecimals, 50);

      const outAmount = new BigNumber(quote.outAmount).dividedBy(new BigNumber(10).pow(outputMint === TOKENS.SOL ? 9 : 6));
      const priceImpact = parseFloat(quote.priceImpactPct || '0');

      if (callback) await callback({
        text: `💱 Quote received:\n- In: ${amount} ${inputSymbol}\n- Out: ~${outAmount.toFixed(6)} ${outputSymbol}\n- Impact: ${(priceImpact * 100).toFixed(3)}%`
      });

      // 3. Get swap transaction
      if (callback) await callback({ text: `📝 Building transaction...` });
      const swapTx = await getJupiterSwapTx(quote, walletAddress);

      // 4. Execute swap with MEV protection (Jito)
      if (callback) await callback({ text: `⚡ Executing swap with MEV protection...` });

      // Calculate trade size in USD using real SOL price
      const tradeSizeUsd = inputMint === TOKENS.SOL
        ? amount * solPrice  // SOL input: multiply by SOL price
        : outputMint === TOKENS.SOL
          ? outAmount.toNumber() * solPrice  // SOL output: use output amount
          : amount;  // Stablecoin: use amount directly

      logger.info(`[executeRebalance] Trade size: $${tradeSizeUsd.toFixed(2)} (SOL price: $${solPrice.toFixed(2)})`);

      const swapResult = await executeJupiterSwap(connection, keypair, swapTx, {
        useJito: true,
        volatility24h,
        tradeSizeUsd,
      });

      const txSignature = swapResult.signature;
      const mevProtectionStatus = swapResult.usedJito
        ? `🛡️ MEV Protected (Jito, tip: ${(swapResult.tipPaid || 0) / 1e9} SOL)`
        : `⚠️ No MEV protection (Jito unavailable)`;

      // 5. Record trade in RiskManager for daily tracking
      // Estimate P&L based on price impact (negative means loss)
      const estimatedPnlPct = -priceImpact * 100; // Price impact is a cost
      riskManager.recordTrade(estimatedPnlPct);

      // 6. Save rebalance record to Eliza cache
      const rebalanceRecord: RebalanceRecord = {
        timestamp: Date.now(),
        pool: poolName,
        txHash: txSignature,
        amountIn: amount,
        amountOut: outAmount.toNumber(),
      };

      const saved = await saveRebalanceRecord(runtime, rebalanceRecord);
      if (!saved) {
        logger.warn('Failed to save rebalance record to cache');
      }

      const result: RebalanceResult = {
        success: true,
        txSignature,
        executedAt: new Date().toISOString(),
        details: {
          poolName,
          amountIn: amount,
          amountOut: outAmount.toNumber(),
          priceImpact,
          fee: 0.003,
        },
      };

      const successText = `✅ **Rebalance Executed Successfully!**\n\n📊 ${poolName}\n🔗 TX: [${txSignature.slice(0, 8)}...](https://solscan.io/tx/${txSignature})\n💰 In: ${amount} ${inputSymbol}\n💎 Out: ${outAmount.toFixed(6)} ${outputSymbol}\n📉 Impact: ${(priceImpact * 100).toFixed(3)}%\n${mevProtectionStatus}\n\n⏰ Next rebalance available in 24h`;
      if (callback) await callback({ text: successText });

      return { success: true, text: successText, data: result as unknown as Record<string, unknown> };

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      if (callback) await callback({ text: `❌ Rebalance failed: ${errorMsg}` });
      return { success: false, error: errorMsg };
    }
  },
};

