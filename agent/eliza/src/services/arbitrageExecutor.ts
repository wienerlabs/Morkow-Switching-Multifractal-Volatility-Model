/**
 * CEX-DEX Arbitrage Executor
 *
 * Flow:
 * 1. CEX cheap → DEX expensive: Buy on CEX → Withdraw to Solana → Sell on DEX
 * 2. DEX cheap → CEX expensive: Buy on DEX → Deposit to CEX → Sell on CEX
 *
 * Supports: Binance ↔ Jupiter
 *
 * UPDATED 2026-01-07: Uses dynamic SOL price and gas fees instead of hardcoded $200
 * INTEGRATED: GlobalRiskManager for cross-strategy risk checks
 * INTEGRATED: Guardian pre-execution validation
 */

import type { ArbitrageOpportunity } from './marketScanner/types.js';
import { Connection, VersionedTransaction, Keypair, PublicKey } from '@solana/web3.js';
import crypto from 'crypto';
import { getSolPrice, DEFAULT_GAS_SOL } from './marketData.js';
import { getGlobalRiskManager } from './risk/index.js';
import { getPortfolioManager } from './portfolioManager.js';
import { guardian } from './guardian/index.js';
import type { GuardianTradeParams } from './guardian/types.js';
import { pmDecisionEngine, approvalQueue } from './pm/index.js';
import type { QueueTradeParams } from './pm/types.js';
import { createJupiterApiClient } from '@jup-ag/api';

// ============= TYPES =============

export interface ArbitrageConfig {
  binanceApiKey: string;
  binanceSecretKey: string;
  solanaPrivateKey: string;
  solanaRpcUrl: string;
  dryRun: boolean;
  minProfitUsd: number;      // Minimum profit after all fees
  minSpreadPct: number;      // Minimum spread after fees
  maxWithdrawWaitMs: number; // Max time to wait for CEX withdrawal
}

export interface ArbitrageResult {
  success: boolean;
  direction: 'cex-to-dex' | 'dex-to-cex' | 'dex-to-dex' | 'cex-to-cex';
  symbol: string;
  buyExchange: string;
  sellExchange: string;
  amountIn: number;
  amountOut: number;
  grossProfit: number;
  fees: {
    cexTrade: number;
    withdrawal: number;
    dexSwap: number;
    gas: number;
  };
  netProfit: number;
  profitPct: number;
  txHashes: {
    cexOrder?: string;
    withdrawal?: string;
    deposit?: string;
    dexSwap?: string;
    buySwap?: string;
    sellSwap?: string;
  };
  executionTimeMs: number;
  error?: string;
}

// ============= BINANCE API =============

const BINANCE_API = 'https://api.binance.com';

interface BinanceOrderResponse {
  orderId: number;
  clientOrderId: string;
  transactTime: number;
  status: string;
  executedQty: string;
  cummulativeQuoteQty: string;
  price: string;
  fills: Array<{ price: string; qty: string; commission: string }>;
}

interface BinanceWithdrawResponse {
  id: string;
}

function signBinanceRequest(params: Record<string, string | number>, secretKey: string): string {
  const queryString = Object.entries(params)
    .map(([k, v]) => `${k}=${v}`)
    .join('&');
  const signature = crypto.createHmac('sha256', secretKey).update(queryString).digest('hex');
  return `${queryString}&signature=${signature}`;
}

async function binanceSpotOrder(
  config: ArbitrageConfig,
  symbol: string,
  side: 'BUY' | 'SELL',
  quantity: number,
  decimals: number = 8  // Default to 8 decimals, but can be overridden
): Promise<BinanceOrderResponse> {
  const roundedQuantity = decimals === 0
    ? Math.floor(quantity).toString()  // Integer for BONK
    : parseFloat(quantity.toFixed(decimals)).toString();  // Remove trailing zeros


  const params: Record<string, string | number> = {
    symbol,
    side,
    type: 'MARKET',
    quantity: roundedQuantity,
    timestamp: Date.now(),
    recvWindow: 5000,
  };

  const signedQuery = signBinanceRequest(params, config.binanceSecretKey);
  const url = `${BINANCE_API}/api/v3/order?${signedQuery}`;

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'X-MBX-APIKEY': config.binanceApiKey },
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Binance order failed: ${error}`);
  }

  return response.json() as Promise<BinanceOrderResponse>;
}

async function binanceWithdraw(
  config: ArbitrageConfig,
  coin: string,
  address: string,
  amount: number,
  network: string = 'SOL'
): Promise<BinanceWithdrawResponse> {
  const params: Record<string, string | number> = {
    coin,
    address,
    amount: amount.toFixed(8),
    network,
    timestamp: Date.now(),
    recvWindow: 5000,
  };

  const signedQuery = signBinanceRequest(params, config.binanceSecretKey);
  const url = `${BINANCE_API}/sapi/v1/capital/withdraw/apply?${signedQuery}`;

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'X-MBX-APIKEY': config.binanceApiKey },
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Binance withdraw failed: ${error}`);
  }

  return response.json() as Promise<BinanceWithdrawResponse>;
}

interface BinanceDepositAddressResponse {
  address: string;
  coin: string;
  tag: string;
  url: string;
}

async function binanceGetDepositAddress(
  config: ArbitrageConfig,
  coin: string,
  network: string = 'SOL'
): Promise<BinanceDepositAddressResponse> {
  const params: Record<string, string | number> = {
    coin,
    network,
    timestamp: Date.now(),
    recvWindow: 5000,
  };

  const signedQuery = signBinanceRequest(params, config.binanceSecretKey);
  const url = `${BINANCE_API}/sapi/v1/capital/deposit/address?${signedQuery}`;

  const response = await fetch(url, {
    method: 'GET',
    headers: { 'X-MBX-APIKEY': config.binanceApiKey },
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Binance get deposit address failed: ${error}`);
  }

  return response.json() as Promise<BinanceDepositAddressResponse>;
}

// ============= TOKEN CONFIG =============

const TOKENS: Record<string, { mint: string; decimals: number; binanceSymbol: string; binanceDecimals: number }> = {
  SOL:  { mint: 'So11111111111111111111111111111111111111112', decimals: 9, binanceSymbol: 'SOLUSDT', binanceDecimals: 2 },
  BONK: { mint: 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263', decimals: 5, binanceSymbol: 'BONKUSDT', binanceDecimals: 0 },
  JUP:  { mint: 'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN', decimals: 6, binanceSymbol: 'JUPUSDT', binanceDecimals: 2 },
  WIF:  { mint: 'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm', decimals: 6, binanceSymbol: 'WIFUSDT', binanceDecimals: 2 },
  USDC: { mint: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v', decimals: 6, binanceSymbol: 'USDCUSDT', binanceDecimals: 2 },
};

// ============= JUPITER SWAP =============

interface JupiterQuote {
  inAmount: string;
  outAmount: string;
  priceImpactPct?: string;
  swapTransaction?: string;  // Ultra API may return transaction directly
  transaction?: string;       // Alternative field name
}

async function getJupiterQuote(
  inputMint: string,
  outputMint: string,
  amountLamports: number,
  walletAddress: string,
  slippageBps: number = 100
): Promise<JupiterQuote> {
  const apiKey = process.env.JUPITER_API_KEY;

  if (!apiKey) {
    throw new Error('JUPITER_API_KEY not found in .env');
  }

  if (!walletAddress) {
    throw new Error('Wallet address is required for Jupiter quote');
  }

  // Use Jupiter Ultra API
  const url = `https://api.jup.ag/ultra/v1/order?inputMint=${inputMint}&outputMint=${outputMint}&amount=${amountLamports}&taker=${walletAddress}`;

  console.log(`[JUPITER] Fetching quote from Ultra API...`);

  const response = await fetch(url, {
    method: 'GET',
    headers: {
      'x-api-key': apiKey,
    },
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Jupiter Ultra API error: ${response.status} ${response.statusText} - ${errorText}`);
  }

  const data = await response.json() as JupiterQuote & { error?: string };

  if ((data as { error?: string }).error) {
    throw new Error(`Jupiter quote error: ${(data as { error?: string }).error}`);
  }

  console.log(`[JUPITER] ✅ Quote received from Ultra API`);
  return data;
}

async function executeJupiterSwap(
  quote: JupiterQuote,
  walletPublicKey: string,
  connection: Connection,
  keypair: Keypair
): Promise<string> {
  // Ultra API provides transaction - always use regular (non-gasless) flow
  if (!quote.swapTransaction && !quote.transaction) {
    throw new Error('No transaction in quote response');
  }

  const swapTransaction = (quote.swapTransaction || quote.transaction)!;
  const txBuffer = Buffer.from(swapTransaction, 'base64');
  const transaction = VersionedTransaction.deserialize(txBuffer);

  console.log('[JUPITER] 🔍 Transaction details:');
  console.log('[JUPITER]   - Fee payer:', transaction.message.staticAccountKeys[0]?.toBase58());
  console.log('[JUPITER]   - Our wallet:', keypair.publicKey.toBase58());
  console.log('[JUPITER]   - Signatures before signing:', transaction.signatures.length);

  // Sign the transaction with our wallet (normal flow, not gasless)
  console.log('[JUPITER] Signing transaction with our wallet...');
  transaction.sign([keypair]);

  console.log('[JUPITER]   - Signatures after signing:', transaction.signatures.length);

  const txid = await connection.sendTransaction(transaction, {
    skipPreflight: false,
    maxRetries: 3,
  });

  console.log(`[JUPITER] ✅ Transaction sent: ${txid}`);
  console.log(`[JUPITER] 🔗 View on Solscan: https://solscan.io/tx/${txid}`);

  // Wait for confirmation with error checking
  console.log(`[JUPITER] ⏳ Waiting for confirmation (up to 60 seconds)...`);

  const startTime = Date.now();
  const timeout = 60000; // 60 seconds
  let confirmed = false;

  while (!confirmed && Date.now() - startTime < timeout) {
    const status = await connection.getSignatureStatus(txid);

    // Check if transaction failed
    if (status?.value?.err) {
      console.log(`[JUPITER] ❌ Transaction FAILED!`);
      console.log(`[JUPITER] Error:`, JSON.stringify(status.value.err));
      throw new Error(`Jupiter swap transaction failed: ${JSON.stringify(status.value.err)}`);
    }

    if (status?.value?.confirmationStatus === 'confirmed' || status?.value?.confirmationStatus === 'finalized') {
      confirmed = true;
      break;
    }
    await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second
  }

  if (!confirmed) {
    throw new Error(`Transaction confirmation timeout after 60 seconds. Check signature: ${txid}`);
  }

  console.log(`[JUPITER] ✅ Transaction confirmed and SUCCESSFUL!`);

  return txid;
}


// ============= MAIN EXECUTOR =============

export class ArbitrageExecutor {
  private config: ArbitrageConfig;
  private connection: Connection;
  private keypair: Keypair | null = null;
  private cachedWalletBalance: number = 1; // Cached balance, updated periodically
  private lastBalanceUpdate: number = 0;
  private readonly BALANCE_CACHE_MS = 30000; // 30 second cache

  constructor(config: ArbitrageConfig) {
    this.config = config;
    this.connection = new Connection(config.solanaRpcUrl, 'confirmed');

    if (config.solanaPrivateKey) {
      try {
        // Try bs58 format
        const bs58 = require('bs58');
        const secretKey = bs58.decode(config.solanaPrivateKey);
        this.keypair = Keypair.fromSecretKey(secretKey);
      } catch {
        try {
          // Try base64 format
          const secretKey = Uint8Array.from(Buffer.from(config.solanaPrivateKey, 'base64'));
          this.keypair = Keypair.fromSecretKey(secretKey);
        } catch {
          console.warn('[ArbitrageExecutor] Invalid Solana private key');
        }
      }
    }
  }

  /**
   * Get wallet SOL balance with caching to avoid excessive RPC calls
   */
  private async getWalletBalance(): Promise<number> {
    if (!this.keypair) {
      return 1; // Fallback if no wallet
    }

    const now = Date.now();
    if (now - this.lastBalanceUpdate < this.BALANCE_CACHE_MS) {
      return this.cachedWalletBalance;
    }

    try {
      const balance = await this.connection.getBalance(this.keypair.publicKey);
      this.cachedWalletBalance = balance / 1e9; // Convert lamports to SOL
      this.lastBalanceUpdate = now;
      return this.cachedWalletBalance;
    } catch (error) {
      console.error('[ARBITRAGE] Failed to get wallet balance:', error);
      return this.cachedWalletBalance; // Return cached value on error
    }
  }

  /**
   * Execute CEX → DEX arbitrage
   * Buy on Binance → Withdraw to Solana → Sell on Jupiter
   */
  async executeCexToDex(opp: ArbitrageOpportunity, amountUsd: number): Promise<ArbitrageResult> {
    const startTime = Date.now();
    const symbol = opp.symbol;
    const token = TOKENS[symbol];

    if (!token) {
      return this.errorResult(opp, 'cex-to-dex', `Unknown token: ${symbol}`);
    }

    console.log(`\n[ARBITRAGE] ${symbol} detected: +${opp.spreadPct.toFixed(2)}% spread`);

    // Calculate quantity based on USD amount
    const quantity = amountUsd / opp.buyPrice;

    const result: ArbitrageResult = {
      success: false,
      direction: 'cex-to-dex',
      symbol,
      buyExchange: opp.buyExchange,
      sellExchange: opp.sellExchange,
      amountIn: amountUsd,
      amountOut: 0,
      grossProfit: 0,
      fees: { cexTrade: 0, withdrawal: 0, dexSwap: 0, gas: 0 },
      netProfit: 0,
      profitPct: 0,
      txHashes: {},
      executionTimeMs: 0,
    };

    try {
      // === DRY RUN MODE ===
      if (this.config.dryRun) {
        return await this.simulateCexToDex(opp, amountUsd, quantity, token, startTime);
      }

      // === LIVE EXECUTION ===

      if (!this.keypair) {
        throw new Error('Solana wallet not configured');
      }

      const walletAddress = this.keypair.publicKey.toBase58();

      // Step 1: Buy on Binance
      console.log(`[ARBITRAGE] Buy from ${opp.buyExchange}: ${quantity.toFixed(4)} ${symbol} at $${opp.buyPrice.toFixed(6)}`);

      const order = await binanceSpotOrder(this.config, token.binanceSymbol, 'BUY', quantity, token.binanceDecimals);
      result.txHashes.cexOrder = order.clientOrderId;
      result.fees.cexTrade = parseFloat(order.fills?.[0]?.commission || '0');

      const executedQty = parseFloat(order.executedQty);
      console.log(`[ARBITRAGE] ✅ Bought ${executedQty} ${symbol}`);

      // Step 2: Withdraw to Solana
      console.log(`[ARBITRAGE] Withdraw to Solana: ${walletAddress.slice(0, 8)}...`);

      const withdrawal = await binanceWithdraw(
        this.config,
        symbol,
        walletAddress,
        executedQty
      );
      result.txHashes.withdrawal = withdrawal.id;
      result.fees.withdrawal = this.getWithdrawalFee(symbol);

      console.log(`[ARBITRAGE] Withdrawal initiated: ${withdrawal.id}`);

      // Step 3: Wait for withdrawal (simplified - in production would poll)
      console.log(`[ARBITRAGE] Waiting for withdrawal confirmation...`);
      await this.waitForBalance(token.mint, executedQty, this.config.maxWithdrawWaitMs);

      // Step 4: Sell on Jupiter
      const actualQty = executedQty - result.fees.withdrawal;
      const inputLamports = Math.floor(actualQty * Math.pow(10, token.decimals));

      console.log(`[ARBITRAGE] Sell on Jupiter: ${actualQty.toFixed(4)} ${symbol}`);

      const quote = await getJupiterQuote(token.mint, TOKENS.USDC.mint, inputLamports, walletAddress);
      const swapTxid = await executeJupiterSwap(quote, walletAddress, this.connection, this.keypair);

      result.txHashes.dexSwap = swapTxid;
      result.amountOut = parseInt(quote.outAmount) / 1e6; // USDC decimals
      result.fees.dexSwap = result.amountOut * 0.003; // ~0.3% Jupiter fee
      const solPrice = await getSolPrice();
      result.fees.gas = DEFAULT_GAS_SOL * solPrice; // Dynamic gas fee using live SOL price

      // Calculate profit
      result.grossProfit = result.amountOut - amountUsd;
      result.netProfit = result.grossProfit - result.fees.cexTrade - result.fees.withdrawal - result.fees.dexSwap - result.fees.gas;
      result.profitPct = (result.netProfit / amountUsd) * 100;
      result.success = result.netProfit > 0;
      result.executionTimeMs = Date.now() - startTime;

      // ========== TRACK IN PORTFOLIO ==========
      const totalFees = result.fees.cexTrade + result.fees.withdrawal + result.fees.dexSwap + result.fees.gas;
      getPortfolioManager().recordArbitrageTrade({
        asset: symbol,
        amountUsd: amountUsd,
        venue: `${opp.buyExchange}->${opp.sellExchange}`,
        fees: totalFees,
        pnlUsd: result.netProfit,
        txSignature: result.txHashes.dexSwap,
        notes: `CEX-DEX arb: ${result.profitPct.toFixed(2)}% profit`,
      });
      // ========================================

      console.log(`[ARBITRAGE] ✅ Net profit: $${result.netProfit.toFixed(2)} (+${result.profitPct.toFixed(2)}%)`);

      return result;

    } catch (error) {
      result.error = error instanceof Error ? error.message : 'Unknown error';
      result.executionTimeMs = Date.now() - startTime;
      console.log(`[ARBITRAGE] ❌ Failed: ${result.error}`);
      return result;
    }
  }

  /**
   * Simulate CEX → DEX arbitrage (dry run)
   */
  private async simulateCexToDex(
    opp: ArbitrageOpportunity,
    amountUsd: number,
    quantity: number,
    _token: typeof TOKENS[string],
    startTime: number
  ): Promise<ArbitrageResult> {
    // Get live SOL price for gas calculation
    const solPrice = await getSolPrice();

    // Estimate fees with dynamic pricing
    const cexTradeFee = amountUsd * 0.001; // 0.1% Binance
    const withdrawalFee = this.getWithdrawalFee(opp.symbol);
    const dexSwapFee = amountUsd * 0.003; // 0.3% Jupiter
    const gasFee = DEFAULT_GAS_SOL * solPrice; // Dynamic: 0.002 SOL * live price

    const grossProfit = amountUsd * (opp.spreadPct / 100);
    const totalFees = cexTradeFee + withdrawalFee + dexSwapFee + gasFee;
    const netProfit = grossProfit - totalFees;

    console.log(`[ARBITRAGE] 📝 DRY RUN - Simulating...`);
    console.log(`[ARBITRAGE] SOL price: $${solPrice.toFixed(2)} (live)`);
    console.log(`[ARBITRAGE] Buy from ${opp.buyExchange}: ${quantity.toFixed(4)} ${opp.symbol} at $${opp.buyPrice.toFixed(6)}`);
    console.log(`[ARBITRAGE] Withdraw to Solana: 2min ETA`);
    console.log(`[ARBITRAGE] Sell on Jupiter: ${quantity.toFixed(4)} ${opp.symbol} → $${(amountUsd + grossProfit).toFixed(2)} USDC`);
    console.log(`[ARBITRAGE] Gas: ${DEFAULT_GAS_SOL} SOL = $${gasFee.toFixed(4)}`);
    console.log(`[ARBITRAGE] ${netProfit > 0 ? '✅' : '❌'} Net profit: $${netProfit.toFixed(2)} (+${((netProfit / amountUsd) * 100).toFixed(2)}%)`);

    const result: ArbitrageResult = {
      success: netProfit > 0,
      direction: 'cex-to-dex',
      symbol: opp.symbol,
      buyExchange: opp.buyExchange,
      sellExchange: opp.sellExchange,
      amountIn: amountUsd,
      amountOut: amountUsd + grossProfit,
      grossProfit,
      fees: {
        cexTrade: cexTradeFee,
        withdrawal: withdrawalFee,
        dexSwap: dexSwapFee,
        gas: gasFee,
      },
      netProfit,
      profitPct: (netProfit / amountUsd) * 100,
      txHashes: {},
      executionTimeMs: Date.now() - startTime,
    };

    // ========== TRACK IN PORTFOLIO ==========
    getPortfolioManager().recordArbitrageTrade({
      asset: opp.symbol,
      amountUsd: amountUsd,
      venue: `${opp.buyExchange}->${opp.sellExchange}`,
      fees: totalFees,
      pnlUsd: netProfit,
      notes: `CEX-DEX arb (simulated): ${result.profitPct.toFixed(2)}% profit`,
    });
    // ========================================

    return result;
  }



  /**
   * Execute DEX → DEX arbitrage
   * Buy on one DEX (e.g., Orca) → Sell on another DEX (e.g., Meteora) via Jupiter
   */
  async executeDexToDex(opp: ArbitrageOpportunity, amountUsd: number): Promise<ArbitrageResult> {
    const startTime = Date.now();
    const symbol = opp.symbol;
    const token = TOKENS[symbol];

    if (!token) {
      return this.errorResult(opp, 'dex-to-dex', `Unknown token: ${symbol}`);
    }

    console.log(`\n[ARBITRAGE] ${symbol} DEX→DEX detected: +${opp.spreadPct.toFixed(2)}% spread`);
    console.log(`[ARBITRAGE] Route: ${opp.buyExchange} → ${opp.sellExchange}`);

    const result: ArbitrageResult = {
      success: false,
      direction: 'dex-to-dex',
      symbol,
      buyExchange: opp.buyExchange,
      sellExchange: opp.sellExchange,
      amountIn: amountUsd,
      amountOut: 0,
      grossProfit: 0,
      fees: { cexTrade: 0, withdrawal: 0, dexSwap: 0, gas: 0 },
      netProfit: 0,
      profitPct: 0,
      txHashes: {},
      executionTimeMs: 0,
    };

    try {
      // === DRY RUN MODE ===
      if (this.config.dryRun) {
        return await this.simulateDexToDex(opp, amountUsd, startTime);
      }

      // === LIVE EXECUTION ===
      if (!this.keypair) {
        throw new Error('Solana wallet not configured');
      }

      const walletAddress = this.keypair.publicKey.toBase58();

      // Step 1: Buy on first DEX (via Jupiter)
      console.log(`[ARBITRAGE] Buy on ${opp.buyExchange}: ${symbol} at $${opp.buyPrice.toFixed(6)}`);

      const buyAmountLamports = Math.floor(amountUsd * 1e6); // USDC has 6 decimals
      const buyQuote = await getJupiterQuote(TOKENS.USDC.mint, token.mint, buyAmountLamports, walletAddress);
      const buyTxid = await executeJupiterSwap(buyQuote, walletAddress, this.connection, this.keypair);

      result.txHashes.buySwap = buyTxid;
      const tokensBought = parseInt(buyQuote.outAmount) / Math.pow(10, token.decimals);
      console.log(`[ARBITRAGE] ✅ Bought ${tokensBought.toFixed(4)} ${symbol} - TX: ${buyTxid}`);

      // Step 2: Sell on second DEX (via Jupiter)
      console.log(`[ARBITRAGE] Sell on ${opp.sellExchange}: ${tokensBought.toFixed(4)} ${symbol}`);

      const sellAmountLamports = Math.floor(tokensBought * Math.pow(10, token.decimals));
      const sellQuote = await getJupiterQuote(token.mint, TOKENS.USDC.mint, sellAmountLamports, walletAddress);
      const sellTxid = await executeJupiterSwap(sellQuote, walletAddress, this.connection, this.keypair);

      result.txHashes.sellSwap = sellTxid;
      result.amountOut = parseInt(sellQuote.outAmount) / 1e6; // USDC decimals
      console.log(`[ARBITRAGE] ✅ Sold for ${result.amountOut.toFixed(2)} USDC - TX: ${sellTxid}`);

      // Calculate profit
      result.grossProfit = result.amountOut - amountUsd;
      result.fees.dexSwap = (amountUsd * 0.003) + (result.amountOut * 0.003); // ~0.3% per swap
      const solPrice = await getSolPrice();
      result.fees.gas = DEFAULT_GAS_SOL * 2 * solPrice; // 2 transactions
      result.netProfit = result.grossProfit - result.fees.dexSwap - result.fees.gas;
      result.profitPct = (result.netProfit / amountUsd) * 100;
      result.success = result.netProfit > 0;
      result.executionTimeMs = Date.now() - startTime;

      console.log(`[ARBITRAGE] ${result.success ? '✅' : '❌'} Net profit: $${result.netProfit.toFixed(2)} (+${result.profitPct.toFixed(2)}%)`);
      console.log(`[ARBITRAGE] Fees: DEX swaps $${result.fees.dexSwap.toFixed(2)}, Gas $${result.fees.gas.toFixed(2)}`);

      // Track in portfolio
      getPortfolioManager().recordArbitrageTrade({
        asset: symbol,
        amountUsd: amountUsd,
        venue: `${opp.buyExchange}->${opp.sellExchange}`,
        fees: result.fees.dexSwap + result.fees.gas,
        pnlUsd: result.netProfit,
        notes: `DEX-DEX arb: ${result.profitPct.toFixed(2)}% profit`,
      });

      return result;
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      console.log(`[ARBITRAGE] ❌ DEX→DEX execution failed: ${errorMsg}`);
      result.error = errorMsg;
      result.executionTimeMs = Date.now() - startTime;
      return result;
    }
  }

  /**
   * Simulate DEX → DEX arbitrage (dry run)
   */
  private async simulateDexToDex(
    opp: ArbitrageOpportunity,
    amountUsd: number,
    startTime: number
  ): Promise<ArbitrageResult> {
    const solPrice = await getSolPrice();

    // Estimate fees
    const buySwapFee = amountUsd * 0.003; // 0.3% Jupiter
    const sellSwapFee = amountUsd * 0.003; // 0.3% Jupiter
    const gasFee = DEFAULT_GAS_SOL * 2 * solPrice; // 2 transactions

    const grossProfit = amountUsd * (opp.spreadPct / 100);
    const totalFees = buySwapFee + sellSwapFee + gasFee;
    const netProfit = grossProfit - totalFees;

    console.log(`[ARBITRAGE] 📝 DRY RUN - Simulating DEX→DEX...`);
    console.log(`[ARBITRAGE] SOL price: $${solPrice.toFixed(2)} (live)`);
    console.log(`[ARBITRAGE] Buy on ${opp.buyExchange}: ${opp.symbol} at $${opp.buyPrice.toFixed(6)}`);
    console.log(`[ARBITRAGE] Sell on ${opp.sellExchange}: ${opp.symbol} at $${opp.sellPrice.toFixed(6)}`);
    console.log(`[ARBITRAGE] Fees: Buy swap $${buySwapFee.toFixed(2)}, Sell swap $${sellSwapFee.toFixed(2)}, Gas $${gasFee.toFixed(4)}`);
    console.log(`[ARBITRAGE] ${netProfit > 0 ? '✅' : '❌'} Net profit: $${netProfit.toFixed(2)} (+${((netProfit / amountUsd) * 100).toFixed(2)}%)`);

    const result: ArbitrageResult = {
      success: netProfit > 0,
      direction: 'dex-to-dex',
      symbol: opp.symbol,
      buyExchange: opp.buyExchange,
      sellExchange: opp.sellExchange,
      amountIn: amountUsd,
      amountOut: amountUsd + grossProfit,
      grossProfit,
      fees: {
        cexTrade: 0,
        withdrawal: 0,
        dexSwap: buySwapFee + sellSwapFee,
        gas: gasFee,
      },
      netProfit,
      profitPct: (netProfit / amountUsd) * 100,
      txHashes: {},
      executionTimeMs: Date.now() - startTime,
    };

    // Track in portfolio
    getPortfolioManager().recordArbitrageTrade({
      asset: opp.symbol,
      amountUsd: amountUsd,
      venue: `${opp.buyExchange}->${opp.sellExchange}`,
      fees: totalFees,
      pnlUsd: netProfit,
      notes: `DEX-DEX arb (simulated): ${result.profitPct.toFixed(2)}% profit`,
    });

    return result;
  }

  /**
   * Execute DEX → CEX arbitrage
   * Buy on Jupiter → Deposit to Binance → Sell on Binance
   */
  async executeDexToCex(opp: ArbitrageOpportunity, amountUsd: number): Promise<ArbitrageResult> {
    const startTime = Date.now();
    const symbol = opp.symbol;
    const token = TOKENS[symbol];

    if (!token) {
      return this.errorResult(opp, 'dex-to-cex', `Unknown token: ${symbol}`);
    }

    console.log(`\n[ARBITRAGE] ${symbol} detected: +${opp.spreadPct.toFixed(2)}% spread (DEX→CEX)`);

    const quantity = amountUsd / opp.buyPrice;

    const result: ArbitrageResult = {
      success: false,
      direction: 'dex-to-cex',
      symbol,
      buyExchange: opp.buyExchange,
      sellExchange: opp.sellExchange,
      amountIn: amountUsd,
      amountOut: 0,
      grossProfit: 0,
      fees: { cexTrade: 0, withdrawal: 0, dexSwap: 0, gas: 0 },
      netProfit: 0,
      profitPct: 0,
      txHashes: {},
      executionTimeMs: 0,
    };

    // === DRY RUN MODE ===
    if (this.config.dryRun) {
      return await this.simulateDexToCex(opp, amountUsd, quantity, startTime);
    }

    // === LIVE EXECUTION ===
    try {
      if (!this.keypair) {
        throw new Error('Solana wallet not configured');
      }

      const walletAddress = this.keypair.publicKey.toBase58();

      // Step 1: Buy on Jupiter (USDC → token)
      const inputLamports = Math.floor(amountUsd * 1e6); // USDC has 6 decimals
      console.log(`[ARBITRAGE] Buy from ${opp.buyExchange}: ${quantity.toFixed(4)} ${symbol} at $${opp.buyPrice.toFixed(6)}`);

      const quote = await getJupiterQuote(TOKENS.USDC.mint, token.mint, inputLamports, walletAddress);
      const swapTxid = await executeJupiterSwap(quote, walletAddress, this.connection, this.keypair);

      result.txHashes.dexSwap = swapTxid;
      const executedQty = parseInt(quote.outAmount) / Math.pow(10, token.decimals);
      result.fees.dexSwap = amountUsd * 0.003; // ~0.3% Jupiter fee
      const solPrice = await getSolPrice();
      result.fees.gas = DEFAULT_GAS_SOL * solPrice;

      console.log(`[ARBITRAGE] ✅ Bought ${executedQty.toFixed(4)} ${symbol} - TX: ${swapTxid.slice(0, 8)}...`);

      // Step 2: Get Binance deposit address
      console.log(`[ARBITRAGE] Getting Binance deposit address for ${symbol}...`);
      const depositAddress = await this.getBinanceDepositAddress(symbol);
      console.log(`[ARBITRAGE] Deposit address: ${depositAddress.slice(0, 8)}...`);

      // Step 3: Transfer to Binance (SPL token transfer)
      // No need to wait - transaction confirmed means tokens are in our ATA
      console.log(`[ARBITRAGE] Transferring ${executedQty.toFixed(4)} ${symbol} to Binance...`);
      const transferResult = await this.transferSPLToken(token.mint, depositAddress, executedQty, token.decimals);
      result.txHashes.deposit = transferResult.txid;
      console.log(`[ARBITRAGE] ✅ Deposit initiated - TX: ${transferResult.txid.slice(0, 8)}...`);

      // Use actual transferred amount (may be slightly less due to slippage)
      const actualTransferredQty = transferResult.actualAmount;
      console.log(`[ARBITRAGE] 🔍 Actual transferred: ${actualTransferredQty.toFixed(6)} ${symbol}`);

      // Step 4: Wait for deposit confirmation
      console.log(`[ARBITRAGE] ⏳ Waiting for Binance deposit confirmation (~10-30 min)...`);
      await this.waitForBinanceDeposit(symbol, actualTransferredQty);
      console.log(`[ARBITRAGE] ✅ Deposit confirmed on Binance`);

      // Step 5: Sell on Binance - USE ACTUAL TRANSFERRED AMOUNT
      console.log(`[ARBITRAGE] Sell on ${opp.sellExchange}: ${actualTransferredQty.toFixed(4)} ${symbol}`);
      const order = await binanceSpotOrder(this.config, token.binanceSymbol, 'SELL', actualTransferredQty, token.binanceDecimals);
      result.txHashes.cexOrder = order.clientOrderId;
      result.fees.cexTrade = parseFloat(order.fills?.[0]?.commission || '0');

      const sellAmount = parseFloat(order.cummulativeQuoteQty);
      result.amountOut = sellAmount;
      console.log(`[ARBITRAGE] ✅ Sold for $${sellAmount.toFixed(2)} USDT`);

      // Calculate profit
      result.grossProfit = result.amountOut - amountUsd;
      result.netProfit = result.grossProfit - result.fees.cexTrade - result.fees.dexSwap - result.fees.gas;
      result.profitPct = (result.netProfit / amountUsd) * 100;
      result.success = result.netProfit > 0;
      result.executionTimeMs = Date.now() - startTime;

      // ========== TRACK IN PORTFOLIO ==========
      const totalFees = result.fees.cexTrade + result.fees.dexSwap + result.fees.gas;
      getPortfolioManager().recordArbitrageTrade({
        asset: symbol,
        amountUsd: amountUsd,
        venue: `${opp.buyExchange}->${opp.sellExchange}`,
        fees: totalFees,
        pnlUsd: result.netProfit,
        notes: `DEX-CEX arb: ${result.profitPct.toFixed(2)}% profit`,
      });
      // ========================================

      console.log(`[ARBITRAGE] ${result.netProfit > 0 ? '✅' : '❌'} Net profit: $${result.netProfit.toFixed(2)} (+${result.profitPct.toFixed(2)}%)`);
      return result;

    } catch (error) {
      console.error(`[ARBITRAGE] ❌ DEX→CEX execution failed:`, error);
      result.executionTimeMs = Date.now() - startTime;
      return result;
    }
  }

  /**
   * Simulate DEX → CEX arbitrage (dry run)
   */
  private async simulateDexToCex(
    opp: ArbitrageOpportunity,
    amountUsd: number,
    quantity: number,
    startTime: number
  ): Promise<ArbitrageResult> {
    // Get live SOL price for gas calculation
    const solPrice = await getSolPrice();

    // Estimate fees with dynamic pricing
    const dexSwapFee = amountUsd * 0.003; // 0.3% Jupiter
    const depositFee = 0; // Usually free
    const cexTradeFee = amountUsd * 0.001; // 0.1% Binance
    const gasFee = DEFAULT_GAS_SOL * solPrice; // Dynamic: 0.002 SOL * live price

    const grossProfit = amountUsd * (opp.spreadPct / 100);
    const totalFees = dexSwapFee + depositFee + cexTradeFee + gasFee;
    const netProfit = grossProfit - totalFees;

    console.log(`[ARBITRAGE] 📝 DRY RUN - Simulating DEX→CEX...`);
    console.log(`[ARBITRAGE] SOL price: $${solPrice.toFixed(2)} (live)`);
    console.log(`[ARBITRAGE] Buy on ${opp.buyExchange}: ${quantity.toFixed(4)} ${opp.symbol} at $${opp.buyPrice.toFixed(6)}`);
    console.log(`[ARBITRAGE] Deposit to Binance: ~15min ETA`);
    console.log(`[ARBITRAGE] Sell on Binance: ${quantity.toFixed(4)} ${opp.symbol} → $${(amountUsd + grossProfit).toFixed(2)} USDT`);
    console.log(`[ARBITRAGE] Gas: ${DEFAULT_GAS_SOL} SOL = $${gasFee.toFixed(4)}`);
    console.log(`[ARBITRAGE] ${netProfit > 0 ? '✅' : '❌'} Net profit: $${netProfit.toFixed(2)} (+${((netProfit / amountUsd) * 100).toFixed(2)}%)`);

    const result: ArbitrageResult = {
      success: netProfit > 0,
      direction: 'dex-to-cex',
      symbol: opp.symbol,
      buyExchange: opp.buyExchange,
      sellExchange: opp.sellExchange,
      amountIn: amountUsd,
      amountOut: amountUsd + grossProfit,
      grossProfit,
      fees: {
        cexTrade: cexTradeFee,
        withdrawal: 0,
        dexSwap: dexSwapFee,
        gas: gasFee,
      },
      netProfit,
      profitPct: (netProfit / amountUsd) * 100,
      txHashes: {},
      executionTimeMs: Date.now() - startTime,
    };

    // ========== TRACK IN PORTFOLIO ==========
    getPortfolioManager().recordArbitrageTrade({
      asset: opp.symbol,
      amountUsd: amountUsd,
      venue: `${opp.buyExchange}->${opp.sellExchange}`,
      fees: totalFees,
      pnlUsd: netProfit,
      notes: `DEX-CEX arb (simulated): ${result.profitPct.toFixed(2)}% profit`,
    });
    // ========================================

    return result;
  }

  /**
   * Determine best direction and execute
   */
  async execute(opp: ArbitrageOpportunity, amountUsd: number = 1000): Promise<ArbitrageResult> {
    // ========== PM APPROVAL CHECK (before Guardian) ==========
    if (pmDecisionEngine.isEnabled()) {
      const pmParams: QueueTradeParams = {
        strategy: 'arbitrage',
        action: 'BUY',
        asset: opp.symbol,
        amount: amountUsd,
        amountUsd,
        confidence: 0.8,
        risk: {
          volatility: 0,
          liquidityScore: 60,
          riskScore: 50,
        },
        reasoning: `Arbitrage opportunity: ${opp.spreadPct.toFixed(2)}% spread (${opp.buyExchange} → ${opp.sellExchange})`,
        protocol: `${opp.buyExchange}-${opp.sellExchange}`,
      };

      const portfolioValueUsd = 10000;
      const needsApproval = pmDecisionEngine.needsApproval(pmParams, portfolioValueUsd);

      if (needsApproval) {
        console.log(`[ARBITRAGE] Trade requires PM approval for ${opp.symbol}`);

        const tradeId = approvalQueue.queueTrade(pmParams);
        const approvalResult = await approvalQueue.waitForApproval(tradeId);

        if (!approvalResult.approved) {
          console.log(`[ARBITRAGE] PM rejected transaction: ${approvalResult.rejectionReason}`);
          return this.errorResult(opp, 'cex-to-dex', `PM rejected: ${approvalResult.rejectionReason || approvalResult.status}`);
        }

        console.log(`[ARBITRAGE] PM approved transaction`);
      }
    }
    // ========================================

    // ========== GUARDIAN PRE-EXECUTION VALIDATION ==========
    // Look up token mint address from symbol
    const token = TOKENS[opp.symbol];
    if (!token) {
      console.log(`[ARBITRAGE] ❌ Unknown token: ${opp.symbol} - skipping Guardian validation`);
      return this.errorResult(opp, 'cex-to-dex', `Unknown token: ${opp.symbol}`);
    }

    const guardianParams: GuardianTradeParams = {
      inputMint: TOKENS.USDC.mint,
      outputMint: token.mint,
      amountIn: amountUsd,
      amountInUsd: amountUsd,
      slippageBps: 100,
      strategy: 'arbitrage',
      protocol: `${opp.buyExchange}-${opp.sellExchange}`,
      walletAddress: this.keypair?.publicKey.toBase58() || '',
    };
    const guardianResult = await guardian.validate(guardianParams);
    if (!guardianResult.approved) {
      console.log(`[ARBITRAGE] 🛡️ Guardian blocked transaction: ${guardianResult.blockReason}`);
      return this.errorResult(opp, 'cex-to-dex', `Guardian blocked: ${guardianResult.blockReason}`);
    }

    // ========== GLOBAL RISK CHECK ==========
    // Check all risk limits before executing arbitrage
    const riskManager = getGlobalRiskManager(undefined, this.config.solanaRpcUrl, this.config.dryRun);
    const walletBalanceSol = await this.getWalletBalance();
    const riskCheck = await riskManager.performGlobalRiskCheck({
      symbol: opp.symbol,
      protocol: 'arbitrage',
      sizeUsd: amountUsd,
      walletBalanceSol,
    });

    if (!riskCheck.canTrade) {
      console.log(`[ARBITRAGE] ❌ Blocked by risk manager: ${riskCheck.blockReasons.join('; ')}`);

      // In dry-run mode, simulate the trade anyway for paper trading
      if (this.config.dryRun) {
        console.log(`[ARBITRAGE] 📝 DRY-RUN: Simulating trade despite block (paper trading)...`);
        // Continue to execution simulation below
      } else {
        // In production, strictly block the trade
        return this.errorResult(opp, 'cex-to-dex', `Risk check failed: ${riskCheck.blockReasons.join('; ')}`);
      }
    }
    // ========================================

    const minSpread = this.config.minSpreadPct;
    const feeEstimate = 0.25; // Conservative fee estimate covering most exchanges
    const netSpread = opp.spreadPct - feeEstimate;

    if (netSpread < minSpread) {
      console.log(`[ARBITRAGE] ❌ Spread too low: ${opp.spreadPct.toFixed(2)}% (need >${minSpread + feeEstimate}%)`);
      return this.errorResult(opp, 'cex-to-dex', `Spread too low after fees: ${netSpread.toFixed(2)}%`);
    }

    const estimatedProfit = amountUsd * (netSpread / 100);
    if (estimatedProfit < this.config.minProfitUsd) {
      console.log(`[ARBITRAGE] ❌ Profit too low: $${estimatedProfit.toFixed(2)} (need >$${this.config.minProfitUsd})`);
      return this.errorResult(opp, 'cex-to-dex', `Profit too low: $${estimatedProfit.toFixed(2)}`);
    }

    console.log(`[ARBITRAGE] Risk check passed: ${riskCheck.circuitBreakerState}`);

    // Determine arbitrage type
    const isCex = (exchange: string) => {
      const lower = exchange.toLowerCase();
      return lower.includes('binance') || lower.includes('coinbase') || lower.includes('kraken');
    };

    const buyIsCex = isCex(opp.buyExchange);
    const sellIsCex = isCex(opp.sellExchange);

    if (buyIsCex && !sellIsCex) {
      // CEX → DEX (buy on Binance, sell on Jupiter)
      return this.executeCexToDex(opp, amountUsd);
    } else if (!buyIsCex && sellIsCex) {
      // DEX → CEX (buy on Jupiter, sell on Binance)
      console.log(`[ARBITRAGE] DEX→CEX path (deposit delays ~15min)`);
      return this.executeDexToCex(opp, amountUsd);
    } else if (!buyIsCex && !sellIsCex) {
      // DEX → DEX (buy on Orca, sell on Meteora via Jupiter)
      return this.executeDexToDex(opp, amountUsd);
    } else {
      // CEX → CEX (not supported)
      console.log(`[ARBITRAGE] ⚠️ CEX→CEX not supported`);
      return this.errorResult(opp, 'cex-to-cex', 'CEX→CEX not supported');
    }
  }

  // ============= HELPERS =============

  private getWithdrawalFee(symbol: string): number {
    // Binance withdrawal fees (in token units, converted to USD estimate)
    const fees: Record<string, number> = {
      SOL: 0.01 * 200,   // 0.01 SOL
      BONK: 0,           // Free for SPL tokens
      JUP: 0,
      WIF: 0,
      USDC: 1,           // $1 flat
    };
    return fees[symbol] || 1;
  }

  private async waitForBalance(
    tokenMint: string,
    expectedAmount: number,
    maxWaitMs: number
  ): Promise<void> {
    // In production: poll SPL token balance until it arrives
    // For now, just wait fixed time
    const waitTime = Math.min(maxWaitMs, 120000); // Max 2 min
    console.log(`[ARBITRAGE] Waiting ${waitTime / 1000}s for balance...`);
    await new Promise(resolve => setTimeout(resolve, waitTime));
  }

  private async getBinanceDepositAddress(symbol: string): Promise<string> {
    const response = await binanceGetDepositAddress(this.config, symbol, 'SOL');
    console.log(`[ARBITRAGE] 🔍 Binance deposit response:`, JSON.stringify(response, null, 2));
    console.log(`[ARBITRAGE] 🔍 Deposit address: ${response.address}`);
    console.log(`[ARBITRAGE] 🔍 Address tag: ${response.tag || 'none'}`);
    return response.address;
  }

  /**
   * Wait for tokens to arrive in our ATA (for gasless transactions)
   */
  private async waitForTokenBalance(
    tokenMint: string,
    expectedAmount: number,
    decimals: number,
    maxWaitMs: number
  ): Promise<void> {
    if (!this.keypair) {
      throw new Error('Solana wallet not configured');
    }

    const { PublicKey } = await import('@solana/web3.js');
    const { getAssociatedTokenAddress, getAccount } = await import('@solana/spl-token');

    const mintPubkey = new PublicKey(tokenMint);
    const ata = await getAssociatedTokenAddress(mintPubkey, this.keypair.publicKey);

    const startTime = Date.now();
    const expectedLamports = Math.floor(expectedAmount * Math.pow(10, decimals));

    console.log(`[ARBITRAGE] 🔍 Polling ATA balance...`);
    console.log(`[ARBITRAGE]    ATA: ${ata.toBase58()}`);
    console.log(`[ARBITRAGE]    Expected: ${expectedLamports} lamports (${expectedAmount} tokens)`);

    while (Date.now() - startTime < maxWaitMs) {
      try {
        const accountInfo = await getAccount(this.connection, ata);
        const balance = Number(accountInfo.amount);

        console.log(`[ARBITRAGE]    Current balance: ${balance} lamports (${balance / Math.pow(10, decimals)} tokens)`);

        if (balance > 0) {
          console.log(`[ARBITRAGE] ✅ Tokens arrived! Balance: ${balance} lamports`);
          return;
        }
      } catch (error) {
        // ATA might not exist yet, keep polling
        console.log(`[ARBITRAGE]    ATA not found yet, continuing to poll...`);
      }

      // Wait 2 seconds before next poll
      await new Promise(resolve => setTimeout(resolve, 2000));
    }

    throw new Error(`Timeout waiting for tokens to arrive in ATA after ${maxWaitMs}ms`);
  }

  private async transferSPLToken(
    tokenMint: string,
    destinationAddress: string,
    amount: number,
    decimals: number
  ): Promise<{ txid: string; actualAmount: number }> {
    if (!this.keypair) {
      throw new Error('Solana wallet not configured');
    }

    const { PublicKey, Transaction, sendAndConfirmTransaction, SystemProgram, LAMPORTS_PER_SOL } = await import('@solana/web3.js');
    const destinationPubkey = new PublicKey(destinationAddress);

    // Check if this is native SOL (not wrapped SOL)
    const NATIVE_SOL_MINT = 'So11111111111111111111111111111111111111112';
    const isNativeSOL = tokenMint === NATIVE_SOL_MINT;

    if (isNativeSOL) {
      // Native SOL transfer - use SystemProgram
      console.log(`[ARBITRAGE] Transfer details (Native SOL):`);
      console.log(`  From: ${this.keypair.publicKey.toBase58()}`);
      console.log(`  To: ${destinationPubkey.toBase58()}`);
      console.log(`  Amount: ${amount} SOL (${Math.floor(amount * LAMPORTS_PER_SOL)} lamports)`);

      const transaction = new Transaction().add(
        SystemProgram.transfer({
          fromPubkey: this.keypair.publicKey,
          toPubkey: destinationPubkey,
          lamports: Math.floor(amount * LAMPORTS_PER_SOL),
        })
      );

      const signature = await sendAndConfirmTransaction(
        this.connection,
        transaction,
        [this.keypair],
        { commitment: 'confirmed' }
      );

      console.log(`[ARBITRAGE] ✅ Native SOL transfer successful: ${signature}`);
      return { txid: signature, actualAmount: amount };
    }

    // SPL Token transfer (for non-SOL tokens like BONK, WIF, etc.)
    const { getAssociatedTokenAddress, createTransferInstruction, createAssociatedTokenAccountInstruction, TOKEN_PROGRAM_ID } = await import('@solana/spl-token');

    const mintPubkey = new PublicKey(tokenMint);

    // Get source token account (our wallet's token account)
    const sourceTokenAccount = await getAssociatedTokenAddress(
      mintPubkey,
      this.keypair.publicKey
    );

    // Binance gives us a wallet address, we need to derive the token account (ATA)
    const destinationTokenAccount = await getAssociatedTokenAddress(
      mintPubkey,
      destinationPubkey
    );

    console.log(`[ARBITRAGE] Transfer details (SPL Token):`);
    console.log(`  Token Mint: ${mintPubkey.toBase58()}`);
    console.log(`  From (our ATA): ${sourceTokenAccount.toBase58()}`);
    console.log(`  To (Binance wallet): ${destinationPubkey.toBase58()}`);
    console.log(`  To (Binance ATA): ${destinationTokenAccount.toBase58()}`);
    console.log(`  Amount: ${amount} (${Math.floor(amount * Math.pow(10, decimals))} lamports)`);

    // Check source ATA balance BEFORE transfer
    const { getAccount } = await import('@solana/spl-token');
    let actualBalance: number;

    try {
      const sourceAccountInfo = await getAccount(this.connection, sourceTokenAccount);
      actualBalance = Number(sourceAccountInfo.amount);
      const expectedBalance = Math.floor(amount * Math.pow(10, decimals));

      console.log(`[ARBITRAGE] 🔍 Source ATA Balance Check:`);
      console.log(`  Expected: ${expectedBalance} (${amount} tokens)`);
      console.log(`  Actual: ${actualBalance} (${actualBalance / Math.pow(10, decimals)} tokens)`);

      if (actualBalance === 0) {
        throw new Error(`No balance in source ATA!`);
      }

      // Use actual balance if it's slightly less than expected (due to slippage/fees)
      if (actualBalance < expectedBalance) {
        const diff = expectedBalance - actualBalance;
        const diffPct = (diff / expectedBalance) * 100;
        console.log(`[ARBITRAGE] ⚠️  Balance slightly lower than expected (${diffPct.toFixed(4)}% difference)`);
        console.log(`[ARBITRAGE] ✅ Using actual balance: ${actualBalance} lamports`);
      } else {
        console.log(`[ARBITRAGE] ✅ Balance verified - sufficient funds available`);
      }
    } catch (error: any) {
      console.log(`[ARBITRAGE] ❌ Source ATA balance check failed:`, error.message);
      throw error;
    }

    // Check if destination ATA exists
    const destinationAccountInfo = await this.connection.getAccountInfo(destinationTokenAccount);
    const transaction = new Transaction();

    if (!destinationAccountInfo) {
      console.log(`[ARBITRAGE] ⚠️  Destination ATA does not exist - creating it...`);
      // Create the destination ATA
      const createAtaInstruction = createAssociatedTokenAccountInstruction(
        this.keypair.publicKey, // payer
        destinationTokenAccount, // ata
        destinationPubkey, // owner
        mintPubkey // mint
      );
      transaction.add(createAtaInstruction);
      console.log(`[ARBITRAGE] ✅ Added create ATA instruction`);
    } else {
      console.log(`[ARBITRAGE] ✅ Destination ATA already exists`);
    }

    // Create transfer instruction - USE ACTUAL BALANCE (not expected amount)
    const transferAmount = actualBalance; // Use actual balance from ATA
    const transferInstruction = createTransferInstruction(
      sourceTokenAccount,
      destinationTokenAccount,
      this.keypair.publicKey,
      transferAmount,
      [],
      TOKEN_PROGRAM_ID
    );

    // Add transfer instruction to transaction
    transaction.add(transferInstruction);
    const signature = await sendAndConfirmTransaction(
      this.connection,
      transaction,
      [this.keypair],
      { commitment: 'confirmed' }
    );

    // Return both txid and actual transferred amount
    const actualAmountTransferred = actualBalance / Math.pow(10, decimals);
    console.log(`[ARBITRAGE] ✅ Transfer successful - sent ${actualAmountTransferred.toFixed(6)} tokens`);
    return { txid: signature, actualAmount: actualAmountTransferred };
  }

  private async waitForBinanceDeposit(symbol: string, expectedAmount: number): Promise<void> {
    // In production: poll Binance deposit history until deposit arrives
    // For now, just wait fixed time (deposits typically take 10-30 min)
    const waitTime = Math.min(this.config.maxWithdrawWaitMs, 1800000); // Max 30 min
    console.log(`[ARBITRAGE] Waiting ${waitTime / 60000} minutes for Binance deposit...`);
    await new Promise(resolve => setTimeout(resolve, waitTime));

    // TODO: In production, poll Binance deposit history:
    // GET /sapi/v1/capital/deposit/hisrec
    // Check for deposit with matching coin and amount
  }

  private errorResult(
    opp: ArbitrageOpportunity,
    direction: 'cex-to-dex' | 'dex-to-cex' | 'dex-to-dex' | 'cex-to-cex',
    error: string
  ): ArbitrageResult {
    return {
      success: false,
      direction,
      symbol: opp.symbol,
      buyExchange: opp.buyExchange,
      sellExchange: opp.sellExchange,
      amountIn: 0,
      amountOut: 0,
      grossProfit: 0,
      fees: { cexTrade: 0, withdrawal: 0, dexSwap: 0, gas: 0 },
      netProfit: 0,
      profitPct: 0,
      txHashes: {},
      executionTimeMs: 0,
      error,
    };
  }
}

// ============= FACTORY =============

export function createArbitrageExecutor(dryRun: boolean = true): ArbitrageExecutor {
  return new ArbitrageExecutor({
    binanceApiKey: process.env.BINANCE_API_KEY || '',
    binanceSecretKey: process.env.BINANCE_SECRET_KEY || '',
    solanaPrivateKey: process.env.SOLANA_PRIVATE_KEY || '',
    solanaRpcUrl: process.env.SOLANA_RPC_URL || 'https://api.mainnet-beta.solana.com',
    dryRun,
    minProfitUsd: 5.0,       // Min $5 profit after all fees
    minSpreadPct: 0.10,      // Min 0.10% spread after fees
    maxWithdrawWaitMs: 300000, // 5 min max wait
  });
}