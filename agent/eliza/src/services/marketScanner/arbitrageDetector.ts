/**
 * Arbitrage Detector - CEX/DEX spread analysis
 */

import type { CEXPrice, DEXPrice, ArbitrageOpportunity } from './types.js';

// Fee estimates per exchange (%)
const EXCHANGE_FEES: Record<string, number> = {
  binance: 0.1,
  coinbase: 0.5,
  kraken: 0.26,
  raydium: 0.25,
  orca: 0.25,
  meteora: 0.1,
  jupiter: 0.0, // Aggregator, no extra fee
};

// Withdrawal fees (estimated in USD)
const WITHDRAWAL_FEES: Record<string, number> = {
  binance: 0.01, // SOL
  coinbase: 0.0,
  kraken: 0.01,
};

interface PricePoint {
  symbol: string;
  exchange: string;
  price: number;
  type: 'cex' | 'dex';
  poolAddress?: string;
}

// DEXScreener is a data aggregator, not an executable venue
const NON_EXECUTABLE_VENUES = ['dexscreener', 'birdeye'];

export function detectArbitrage(
  cexPrices: CEXPrice[],
  dexPrices: DEXPrice[],
  minSpreadPct = 0.5,
  tradeAmountUsd = 1000
): ArbitrageOpportunity[] {
  const opportunities: ArbitrageOpportunity[] = [];

  // Group prices by symbol
  const pricesBySymbol = new Map<string, PricePoint[]>();

  for (const p of cexPrices) {
    // Skip if price is invalid
    if (!p.price || p.price <= 0) continue;

    const key = p.symbol;
    if (!pricesBySymbol.has(key)) pricesBySymbol.set(key, []);
    pricesBySymbol.get(key)!.push({
      symbol: p.symbol,
      exchange: p.exchange,
      price: p.price,
      type: 'cex',
    });
  }

  for (const p of dexPrices) {
    // Skip if price is invalid or venue is not executable
    if (!p.price || p.price <= 0) continue;
    if (NON_EXECUTABLE_VENUES.includes(p.dex.toLowerCase())) continue;

    const key = p.symbol;
    if (!pricesBySymbol.has(key)) pricesBySymbol.set(key, []);
    pricesBySymbol.get(key)!.push({
      symbol: p.symbol,
      exchange: p.dex,
      price: p.price,
      type: 'dex',
      poolAddress: p.poolAddress,
    });
  }
  
  // Find arbitrage opportunities
  for (const [symbol, prices] of pricesBySymbol) {
    if (prices.length < 2) continue;

    // Filter out obviously wrong prices (data errors)
    const validPrices = prices.filter(p => {
      // Price must be positive
      if (p.price <= 0) return false;

      // Sanity checks per token (prevent decimal errors)
      if (symbol === 'SOL' && (p.price < 50 || p.price > 500)) return false;
      if (symbol === 'BONK' && (p.price < 0.000001 || p.price > 0.001)) return false;
      if (symbol === 'JUP' && (p.price < 0.1 || p.price > 10)) return false;
      if (symbol === 'WIF' && (p.price < 0.01 || p.price > 10)) return false;
      if (symbol === 'ETH' && (p.price < 1000 || p.price > 10000)) return false;
      if (symbol === 'USDC' && (p.price < 0.95 || p.price > 1.05)) return false;
      if (symbol === 'USDT' && (p.price < 0.95 || p.price > 1.05)) return false;

      return true;
    });

    if (validPrices.length < 2) continue;

    // Sort by price to find min/max
    const sorted = [...validPrices].sort((a, b) => a.price - b.price);
    const lowest = sorted[0];
    const highest = sorted[sorted.length - 1];

    // DEBUG: Log price comparison
    console.log(`[ArbitrageDetector] ${symbol}: lowest=${lowest.exchange}@$${lowest.price.toFixed(6)}, highest=${highest.exchange}@$${highest.price.toFixed(6)}`);

    const spreadPct = ((highest.price - lowest.price) / lowest.price) * 100;

    // Sanity check: Realistic arbitrage spreads are 0.1% - 10%
    // Anything above 10% is likely a data error
    if (spreadPct < minSpreadPct || spreadPct > 10) {
      if (spreadPct > 10) {
        console.log(`[ArbitrageDetector] ⚠️  ${symbol}: Spread ${spreadPct.toFixed(2)}% too high (>10%), likely data error - SKIPPED`);
      }
      continue;
    }
    
    const buyFee = EXCHANGE_FEES[lowest.exchange] || 0.25;
    const sellFee = EXCHANGE_FEES[highest.exchange] || 0.25;
    const withdrawFee = lowest.type === 'cex' ? (WITHDRAWAL_FEES[lowest.exchange] || 0) : 0;

    const totalFeePct = buyFee + sellFee;
    const netSpreadPct = spreadPct - totalFeePct;

    if (netSpreadPct <= 0) continue;
    
    const estimatedProfit = (tradeAmountUsd * spreadPct) / 100;
    const fees = (tradeAmountUsd * totalFeePct) / 100 + withdrawFee;
    const netProfit = estimatedProfit - fees;
    
    // Confidence based on spread size and exchange reliability
    let confidence: 'high' | 'medium' | 'low' = 'low';
    if (netSpreadPct > 1.5 && lowest.type === 'cex' && highest.type === 'cex') {
      confidence = 'high';
    } else if (netSpreadPct > 0.8) {
      confidence = 'medium';
    }

    opportunities.push({
      symbol,
      buyExchange: lowest.exchange,
      sellExchange: highest.exchange,
      buyPrice: lowest.price,
      sellPrice: highest.price,
      spreadPct,
      estimatedProfit,
      fees,
      netProfit,
      confidence,
      buyPoolAddress: lowest.poolAddress,
      sellPoolAddress: highest.poolAddress,
    } as ArbitrageOpportunity);
  }
  
  // Sort by net profit
  return opportunities.sort((a, b) => b.netProfit - a.netProfit);
}

export function formatArbitrageOpportunity(opp: ArbitrageOpportunity): string {
  const arrow = opp.buyExchange.charAt(0).toUpperCase() + opp.buyExchange.slice(1) +
    ' → ' +
    opp.sellExchange.charAt(0).toUpperCase() + opp.sellExchange.slice(1);
  
  const profitEmoji = opp.netProfit > 10 ? '🔥' : opp.netProfit > 5 ? '✨' : '💰';
  
  return `${opp.symbol}: ${arrow} | Profit: +${opp.spreadPct.toFixed(2)}% (net +${opp.netProfit.toFixed(2)}$) ${profitEmoji}`;
}

