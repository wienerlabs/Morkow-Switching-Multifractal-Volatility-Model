/**
 * Console Dashboard - Pretty market data output
 */

import type { MarketSnapshot, ArbitrageOpportunity, LPPool, NewToken } from './types.js';
import { getRiskEmoji } from './strategySelector.js';

const COLORS = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  dim: '\x1b[2m',
  cyan: '\x1b[36m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  magenta: '\x1b[35m',
  blue: '\x1b[34m',
  white: '\x1b[37m',
  bgBlue: '\x1b[44m',
};

const c = COLORS;

function formatNumber(n: number): string {
  if (n >= 1_000_000_000) return (n / 1_000_000_000).toFixed(1) + 'B';
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
  return n.toFixed(0);
}

// Smart price formatting - handles very small prices like BONK, SHIB
function formatPrice(price: number): string {
  if (price >= 1000) return price.toFixed(0);
  if (price >= 1) return price.toFixed(2);
  if (price >= 0.01) return price.toFixed(4);
  if (price >= 0.0001) return price.toFixed(6);
  // For very small prices like BONK (0.00001), use scientific notation or full precision
  if (price >= 0.000001) return price.toFixed(8);
  return price.toExponential(2);
}

function formatPct(n: number, showPlus = true): string {
  const sign = n >= 0 && showPlus ? '+' : '';
  const color = n >= 0 ? c.green : c.red;
  return `${color}${sign}${n.toFixed(2)}%${c.reset}`;
}

export function renderDashboard(snapshot: MarketSnapshot): void {
  const timestamp = new Date(snapshot.timestamp).toLocaleString();

  console.clear();
  console.log(`\n${c.bgBlue}${c.white}${c.bright} CORTEX MARKET SCANNER ${c.reset} - ${c.dim}${timestamp}${c.reset}\n`);
  console.log('━'.repeat(60));

  // Global Market Data (from CryptoRank)
  if (snapshot.globalData) {
    const g = snapshot.globalData;
    const fgLabel = g.fearGreed <= 25 ? 'Extreme Fear' : g.fearGreed <= 45 ? 'Fear' : g.fearGreed <= 55 ? 'Neutral' : g.fearGreed <= 75 ? 'Greed' : 'Extreme Greed';
    console.log(`\n${c.blue}${c.bright}🌐 GLOBAL MARKET${c.reset}\n`);
    console.log(`   MCap $${(g.totalMarketCap / 1e12).toFixed(2)}T (${formatPct(g.totalMarketCapChange)}) | Vol24h $${formatNumber(g.totalVolume24h)} (${formatPct(g.totalVolume24hChange)})`);
    console.log(`   BTC Dom ${g.btcDominance.toFixed(1)}% (${formatPct(g.btcDominanceChange)}) | ETH Dom ${g.ethDominance.toFixed(1)}% (${formatPct(g.ethDominanceChange)})`);
    console.log(`   Fear/Greed: ${g.fearGreed} (${fgLabel}) | Altcoin Index: ${g.altcoinIndex}`);
  }

  // Price Comparison Section
  console.log(`\n${c.blue}${c.bright}📊 PRICE COMPARISON${c.reset}\n`);

  // Group prices by symbol
  const pricesBySymbol = new Map<string, { cex: Map<string, number>; dex: Map<string, number> }>();

  for (const p of snapshot.cexPrices) {
    if (!pricesBySymbol.has(p.symbol)) {
      pricesBySymbol.set(p.symbol, { cex: new Map(), dex: new Map() });
    }
    pricesBySymbol.get(p.symbol)!.cex.set(p.exchange, p.price);
  }

  for (const p of snapshot.dexPrices) {
    if (!pricesBySymbol.has(p.symbol)) {
      pricesBySymbol.set(p.symbol, { cex: new Map(), dex: new Map() });
    }
    pricesBySymbol.get(p.symbol)!.dex.set(p.source, p.price);
  }

  for (const [symbol, prices] of pricesBySymbol) {
    // CEX prices - use smart formatting for small prices
    const cexPrices = Array.from(prices.cex.entries())
      .map(([ex, price]) => `${capitalize(ex)} $${formatPrice(price)}`)
      .join(' | ');

    // DEX prices - use smart formatting for small prices
    const dexPrices = Array.from(prices.dex.entries())
      .map(([dex, price]) => `${capitalize(dex)} $${formatPrice(price)}`)
      .join(' | ');

    console.log(`   ${c.bright}${symbol}${c.reset}`);
    if (cexPrices) console.log(`      ${c.dim}CEX:${c.reset} ${cexPrices}`);
    if (dexPrices) console.log(`      ${c.dim}DEX:${c.reset} ${dexPrices}`);
  }

  // Arbitrage Section
  console.log(`\n${c.cyan}${c.bright}🔄 ARBITRAGE${c.reset} (${snapshot.arbitrage.length} fırsat)\n`);
  if (snapshot.arbitrage.length === 0) {
    console.log(`   ${c.dim}Arbitraj fırsatı bulunamadı${c.reset}`);
  } else {
    for (const arb of snapshot.arbitrage.slice(0, 5)) {
      const arrow = `${capitalize(arb.buyExchange)} → ${capitalize(arb.sellExchange)}`;
      const emoji = arb.confidence === 'high' ? '🔥' : arb.confidence === 'medium' ? '✨' : '💰';
      console.log(`   ${c.bright}${arb.symbol}${c.reset}: ${arrow} | Profit: ${formatPct(arb.spreadPct)} (net $${arb.netProfit.toFixed(2)}) ${emoji}`);
    }
  }
  
  // LP Pools Section
  console.log(`\n${c.magenta}${c.bright}💎 TOP LP POOLS${c.reset} (${snapshot.lpPools.length})\n`);
  if (snapshot.lpPools.length === 0) {
    console.log(`   ${c.dim}Pool verisi bulunamadı${c.reset}`);
  } else {
    for (const pool of snapshot.lpPools.slice(0, 8)) {
      const riskEmoji = getRiskEmoji(pool.riskScore <= 3 ? 'low' : pool.riskScore <= 6 ? 'medium' : 'high');
      const dexLabel = pool.dex ? `${c.dim}[${capitalize(pool.dex)}]${c.reset} ` : '';
      console.log(`   ${dexLabel}${c.bright}${pool.name}${c.reset}: APY ${c.green}${pool.apy.toFixed(1)}%${c.reset} | TVL $${formatNumber(pool.tvl)} | Vol $${formatNumber(pool.volume24h)} ${riskEmoji}`);
    }
  }

  // New Tokens Section
  console.log(`\n${c.yellow}${c.bright}🚀 NEW TOKENS${c.reset} (${snapshot.newTokens.length})\n`);
  if (snapshot.newTokens.length === 0) {
    console.log(`   ${c.dim}Yeni token bulunamadı${c.reset}`);
  } else {
    for (const token of snapshot.newTokens.slice(0, 5)) {
      const emoji = token.priceChange24h > 100 ? '🌙' : token.priceChange24h > 50 ? '📈' : '💫';
      console.log(`   ${c.bright}${token.symbol}${c.reset}: 24h ${formatPct(token.priceChange24h)} | Liq $${formatNumber(token.liquidity)} ${emoji}`);
    }
  }
  
  // Best Strategy Section
  console.log('\n' + '━'.repeat(60));
  if (snapshot.bestStrategy) {
    const strat = snapshot.bestStrategy;
    const emoji = strat.type === 'arbitrage' ? '🔄' : strat.type === 'lp' ? '💎' : '🚀';
    const riskEmoji = getRiskEmoji(strat.risk);
    console.log(`\n${c.bright}✨ BEST OPPORTUNITY:${c.reset} ${emoji} ${c.cyan}${strat.name}${c.reset}`);
    console.log(`   Expected: ${formatPct(strat.expectedReturn)} | Risk: ${strat.risk} ${riskEmoji}`);
    console.log(`   ${c.dim}${strat.rationale}${c.reset}`);
  } else {
    console.log(`\n${c.dim}No clear opportunity at this time${c.reset}`);
  }
  
  console.log('\n' + '━'.repeat(60));
  console.log(`${c.dim}Next refresh in 30s... (Ctrl+C to exit)${c.reset}\n`);
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

