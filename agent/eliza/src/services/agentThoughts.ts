/**
 * Agent Thought Process Logger
 * Logs detailed reasoning for each opportunity evaluation
 */

import type { ArbitrageOpportunity, LPPool } from './marketScanner/types.js';
import { logger } from './logger.js';

// ============= EXCHANGE/POOL LINKS =============

const EXCHANGE_LINKS: Record<string, (symbol: string) => string> = {
  binance: (s) => `https://www.binance.com/en/trade/${s}_USDT`,
  coinbase: (s) => `https://www.coinbase.com/advanced-trade/spot/${s}-USD`,
  kraken: (s) => `https://www.kraken.com/prices/${s.toLowerCase()}`,
};

// Always use DexScreener for pool links - most reliable and works for all DEXes
const DEX_LINKS: Record<string, (poolAddress: string) => string> = {
  orca: (addr) => `https://dexscreener.com/solana/${addr}`,
  raydium: (addr) => `https://dexscreener.com/solana/${addr}`,
  meteora: (addr) => `https://dexscreener.com/solana/${addr}`,
  jupiter: () => `https://jup.ag/swap/USDC-SOL`,
  dexscreener: (addr) => `https://dexscreener.com/solana/${addr}`,
};

export function getExchangeLink(exchange: string, symbol: string): string {
  const fn = EXCHANGE_LINKS[exchange.toLowerCase()];
  return fn ? fn(symbol) : `https://www.google.com/search?q=${exchange}+${symbol}`;
}

export function getPoolLink(dex: string, poolAddress: string): string {
  const fn = DEX_LINKS[dex.toLowerCase()];
  return fn ? fn(poolAddress) : `https://dexscreener.com/solana/${poolAddress}`;
}

// ============= THOUGHT TEMPLATES =============

const ARB_THOUGHTS = {
  lowSpread: [
    "Spread yeterli değil, fees'i zar zor karşılıyor",
    "Kar marjı çok ince, risk almaya değmez",
    "Slippage ile negatife düşebilir",
  ],
  highSpread: [
    "Bu kadar yüksek spread gerçekçi değil, data hatası olmalı",
    "Muhtemelen düşük likidite veya stale price",
    "Gerçek execution'da bu fiyatı alamazsın",
  ],
  dexToCex: [
    "DEX→CEX yönü riskli, deposit 15 dakika sürer",
    "Fiyat bu sürede çok değişebilir",
    "CEX→DEX çok daha güvenli",
  ],
  approved: [
    "İyi spread, yeterli profit, güvenli yön",
    "Riski kabul edilebilir seviyede",
    "Execution hızlı olmalı",
  ],
  lowProfit: [
    "Kar çok düşük, işlem masraflarına değmez",
    "Minimum kar eşiğini geçmiyor",
    "Daha büyük fırsatlar beklemeli",
  ],
};

const LP_THOUGHTS = {
  highApy: [
    "APY çok yüksek, sürdürülebilir görünmüyor",
    "Muhtemelen yeni pool veya düşük TVL",
    "IL riski çok fazla olabilir",
  ],
  lowTvl: [
    "TVL çok düşük, likidite riski var",
    "Büyük pozisyon alamazsın",
    "Exit zor olabilir",
  ],
  lowVolume: [
    "İşlem hacmi yetersiz, fee geliri düşük",
    "Pool aktif değil",
    "APY sürdürülebilir değil",
  ],
  approved: [
    "Makul APY ve yeterli TVL",
    "Volume/TVL oranı sağlıklı",
    "Güvenli pool görünüyor",
  ],
  approvedButLower: [
    "Güvenli pool ama return arbitrage'dan düşük",
    "Uzun vadeli strateji için uygun",
    "Pasif gelir kaynağı olabilir",
  ],
};

function randomThought(arr: string[]): string {
  return arr[Math.floor(Math.random() * arr.length)];
}

// ============= LOGGING FUNCTIONS =============

export function logArbitrageEvaluation(
  arb: ArbitrageOpportunity & { buyPoolAddress?: string; sellPoolAddress?: string },
  approved: boolean,
  rejectReason: string | undefined,
  riskAdjustedReturn: number,
  positionSize?: number
): void {
  const CEX = ['binance', 'coinbase', 'kraken'];
  const isCexBuy = CEX.includes(arb.buyExchange.toLowerCase());
  const isCexSell = CEX.includes(arb.sellExchange.toLowerCase());

  let direction: string;
  if (isCexBuy && !isCexSell) {
    direction = 'CEX→DEX ✅ (güvenli)';
  } else if (!isCexBuy && isCexSell) {
    direction = 'DEX→CEX ⚠️ (riskli)';
  } else if (isCexBuy && isCexSell) {
    direction = 'CEX→CEX ⚠️ (transfer gerekli)';
  } else {
    direction = 'DEX→DEX ⚠️ (arbitrage yok)';
  }

  logger.info(`\n[AGENT] 🤔 Evaluating: ${arb.symbol} arbitrage +${arb.spreadPct.toFixed(1)}%`);
  logger.info(`  → Spread: ${arb.spreadPct.toFixed(1)}% (after fees: ~${(arb.spreadPct - 0.5).toFixed(1)}%)`);
  logger.info(`  → Profit: $${arb.netProfit.toFixed(2)} | Direction: ${direction}`);

  if (approved) {
    logger.info(`  → Position: $${positionSize?.toFixed(0) || '?'} | Score: ${riskAdjustedReturn.toFixed(2)}`);
    logger.info(`  💭 "${randomThought(ARB_THOUGHTS.approved)}"`);
    logger.info(`  ✅ APPROVED`);
    logger.info(`\n  📍 BUY: ${arb.buyExchange}`);
    logger.info(`     Link: ${isCexBuy ? getExchangeLink(arb.buyExchange, arb.symbol) : getPoolLink(arb.buyExchange, arb.buyPoolAddress || '')}`);
    logger.info(`     Price: $${arb.buyPrice > 0 ? arb.buyPrice.toFixed(4) : 'N/A'}`);
    logger.info(`  📍 SELL: ${arb.sellExchange}`);
    logger.info(`     Link: ${isCexSell ? getExchangeLink(arb.sellExchange, arb.symbol) : getPoolLink(arb.sellExchange, arb.sellPoolAddress || '')}`);
    logger.info(`     Price: $${arb.sellPrice > 0 ? arb.sellPrice.toFixed(4) : 'N/A'}`);
  } else{
    // Determine thought based on rejection reason
    let thought: string;
    if (rejectReason?.includes('too low') && rejectReason?.includes('Spread')) {
      thought = randomThought(ARB_THOUGHTS.lowSpread);
    } else if (rejectReason?.includes('unrealistic')) {
      thought = randomThought(ARB_THOUGHTS.highSpread);
    } else if (rejectReason?.includes('DEX→CEX')) {
      thought = randomThought(ARB_THOUGHTS.dexToCex);
    } else if (rejectReason?.includes('Profit')) {
      thought = randomThought(ARB_THOUGHTS.lowProfit);
    } else {
      thought = rejectReason || 'Bilinmeyen sebep';
    }
    logger.info(`  💭 "${thought}"`);
    logger.info(`  ❌ REJECTED: ${rejectReason}`);
  }
}

export function logLPEvaluation(
  pool: LPPool,
  approved: boolean,
  rejectReason: string | undefined,
  riskAdjustedReturn: number,
  riskLevel: string
): void {
  const apy = pool.apy || 0;
  const tvl = pool.tvl || 0;
  const vol = pool.volume24h || 0;
  const tvlM = tvl / 1e6;
  const volM = vol / 1e6;
  const volTvl = tvl > 0 ? vol / tvl : 0;

  logger.info(`\n[AGENT] 🤔 Evaluating: LP ${pool.name} [${pool.dex}] +${apy.toFixed(0)}% APY`);
  logger.info(`  → TVL: $${tvlM.toFixed(1)}M | Volume: $${volM.toFixed(1)}M | V/TVL: ${volTvl.toFixed(2)}`);
  logger.info(`  → APY: ${apy.toFixed(0)}% | Risk: ${riskLevel}`);
  
  if (approved) {
    logger.info(`  → Score: ${riskAdjustedReturn.toFixed(2)}`);
    logger.info(`  💭 "${randomThought(LP_THOUGHTS.approved)}"`);
    logger.info(`  ✅ APPROVED`);
    logger.info(`\n  📍 POOL: ${pool.dex}`);
    logger.info(`     Link: ${getPoolLink(pool.dex, pool.address)}`);
    logger.info(`     TVL: $${tvlM.toFixed(1)}M | Volume: $${volM.toFixed(1)}M`);
  } else {
    let thought: string;
    if (rejectReason?.includes('APY too high')) {
      thought = randomThought(LP_THOUGHTS.highApy);
    } else if (rejectReason?.includes('TVL too low')) {
      thought = randomThought(LP_THOUGHTS.lowTvl);
    } else if (rejectReason?.includes('Volume/TVL')) {
      thought = randomThought(LP_THOUGHTS.lowVolume);
    } else {
      thought = rejectReason || 'Filtrelendi';
    }
    logger.info(`  💭 "${thought}"`);
    logger.info(`  ❌ REJECTED: ${rejectReason}`);
  }
}

