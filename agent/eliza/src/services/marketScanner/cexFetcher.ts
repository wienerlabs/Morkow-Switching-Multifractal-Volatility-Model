/**
 * CEX Price Fetcher - Binance, Coinbase, Kraken
 *
 * Uses resilient fetch with retry + per-exchange rate-limited queues.
 */

import type { CEXPrice } from './types.js';
import { logger } from '../logger.js';
import { resilientFetch, Queues } from '../resilience.js';

const BINANCE_API = 'https://api.binance.com/api/v3';
const COINBASE_API = 'https://api.coinbase.com/v2';
const KRAKEN_API = 'https://api.kraken.com/0/public';

// Symbol mapping for each exchange
const SYMBOL_MAP: Record<string, Record<string, string>> = {
  binance: { SOL: 'SOLUSDT', BTC: 'BTCUSDT', ETH: 'ETHUSDT', WIF: 'WIFUSDT', BONK: 'BONKUSDT' },
  coinbase: { SOL: 'SOL-USD', BTC: 'BTC-USD', ETH: 'ETH-USD' },
  kraken: { SOL: 'SOLUSD', BTC: 'XXBTZUSD', ETH: 'XETHZUSD' },
};

export async function fetchBinancePrices(symbols: string[]): Promise<CEXPrice[]> {
  const prices: CEXPrice[] = [];
  const binanceSymbols = symbols.map(s => SYMBOL_MAP.binance[s]).filter(Boolean);

  if (binanceSymbols.length === 0) return prices;

  try {
    const url = `${BINANCE_API}/ticker/24hr?symbols=${JSON.stringify(binanceSymbols)}`;
    const resp = await resilientFetch(url, undefined, {
      queue: Queues.binance(),
      label: 'binance/ticker',
      retries: 3,
    });
    const data = await resp.json() as Array<{
      symbol: string; lastPrice: string; priceChangePercent: string;
      quoteVolume: string; bidPrice: string; askPrice: string;
    }>;

    for (const item of data) {
      const symbol = Object.entries(SYMBOL_MAP.binance).find(([, v]) => v === item.symbol)?.[0];
      if (!symbol) continue;

      const bid = parseFloat(item.bidPrice);
      const ask = parseFloat(item.askPrice);
      prices.push({
        symbol,
        price: parseFloat(item.lastPrice),
        change24h: parseFloat(item.priceChangePercent),
        volume24h: parseFloat(item.quoteVolume),
        source: 'binance',
        exchange: 'binance',
        bid, ask,
        spread: ((ask - bid) / bid) * 100,
        timestamp: Date.now(),
      });
    }
  } catch (e) {
    logger.error('[Binance] Fetch error after retries:', { error: String(e) });
  }
  return prices;
}

export async function fetchCoinbasePrices(symbols: string[]): Promise<CEXPrice[]> {
  const prices: CEXPrice[] = [];
  const queue = Queues.coinbase();

  for (const symbol of symbols) {
    const pair = SYMBOL_MAP.coinbase[symbol];
    if (!pair) continue;

    try {
      const resp = await resilientFetch(
        `${COINBASE_API}/prices/${pair}/spot`,
        undefined,
        { queue, label: `coinbase/${pair}`, retries: 2 },
      );

      const spotData = await resp.json() as { data: { amount: string } };
      const price = parseFloat(spotData.data.amount);

      prices.push({
        symbol,
        price,
        source: 'coinbase',
        exchange: 'coinbase',
        timestamp: Date.now(),
      });
    } catch (e) {
      logger.warn(`[Coinbase] ${symbol} failed after retries`, { error: String(e) });
    }
  }
  return prices;
}

export async function fetchKrakenPrices(symbols: string[]): Promise<CEXPrice[]> {
  const prices: CEXPrice[] = [];
  const krakenPairs = symbols.map(s => SYMBOL_MAP.kraken[s]).filter(Boolean);

  if (krakenPairs.length === 0) return prices;

  try {
    const url = `${KRAKEN_API}/Ticker?pair=${krakenPairs.join(',')}`;
    const resp = await resilientFetch(url, undefined, {
      queue: Queues.kraken(),
      label: 'kraken/ticker',
      retries: 3,
    });
    const data = await resp.json() as { result: Record<string, { c: string[]; b: string[]; a: string[]; v: string[] }> };

    for (const [pair, info] of Object.entries(data.result || {})) {
      const symbol = Object.entries(SYMBOL_MAP.kraken).find(([, v]) => v === pair)?.[0];
      if (!symbol) continue;

      const bid = parseFloat(info.b[0]);
      const ask = parseFloat(info.a[0]);
      prices.push({
        symbol,
        price: parseFloat(info.c[0]),
        volume24h: parseFloat(info.v[1]),
        source: 'kraken',
        exchange: 'kraken',
        bid, ask,
        spread: ((ask - bid) / bid) * 100,
        timestamp: Date.now(),
      });
    }
  } catch (e) {
    logger.warn('[Kraken] Fetch failed after retries', { error: String(e) });
  }
  return prices;
}

export async function fetchAllCEXPrices(symbols: string[]): Promise<CEXPrice[]> {
  const [binance, coinbase, kraken] = await Promise.all([
    fetchBinancePrices(symbols),
    fetchCoinbasePrices(symbols),
    fetchKrakenPrices(symbols),
  ]);
  return [...binance, ...coinbase, ...kraken];
}
