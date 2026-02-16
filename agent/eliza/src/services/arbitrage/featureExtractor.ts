/**
 * Arbitrage Feature Extractor
 * 
 * Extracts features for ML model inference from arbitrage opportunities.
 * Matches the Python cross_dex_features.py feature engineering.
 * 
 * Model expects 27 features (from cross_dex_metadata.json):
 * - Spread features: spread_ma_12, spread_std_12, spread_ma_24, etc.
 * - Volume features: total_volume, volume_ma_12, volume_ratio, etc.
 * - Price features: v3_price, v2_price, price_ma_12, price_volatility
 * - Cost features: gas_gwei, gas_cost_usd, slippage, dex_fees_pct, gas_cost_pct
 * - Time features: hour, day_of_week, is_weekend
 */

import type { ArbitrageOpportunity } from '../marketScanner/types.js';
import { getSolPrice } from '../marketData.js';
import { logger } from '../logger.js';

// Feature names expected by the ONNX model (from metadata)
export const ARBITRAGE_FEATURE_NAMES = [
  'spread_ma_12', 'spread_std_12', 'spread_ma_24', 'spread_std_24',
  'spread_ma_48', 'spread_std_48', 'spread_change', 'spread_pct_change',
  'total_volume', 'volume_ma_12', 'volume_ma_24', 'volume_ma_48', 'volume_ratio',
  'v3_volume', 'v2_volume', 'v3_price', 'v2_price',
  'price_ma_12', 'price_volatility',
  'gas_gwei', 'gas_cost_usd', 'slippage', 'dex_fees_pct', 'gas_cost_pct',
  'hour', 'day_of_week', 'is_weekend'
] as const;

export type ArbitrageFeatureName = typeof ARBITRAGE_FEATURE_NAMES[number];

/**
 * Raw features extracted from opportunity and market data
 */
export interface ArbitrageFeatures {
  spread_ma_12: number;
  spread_std_12: number;
  spread_ma_24: number;
  spread_std_24: number;
  spread_ma_48: number;
  spread_std_48: number;
  spread_change: number;
  spread_pct_change: number;
  total_volume: number;
  volume_ma_12: number;
  volume_ma_24: number;
  volume_ma_48: number;
  volume_ratio: number;
  v3_volume: number;
  v2_volume: number;
  v3_price: number;
  v2_price: number;
  price_ma_12: number;
  price_volatility: number;
  gas_gwei: number;
  gas_cost_usd: number;
  slippage: number;
  dex_fees_pct: number;
  gas_cost_pct: number;
  hour: number;
  day_of_week: number;
  is_weekend: number;
}

/**
 * Historical data for rolling calculations
 */
interface HistoricalData {
  spreads: number[];
  volumes: number[];
  prices: number[];
  timestamps: Date[];
}

/**
 * Arbitrage Feature Extractor
 * 
 * Maintains rolling windows for feature calculation and extracts
 * features matching the Python model's expectations.
 */
export class ArbitrageFeatureExtractor {
  private history: Map<string, HistoricalData> = new Map();
  private readonly maxHistorySize = 100;
  
  // Solana cost parameters (matching Python config)
  private readonly raydiumFeePct = 0.25;  // 0.25%
  private readonly orcaFeePct = 0.30;     // 0.30%
  private readonly baseTxFeeLamports = 5000;
  private readonly priorityFeeLamports = 50000;
  
  constructor() {
    logger.info('[ArbitrageFeatureExtractor] Initialized');
  }

  /**
   * Extract features from an arbitrage opportunity
   */
  async extractFeatures(
    opportunity: ArbitrageOpportunity
  ): Promise<Float32Array> {
    const symbol = opportunity.symbol;
    const now = new Date();

    // Fetch live SOL price for cost calculations
    const solPrice = await getSolPrice();

    // Get or create history for this symbol
    let hist = this.history.get(symbol);
    if (!hist) {
      hist = { spreads: [], volumes: [], prices: [], timestamps: [] };
      this.history.set(symbol, hist);
    }

    // Add current data to history
    const currentSpread = opportunity.spreadPct;
    const currentVolume = this.estimateVolume(opportunity);
    const currentPrice = (opportunity.buyPrice + opportunity.sellPrice) / 2;

    hist.spreads.push(currentSpread);
    hist.volumes.push(currentVolume);
    hist.prices.push(currentPrice);
    hist.timestamps.push(now);

    // Trim history to max size
    if (hist.spreads.length > this.maxHistorySize) {
      hist.spreads.shift();
      hist.volumes.shift();
      hist.prices.shift();
      hist.timestamps.shift();
    }

    // Calculate features
    const features = this.calculateFeatures(opportunity, hist, solPrice, now);

    // Convert to Float32Array in correct order
    return this.toFloat32Array(features);
  }

  /**
   * Calculate all features from opportunity and history
   */
  private calculateFeatures(
    opp: ArbitrageOpportunity,
    hist: HistoricalData,
    solPrice: number,
    now: Date
  ): ArbitrageFeatures {
    const spreads = hist.spreads;
    const volumes = hist.volumes;
    const prices = hist.prices;
    
    // Spread features (rolling windows)
    const spread_ma_12 = this.rollingMean(spreads, 12);
    const spread_std_12 = this.rollingStd(spreads, 12);
    const spread_ma_24 = this.rollingMean(spreads, 24);
    const spread_std_24 = this.rollingStd(spreads, 24);
    const spread_ma_48 = this.rollingMean(spreads, 48);
    const spread_std_48 = this.rollingStd(spreads, 48);
    
    // Spread change
    const prevSpread = spreads.length > 1 ? spreads[spreads.length - 2] : opp.spreadPct;
    const spread_change = opp.spreadPct - prevSpread;
    const spread_pct_change = prevSpread > 0 ? (spread_change / prevSpread) * 100 : 0;
    
    // Volume features
    const total_volume = this.sum(volumes);
    const volume_ma_12 = this.rollingMean(volumes, 12);
    const volume_ma_24 = this.rollingMean(volumes, 24);
    const volume_ma_48 = this.rollingMean(volumes, 48);
    const currentVol = volumes[volumes.length - 1] || 0;
    const volume_ratio = volume_ma_24 > 0 ? currentVol / volume_ma_24 : 1;
    
    // DEX-specific volumes (estimate 50/50 split for now)
    const v3_volume = currentVol * 0.5;
    const v2_volume = currentVol * 0.5;
    
    // Price features
    const v3_price = opp.sellPrice;  // Higher price (sell side)
    const v2_price = opp.buyPrice;   // Lower price (buy side)
    const price_ma_12 = this.rollingMean(prices, 12);
    const price_volatility = this.rollingStd(prices, 12) / (price_ma_12 || 1);
    
    // Cost features (Solana-specific)
    const gas_gwei = 0;  // Solana doesn't use gwei, set to 0
    const txCostSol = (this.baseTxFeeLamports + this.priorityFeeLamports) * 2 / 1e9;
    const gas_cost_usd = txCostSol * solPrice;
    const slippage = 0.1;  // Estimate 0.1% slippage
    const dex_fees_pct = this.raydiumFeePct + this.orcaFeePct;
    const tradeSize = 10000;  // Assume $10k trade
    const gas_cost_pct = (gas_cost_usd / tradeSize) * 100;
    
    // Time features
    const hour = now.getUTCHours();
    const day_of_week = now.getUTCDay();
    const is_weekend = (day_of_week === 0 || day_of_week === 6) ? 1 : 0;
    
    return {
      spread_ma_12, spread_std_12, spread_ma_24, spread_std_24,
      spread_ma_48, spread_std_48, spread_change, spread_pct_change,
      total_volume, volume_ma_12, volume_ma_24, volume_ma_48, volume_ratio,
      v3_volume, v2_volume, v3_price, v2_price,
      price_ma_12, price_volatility,
      gas_gwei, gas_cost_usd, slippage, dex_fees_pct, gas_cost_pct,
      hour, day_of_week, is_weekend
    };
  }

  /**
   * Convert features object to Float32Array in correct order
   */
  private toFloat32Array(features: ArbitrageFeatures): Float32Array {
    const arr = new Float32Array(ARBITRAGE_FEATURE_NAMES.length);
    for (let i = 0; i < ARBITRAGE_FEATURE_NAMES.length; i++) {
      const name = ARBITRAGE_FEATURE_NAMES[i];
      arr[i] = features[name] ?? 0;
    }
    return arr;
  }

  /**
   * Estimate volume from opportunity (if not provided)
   */
  private estimateVolume(opp: ArbitrageOpportunity): number {
    // Estimate based on net profit and spread
    if (opp.spreadPct > 0 && opp.netProfit > 0) {
      return (opp.netProfit / opp.spreadPct) * 100;
    }
    return 10000;  // Default $10k
  }

  /**
   * Calculate rolling mean
   */
  private rollingMean(arr: number[], window: number): number {
    if (arr.length === 0) return 0;
    const slice = arr.slice(-Math.min(window, arr.length));
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  }

  /**
   * Calculate rolling standard deviation
   */
  private rollingStd(arr: number[], window: number): number {
    if (arr.length < 2) return 0;
    const slice = arr.slice(-Math.min(window, arr.length));
    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
    const variance = slice.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / slice.length;
    return Math.sqrt(variance);
  }

  /**
   * Calculate sum
   */
  private sum(arr: number[]): number {
    return arr.reduce((a, b) => a + b, 0);
  }

  /**
   * Clear history for a symbol
   */
  clearHistory(symbol?: string): void {
    if (symbol) {
      this.history.delete(symbol);
    } else {
      this.history.clear();
    }
  }

  /**
   * Get feature names
   */
  getFeatureNames(): readonly string[] {
    return ARBITRAGE_FEATURE_NAMES;
  }
}
