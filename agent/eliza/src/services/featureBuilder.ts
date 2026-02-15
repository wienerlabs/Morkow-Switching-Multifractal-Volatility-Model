/**
 * Feature Builder Service
 * 
 * Fetches OHLCV data and builds 61 ML features for LP pool evaluation.
 * Implements intelligent caching and parallel fetching with rate limiting.
 * 
 * UPDATED: 2026-01-07 - Full 61-feature model integration
 */

import { BirdeyeProvider, TOKENS, type OHLCVData } from '../providers/birdeye.js';
import { engineerFeatures } from '../features/engineer.js';
import type { PoolFeatures } from '../inference/model.js';
import type { LPPool } from './marketScanner/types.js';
import { getSolPrice } from './marketData.js';
import { logger } from './logger.js';

// Cache configuration
const CACHE_TTL_MS = 3 * 60 * 1000; // 3 minutes
const TOKEN_OHLCV_CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes for token prices

// Batch configuration for rate limiting
const BATCH_SIZE = 5; // Pools per batch
const BATCH_DELAY_MS = 2000; // 2 seconds between batches

interface CachedFeatures {
  features: PoolFeatures;
  timestamp: number;
}

interface CachedOHLCV {
  data: OHLCVData[];
  timestamp: number;
}

// Token price history for market features
interface TokenPriceHistory {
  SOL: number[];
  USDC: number[];
  USDT: number[];
}

class FeatureBuilderService {
  private provider: BirdeyeProvider | null = null;
  private featureCache: Map<string, CachedFeatures> = new Map();
  private ohlcvCache: Map<string, CachedOHLCV> = new Map();
  private tokenPriceHistory: TokenPriceHistory = { SOL: [], USDC: [], USDT: [] };
  private tokenHistoryLastUpdate = 0;

  /**
   * Initialize with Birdeye API key
   */
  initialize(apiKey: string): void {
    this.provider = new BirdeyeProvider(apiKey);
    logger.info('[FeatureBuilder] Initialized with Birdeye API');
  }

  /**
   * Check if initialized
   */
  isInitialized(): boolean {
    return this.provider !== null;
  }

  /**
   * Get features for a pool (with caching)
   */
  async getFeaturesForPool(pool: LPPool): Promise<PoolFeatures | null> {
    const cacheKey = pool.address;
    const cached = this.featureCache.get(cacheKey);

    // Return cached if fresh
    if (cached && Date.now() - cached.timestamp < CACHE_TTL_MS) {
      return cached.features;
    }

    if (!this.provider) {
      logger.warn('[FeatureBuilder] Not initialized - using fallback features');
      return await this.getFallbackFeatures(pool);
    }

    try {
      // Fetch OHLCV for pool's token pair
      const tokenAddress = this.getTokenAddress(pool);
      if (!tokenAddress) {
        logger.warn(`[FeatureBuilder] Unknown token for pool ${pool.name}`);
        return await this.getFallbackFeatures(pool);
      }

      // Get OHLCV data (168 hours = 7 days for all features)
      const ohlcv = await this.getOHLCVWithCache(tokenAddress, 168);
      if (ohlcv.length === 0) {
        logger.warn(`[FeatureBuilder] No OHLCV data for ${pool.name}`);
        return await this.getFallbackFeatures(pool);
      }

      // Ensure token price history is up to date
      await this.updateTokenPriceHistory();

      // Engineer features
      const features = engineerFeatures(ohlcv, this.tokenPriceHistory, new Date());

      // Update TVL-related features with actual pool data
      features.tvl_proxy = pool.tvl;
      features.tvl_ma_24h = pool.tvl; // Approximation
      features.vol_tvl_ratio = pool.tvl > 0 ? pool.volume24h / pool.tvl : 0;
      features.vol_tvl_ma_24h = features.vol_tvl_ratio;

      // Cache the result
      this.featureCache.set(cacheKey, { features, timestamp: Date.now() });

      return features;
    } catch (error) {
      logger.error(`[FeatureBuilder] Error fetching features for ${pool.name}`, { error: String(error) });
      return await this.getFallbackFeatures(pool);
    }
  }

  /**
   * Batch fetch features for multiple pools with rate limiting
   */
  async batchGetFeatures(pools: LPPool[]): Promise<Map<string, PoolFeatures>> {
    const results = new Map<string, PoolFeatures>();

    // First, update token price history (shared across all pools)
    await this.updateTokenPriceHistory();

    // Process in batches to respect rate limits
    for (let i = 0; i < pools.length; i += BATCH_SIZE) {
      const batch = pools.slice(i, i + BATCH_SIZE);
      
      logger.info(`[FeatureBuilder] Processing batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(pools.length / BATCH_SIZE)}`);

      // Parallel fetch within batch
      const batchResults = await Promise.all(
        batch.map(async (pool) => {
          const features = await this.getFeaturesForPool(pool);
          return { address: pool.address, features };
        })
      );

      for (const { address, features } of batchResults) {
        if (features) {
          results.set(address, features);
        }
      }

      // Delay between batches (except last)
      if (i + BATCH_SIZE < pools.length) {
        await new Promise(resolve => setTimeout(resolve, BATCH_DELAY_MS));
      }
    }

    logger.info(`[FeatureBuilder] Completed: ${results.size}/${pools.length} pools with features`);
    return results;
  }

  /**
   * Get OHLCV data with caching
   */
  private async getOHLCVWithCache(tokenAddress: string, hours: number): Promise<OHLCVData[]> {
    const cacheKey = `${tokenAddress}:${hours}`;
    const cached = this.ohlcvCache.get(cacheKey);

    if (cached && Date.now() - cached.timestamp < CACHE_TTL_MS) {
      return cached.data;
    }

    if (!this.provider) return [];

    try {
      const now = Math.floor(Date.now() / 1000);
      const from = now - hours * 3600;
      const data = await this.provider.getOHLCV(tokenAddress, '1h', from, now);

      this.ohlcvCache.set(cacheKey, { data, timestamp: Date.now() });
      return data;
    } catch (error) {
      logger.error(`[FeatureBuilder] OHLCV fetch failed for ${tokenAddress}`, { error: String(error) });
      return cached?.data ?? [];
    }
  }

  /**
   * Update token price history for SOL, USDC, USDT
   */
  private async updateTokenPriceHistory(): Promise<void> {
    if (Date.now() - this.tokenHistoryLastUpdate < TOKEN_OHLCV_CACHE_TTL_MS) {
      return; // Still fresh
    }

    if (!this.provider) return;

    try {
      const [solOhlcv, usdcOhlcv, usdtOhlcv] = await Promise.all([
        this.getOHLCVWithCache(TOKENS.SOL, 168),
        this.getOHLCVWithCache(TOKENS.USDC, 168),
        this.getOHLCVWithCache(TOKENS.USDT, 168),
      ]);

      this.tokenPriceHistory = {
        SOL: solOhlcv.map(d => d.close),
        USDC: usdcOhlcv.map(d => d.close),
        USDT: usdtOhlcv.map(d => d.close),
      };

      this.tokenHistoryLastUpdate = Date.now();
      logger.info(`[FeatureBuilder] Token history updated: SOL=${this.tokenPriceHistory.SOL.length}h, USDC=${this.tokenPriceHistory.USDC.length}h`);
    } catch (error) {
      logger.error('[FeatureBuilder] Failed to update token history', { error: String(error) });
    }
  }

  /**
   * Get token address from pool
   */
  private getTokenAddress(pool: LPPool): string | null {
    // Map pool tokens to known addresses
    const token0Upper = pool.token0.toUpperCase();
    const token1Upper = pool.token1.toUpperCase();

    // Prefer the non-stable token for OHLCV (more interesting price action)
    if (token0Upper === 'SOL' || token1Upper === 'SOL') return TOKENS.SOL;
    if (token0Upper === 'JUP' || token1Upper === 'JUP') return TOKENS.JUP;
    if (token0Upper === 'RAY' || token1Upper === 'RAY') return TOKENS.RAY;
    if (token0Upper === 'ORCA' || token1Upper === 'ORCA') return TOKENS.ORCA;
    if (token0Upper === 'BONK' || token1Upper === 'BONK') return TOKENS.BONK;

    // Fallback to first token if it has an address
    return null;
  }

  /**
   * Generate fallback features when API unavailable
   * Uses dynamic SOL price from market data service
   */
  private async getFallbackFeatures(pool: LPPool): Promise<PoolFeatures> {
    const now = new Date();
    const hour = now.getUTCHours();
    const day = now.getUTCDay();

    // Get dynamic SOL price (cached, with fallback)
    const solPrice = await getSolPrice();

    return {
      volume_1h: pool.volume24h / 24,
      volume_ma_6h: pool.volume24h / 24,
      volume_ma_24h: pool.volume24h / 24,
      volume_ma_168h: pool.volume24h / 24,
      volume_trend_7d: 0,
      volume_volatility_24h: 5,
      price_close: 0,
      price_high: 0,
      price_low: 0,
      price_range: 0,
      price_range_pct: 0,
      price_ma_6h: 0,
      price_ma_24h: 0,
      price_ma_168h: 0,
      price_trend_7d: 0,
      price_volatility_24h: 5,
      price_volatility_168h: 5,
      price_return_1h: 0,
      price_return_6h: 0,
      price_return_24h: 0,
      price_return_168h: 0,
      tvl_proxy: pool.tvl,
      tvl_ma_24h: pool.tvl,
      tvl_stability_7d: 0.8,
      tvl_trend_7d: 0,
      vol_tvl_ratio: pool.tvl > 0 ? pool.volume24h / pool.tvl : 0,
      vol_tvl_ma_24h: pool.tvl > 0 ? pool.volume24h / pool.tvl : 0,
      il_estimate_24h: 0.5,
      il_estimate_7d: 1.0,
      il_change_24h: 0,
      hour_of_day: hour,
      day_of_week: day,
      is_weekend: day === 0 || day === 6 ? 1 : 0,
      hour_sin: Math.sin((2 * Math.PI * hour) / 24),
      hour_cos: Math.cos((2 * Math.PI * hour) / 24),
      day_sin: Math.sin((2 * Math.PI * day) / 7),
      day_cos: Math.cos((2 * Math.PI * day) / 7),
      SOL_price: solPrice,
      SOL_return_1h: 0,
      SOL_return_24h: 0,
      SOL_volatility_24h: 5,
      SOL_volatility_168h: 5,
      SOL_ma_6h: solPrice,
      SOL_ma_24h: solPrice,
      SOL_trend_7d: 0,
      USDC_price: 1,
      USDC_return_1h: 0,
      USDC_return_24h: 0,
      USDC_volatility_24h: 0.01,
      USDC_volatility_168h: 0.01,
      USDC_ma_6h: 1,
      USDC_ma_24h: 1,
      USDC_trend_7d: 0,
      USDT_price: 1,
      USDT_return_1h: 0,
      USDT_return_24h: 0,
      USDT_volatility_24h: 0.01,
      USDT_volatility_168h: 0.01,
      USDT_ma_6h: 1,
      USDT_ma_24h: 1,
      USDT_trend_7d: 0,
    };
  }

  /**
   * Clear all caches
   */
  clearCache(): void {
    this.featureCache.clear();
    this.ohlcvCache.clear();
    logger.info('[FeatureBuilder] Cache cleared');
  }

  /**
   * Get cache stats
   */
  getCacheStats(): { features: number; ohlcv: number } {
    return {
      features: this.featureCache.size,
      ohlcv: this.ohlcvCache.size,
    };
  }
}

// Singleton instance
export const featureBuilder = new FeatureBuilderService();

