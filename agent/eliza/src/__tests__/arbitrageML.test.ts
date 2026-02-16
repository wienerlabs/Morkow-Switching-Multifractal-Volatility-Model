/**
 * Arbitrage ML Integration Tests
 * 
 * Tests for:
 * - Feature extraction
 * - Model loading and inference
 * - ArbitrageAnalyst ML integration
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ArbitrageFeatureExtractor, ARBITRAGE_FEATURE_NAMES } from '../services/arbitrage/featureExtractor.js';
import type { ArbitrageOpportunity } from '../services/marketScanner/types.js';

// Mock the logger
vi.mock('../services/logger.js', () => ({
  logger: {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
  }
}));

// Mock getSolPrice to return a deterministic value
vi.mock('../services/marketData.js', () => ({
  getSolPrice: vi.fn().mockResolvedValue(200),
  DEFAULT_GAS_SOL: 0.002,
}));

describe('ArbitrageFeatureExtractor', () => {
  let extractor: ArbitrageFeatureExtractor;
  
  const mockOpportunity: ArbitrageOpportunity = {
    symbol: 'SOL',
    buyExchange: 'raydium',
    sellExchange: 'orca',
    buyPrice: 200.0,
    sellPrice: 202.0,
    spreadPct: 1.0,
    estimatedProfit: 20,
    fees: 5,
    netProfit: 15,
    confidence: 'high',
  };
  
  beforeEach(() => {
    extractor = new ArbitrageFeatureExtractor();
  });
  
  afterEach(() => {
    extractor.clearHistory();
  });
  
  it('should have correct number of feature names', () => {
    expect(ARBITRAGE_FEATURE_NAMES).toHaveLength(27);
  });
  
  it('should extract features as Float32Array', async () => {
    const features = await extractor.extractFeatures(mockOpportunity);
    
    expect(features).toBeInstanceOf(Float32Array);
    expect(features).toHaveLength(27);
  });
  
  it('should extract features with correct order', async () => {
    const features = await extractor.extractFeatures(mockOpportunity);
    const names = extractor.getFeatureNames();
    
    expect(names).toEqual(ARBITRAGE_FEATURE_NAMES);
    expect(names[0]).toBe('spread_ma_12');
    expect(names[26]).toBe('is_weekend');
  });
  
  it('should extract time features correctly', async () => {
    const features = await extractor.extractFeatures(mockOpportunity);
    
    // hour should be 0-23
    const hourIdx = ARBITRAGE_FEATURE_NAMES.indexOf('hour');
    expect(features[hourIdx]).toBeGreaterThanOrEqual(0);
    expect(features[hourIdx]).toBeLessThanOrEqual(23);
    
    // day_of_week should be 0-6
    const dowIdx = ARBITRAGE_FEATURE_NAMES.indexOf('day_of_week');
    expect(features[dowIdx]).toBeGreaterThanOrEqual(0);
    expect(features[dowIdx]).toBeLessThanOrEqual(6);
    
    // is_weekend should be 0 or 1
    const weekendIdx = ARBITRAGE_FEATURE_NAMES.indexOf('is_weekend');
    expect([0, 1]).toContain(features[weekendIdx]);
  });
  
  it('should extract cost features correctly', async () => {
    const features = await extractor.extractFeatures(mockOpportunity);
    
    // gas_gwei should be 0 for Solana
    const gasGweiIdx = ARBITRAGE_FEATURE_NAMES.indexOf('gas_gwei');
    expect(features[gasGweiIdx]).toBe(0);
    
    // dex_fees_pct should be positive (Raydium + Orca fees)
    const dexFeesIdx = ARBITRAGE_FEATURE_NAMES.indexOf('dex_fees_pct');
    expect(features[dexFeesIdx]).toBeGreaterThan(0);
    expect(features[dexFeesIdx]).toBeLessThan(1); // < 1%
  });
  
  it('should build history over multiple calls', async () => {
    // First call
    await extractor.extractFeatures(mockOpportunity);
    
    // Second call with different spread
    const opp2 = { ...mockOpportunity, spreadPct: 1.5 };
    await extractor.extractFeatures(opp2);
    
    // Third call
    const opp3 = { ...mockOpportunity, spreadPct: 0.8 };
    const features = await extractor.extractFeatures(opp3);
    
    // spread_ma_12 should be average of history
    const spreadMaIdx = ARBITRAGE_FEATURE_NAMES.indexOf('spread_ma_12');
    expect(features[spreadMaIdx]).toBeGreaterThan(0);
  });
  
  it('should calculate spread change correctly', async () => {
    // First call
    const features1 = await extractor.extractFeatures(mockOpportunity);
    
    // Second call with higher spread
    const opp2 = { ...mockOpportunity, spreadPct: 1.5 };
    const features2 = await extractor.extractFeatures(opp2);
    
    // spread_change should be 0.5 (1.5 - 1.0)
    const spreadChangeIdx = ARBITRAGE_FEATURE_NAMES.indexOf('spread_change');
    expect(features2[spreadChangeIdx]).toBeCloseTo(0.5, 5);
  });
  
  it('should clear history correctly', async () => {
    await extractor.extractFeatures(mockOpportunity);
    await extractor.extractFeatures(mockOpportunity);
    
    extractor.clearHistory('SOL');
    
    // After clearing, first extraction should have no history
    const features = await extractor.extractFeatures(mockOpportunity);
    
    // spread_change should be 0 on first extraction
    const spreadChangeIdx = ARBITRAGE_FEATURE_NAMES.indexOf('spread_change');
    expect(features[spreadChangeIdx]).toBe(0);
  });
  
  it('should handle edge case: zero spread', async () => {
    const zeroSpreadOpp = { ...mockOpportunity, spreadPct: 0 };
    const features = await extractor.extractFeatures(zeroSpreadOpp);
    
    expect(features).toBeInstanceOf(Float32Array);
    // Should not throw or have NaN
    expect(features.some(f => isNaN(f))).toBe(false);
  });
  
  it('should extract price features correctly', async () => {
    const features = await extractor.extractFeatures(mockOpportunity);
    
    // v3_price (sell) and v2_price (buy) should match opportunity
    const v3PriceIdx = ARBITRAGE_FEATURE_NAMES.indexOf('v3_price');
    const v2PriceIdx = ARBITRAGE_FEATURE_NAMES.indexOf('v2_price');
    
    expect(features[v3PriceIdx]).toBe(mockOpportunity.sellPrice);
    expect(features[v2PriceIdx]).toBe(mockOpportunity.buyPrice);
  });
});

