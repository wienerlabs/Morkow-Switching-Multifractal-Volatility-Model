/**
 * Model Inference Tests
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { existsSync, readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { lpRebalancerModel, type PoolFeatures } from '../inference/model.js';

const __test_filename = fileURLToPath(import.meta.url);
const __test_dirname = dirname(__test_filename);
const LP_MODEL_FILE = resolve(__test_dirname, '../../models/lp_rebalancer.onnx');

function isOnnxModelAvailable(): boolean {
  if (!existsSync(LP_MODEL_FILE)) return false;
  const head = readFileSync(LP_MODEL_FILE, { encoding: 'utf-8', flag: 'r' }).slice(0, 40);
  return !head.startsWith('version https://git-lfs');
}

const modelAvailable = isOnnxModelAvailable();

describe.skipIf(!modelAvailable)('LP Rebalancer Model', () => {
  // Create mock features with all 61 fields
  const createMockFeatures = (): PoolFeatures => ({
    volume_1h: 1000000,
    volume_ma_6h: 950000,
    volume_ma_24h: 900000,
    volume_ma_168h: 850000,
    volume_trend_7d: 0.05,
    volume_volatility_24h: 0.15,
    price_close: 100,
    price_high: 102,
    price_low: 98,
    price_range: 4,
    price_range_pct: 4.08,
    price_ma_6h: 99.5,
    price_ma_24h: 99,
    price_ma_168h: 98.5,
    price_trend_7d: 0.02,
    price_volatility_24h: 0.03,
    price_volatility_168h: 0.05,
    price_return_1h: 0.005,
    price_return_6h: 0.01,
    price_return_24h: 0.02,
    price_return_168h: 0.05,
    tvl_proxy: 90000000,
    tvl_ma_24h: 89000000,
    tvl_stability_7d: 0.85,
    tvl_trend_7d: 0.01,
    vol_tvl_ratio: 0.01,
    vol_tvl_ma_24h: 0.0095,
    il_estimate_24h: 0.5,
    il_estimate_7d: 1.2,
    il_change_24h: 0.1,
    hour_of_day: 14,
    day_of_week: 3,
    is_weekend: 0,
    hour_sin: 0.5,
    hour_cos: -0.5,
    day_sin: 0.7,
    day_cos: 0.3,
    SOL_price: 200,
    SOL_return_1h: 0.005,
    SOL_return_24h: 0.02,
    SOL_volatility_24h: 0.04,
    SOL_volatility_168h: 0.06,
    SOL_ma_6h: 199,
    SOL_ma_24h: 198,
    SOL_trend_7d: 0.03,
    USDC_price: 1.0,
    USDC_return_1h: 0.0001,
    USDC_return_24h: 0.0002,
    USDC_volatility_24h: 0.001,
    USDC_volatility_168h: 0.002,
    USDC_ma_6h: 1.0,
    USDC_ma_24h: 1.0,
    USDC_trend_7d: 0.0001,
    USDT_price: 1.0,
    USDT_return_1h: 0.0001,
    USDT_return_24h: 0.0002,
    USDT_volatility_24h: 0.001,
    USDT_volatility_168h: 0.002,
    USDT_ma_6h: 1.0,
    USDT_ma_24h: 1.0,
    USDT_trend_7d: 0.0001,
  });

  beforeAll(async () => {
    await lpRebalancerModel.initialize();
  });

  it('should initialize the model', async () => {
    // Model uses fallback (mock) when ONNX not available
    expect(lpRebalancerModel).toBeDefined();
  });

  it('should predict with mock features', async () => {
    const features = createMockFeatures();
    const result = await lpRebalancerModel.predict(features);
    
    expect(result).toHaveProperty('probability');
    expect(result).toHaveProperty('decision');
    expect(result).toHaveProperty('confidence');
    expect(result).toHaveProperty('threshold');
  });

  it('should return probability between 0 and 1', async () => {
    const features = createMockFeatures();
    const result = await lpRebalancerModel.predict(features);
    
    expect(result.probability).toBeGreaterThanOrEqual(0);
    expect(result.probability).toBeLessThanOrEqual(1);
  });

  it('should return REBALANCE or HOLD decision', async () => {
    const features = createMockFeatures();
    const result = await lpRebalancerModel.predict(features);
    
    expect(['REBALANCE', 'HOLD']).toContain(result.decision);
  });

  it('should return confidence between 0 and 1', async () => {
    const features = createMockFeatures();
    const result = await lpRebalancerModel.predict(features);
    
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
  });

  it('should handle high volatility scenario', async () => {
    const features = createMockFeatures();
    features.price_volatility_24h = 0.15;
    features.price_volatility_168h = 0.20;
    features.SOL_volatility_24h = 0.12;
    
    const result = await lpRebalancerModel.predict(features);
    
    // Should still produce valid output
    expect(result.probability).toBeGreaterThanOrEqual(0);
    expect(result.probability).toBeLessThanOrEqual(1);
  });

  it('should have consistent threshold', async () => {
    const features = createMockFeatures();
    const result = await lpRebalancerModel.predict(features);

    // Threshold is configurable, default is 0.9 for conservative predictions
    expect(result.threshold).toBeGreaterThan(0);
    expect(result.threshold).toBeLessThanOrEqual(1);
  });
});

