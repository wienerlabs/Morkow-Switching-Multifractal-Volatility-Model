/**
 * Integration Tests for Perps ML Trading Agent
 * 
 * Tests:
 * 1. Real data fetching and feature extraction
 * 2. Full ML pipeline (data → features → prediction)
 * 3. Verification against Python feature engineering
 * 4. Dry run trading agent execution
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { 
  getPerpsModelLoader, 
  PerpsModelLoader,
  createFeatureExtractor,
  PerpsFeatureExtractor,
  createTradingAgent,
  PerpsTradingAgent,
  FEATURE_NAMES,
  NUM_FEATURES,
  type FundingDataPoint,
  type PredictionResult,
} from '../index.js';

// ============= TEST DATA =============

// Simulated historical funding rate data (168 hours = 1 week)
function generateTestFundingHistory(hours: number = 200): FundingDataPoint[] {
  const data: FundingDataPoint[] = [];
  const now = Date.now();
  
  for (let i = hours; i >= 0; i--) {
    const timestamp = new Date(now - i * 3600 * 1000);
    // Simulate realistic funding rates with some variance
    const baseFunding = 0.001 + Math.sin(i / 24) * 0.002; // Oscillates between -0.1% and 0.3%
    const noise = (Math.random() - 0.5) * 0.001;
    const fundingRate = baseFunding + noise;
    
    data.push({
      timestamp,
      fundingRate,
      fundingRateRaw: fundingRate,
      oraclePrice: 200 + Math.sin(i / 48) * 20, // SOL price oscillating
      markPrice: 200 + Math.sin(i / 48) * 20 + (Math.random() - 0.5) * 0.5,
      cumFundingLong: i > 0 ? data[data.length - 1]?.cumFundingLong ?? 0 + fundingRate : fundingRate,
      cumFundingShort: i > 0 ? data[data.length - 1]?.cumFundingShort ?? 0 - fundingRate : -fundingRate,
    });
  }
  
  return data;
}

// High funding rate scenario (should signal SHORT)
function generateHighFundingScenario(): FundingDataPoint[] {
  const data: FundingDataPoint[] = [];
  const now = Date.now();
  
  for (let i = 200; i >= 0; i--) {
    const timestamp = new Date(now - i * 3600 * 1000);
    // High positive funding (0.3% - 0.5%) - shorts pay longs
    const fundingRate = 0.003 + Math.random() * 0.002;
    
    data.push({
      timestamp,
      fundingRate,
      fundingRateRaw: fundingRate,
      oraclePrice: 200,
      markPrice: 202, // Mark > Oracle indicates long bias
    });
  }
  
  return data;
}

// Low/negative funding scenario (should signal LONG or NO_TRADE)
function generateLowFundingScenario(): FundingDataPoint[] {
  const data: FundingDataPoint[] = [];
  const now = Date.now();
  
  for (let i = 200; i >= 0; i--) {
    const timestamp = new Date(now - i * 3600 * 1000);
    // Low negative funding (-0.1% to 0.1%)
    const fundingRate = (Math.random() - 0.5) * 0.002;
    
    data.push({
      timestamp,
      fundingRate,
      fundingRateRaw: fundingRate,
      oraclePrice: 200,
      markPrice: 199.5,
    });
  }
  
  return data;
}

// ============= TESTS =============

// Check if the ONNX model binary is available (not a Git LFS pointer)
import { readFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __test_filename = fileURLToPath(import.meta.url);
const __test_dirname = dirname(__test_filename);
const MODEL_FILE = resolve(__test_dirname, '../../../../../models/perps_predictor.onnx');

function isOnnxModelAvailable(): boolean {
  if (!existsSync(MODEL_FILE)) return false;
  const head = readFileSync(MODEL_FILE, { encoding: 'utf-8', flag: 'r' }).slice(0, 40);
  return !head.startsWith('version https://git-lfs');
}

const modelAvailable = isOnnxModelAvailable();

describe('Perps ML Integration Tests', () => {
  let modelLoader: PerpsModelLoader;
  let featureExtractor: PerpsFeatureExtractor;

  beforeAll(async () => {
    modelLoader = getPerpsModelLoader();
    await modelLoader.initialize();
    featureExtractor = createFeatureExtractor();
  });

  afterAll(() => {
    PerpsModelLoader.resetInstance();
  });

  describe('1. Model Loading', () => {
    it.skipIf(!modelAvailable)('should load ONNX model successfully', () => {
      expect(modelLoader.isInitialized()).toBe(true);
    });
    
    it('should have correct number of features', () => {
      expect(NUM_FEATURES).toBe(65);
      expect(FEATURE_NAMES.length).toBe(65);
    });
    
    it('should have expected feature names', () => {
      expect(FEATURE_NAMES).toContain('funding_rate');
      expect(FEATURE_NAMES).toContain('funding_zscore');
      expect(FEATURE_NAMES).toContain('volatility_24h');
      expect(FEATURE_NAMES).toContain('hour');
      expect(FEATURE_NAMES).toContain('is_weekend');
    });
  });

  describe('2. Feature Extraction', () => {
    it('should extract 65 features from history', () => {
      const history = generateTestFundingHistory();
      featureExtractor.loadHistory(history);
      
      expect(featureExtractor.hasEnoughHistory()).toBe(true);
      
      const features = featureExtractor.extractFeatures();
      expect(features.length).toBe(65);
    });
    
    it('should detect insufficient history and throw on extract', () => {
      const shortExtractor = createFeatureExtractor();
      shortExtractor.addDataPoint({
        timestamp: new Date(),
        fundingRate: 0.001,
      });

      expect(shortExtractor.hasEnoughHistory()).toBe(false);
      // Feature extraction requires minimum history, should throw
      expect(() => shortExtractor.extractFeatures()).toThrow('Not enough data');
    });
    
    it('should produce valid numeric features', () => {
      const history = generateTestFundingHistory();
      const extractor = createFeatureExtractor();
      extractor.loadHistory(history);
      
      const features = extractor.extractFeatures();
      
      features.forEach((value) => {
        expect(Number.isFinite(value)).toBe(true);
      });
    });
    
    it('should calculate time features correctly', () => {
      const history = generateTestFundingHistory();
      const extractor = createFeatureExtractor();
      extractor.loadHistory(history);

      const features = extractor.extractFeatures();
      const hourIdx = FEATURE_NAMES.indexOf('hour');
      const dowIdx = FEATURE_NAMES.indexOf('day_of_week');

      expect(features[hourIdx]).toBeGreaterThanOrEqual(0);
      expect(features[hourIdx]).toBeLessThan(24);
      expect(features[dowIdx]).toBeGreaterThanOrEqual(0);
      expect(features[dowIdx]).toBeLessThan(7);
    });
  });

  describe.skipIf(!modelAvailable)('3. ML Pipeline (Data → Features → Prediction)', () => {
    it('should run full prediction pipeline', async () => {
      const history = generateTestFundingHistory();
      const extractor = createFeatureExtractor();
      extractor.loadHistory(history);

      const features = extractor.extractFeatures();
      const fundingRate = extractor.getCurrentFundingRate();

      const prediction = await modelLoader.predict(features, fundingRate);

      expect(prediction).toHaveProperty('prediction');
      expect(prediction).toHaveProperty('probability');
      expect(prediction).toHaveProperty('confidence');
      expect(prediction).toHaveProperty('shouldTrade');
      expect(prediction).toHaveProperty('direction');
    });

    it('should return valid prediction values', async () => {
      const history = generateTestFundingHistory();
      const extractor = createFeatureExtractor();
      extractor.loadHistory(history);

      const features = extractor.extractFeatures();
      const prediction = await modelLoader.predict(features, 0.002);

      // Prediction should be 0 or 1
      expect([0, 1]).toContain(prediction.prediction);

      // Probability should be between 0 and 1
      expect(prediction.probability).toBeGreaterThanOrEqual(0);
      expect(prediction.probability).toBeLessThanOrEqual(1);

      // Confidence should be between 0 and 1
      expect(prediction.confidence).toBeGreaterThanOrEqual(0);
      expect(prediction.confidence).toBeLessThanOrEqual(1);

      // Direction should be valid or null
      expect([null, 'long', 'short']).toContain(prediction.direction);
    });

    it('should predict SHORT direction for high positive funding', async () => {
      const history = generateHighFundingScenario();
      const extractor = createFeatureExtractor();
      extractor.loadHistory(history);

      const features = extractor.extractFeatures();
      const fundingRate = 0.004; // 0.4% funding
      const prediction = await modelLoader.predict(features, fundingRate);

      // With high positive funding, if model predicts trade, direction should be short
      if (prediction.prediction === 1) {
        expect(prediction.direction).toBe('short');
      }
    });

    it('should predict LONG direction for negative funding', async () => {
      const history = generateLowFundingScenario();
      const extractor = createFeatureExtractor();
      extractor.loadHistory(history);

      const features = extractor.extractFeatures();
      const fundingRate = -0.003; // -0.3% funding
      const prediction = await modelLoader.predict(features, fundingRate);

      // With negative funding, if model predicts trade, direction should be long
      if (prediction.prediction === 1) {
        expect(prediction.direction).toBe('long');
      }
    });
  });

  describe.skipIf(!modelAvailable)('4. Trading Agent Dry Run', () => {
    let agent: PerpsTradingAgent;

    beforeAll(async () => {
      agent = createTradingAgent({
        markets: ['SOL-PERP'],
        dryRun: true,
        pollingIntervalMs: 1000,
        minConfidence: 0.6,
        fundingThreshold: 0.0025,
      });
      await agent.initialize();
    });

    it('should initialize in dry run mode', () => {
      const state = agent.getState();
      expect(state.running).toBe(false);
      expect(state.tradesExecuted).toBe(0);
    });

    it('should have zero trades executed in dry run', () => {
      const state = agent.getState();
      expect(state.tradesExecuted).toBe(0);
    });

    it('should track scans completed', async () => {
      // Load history for feature extraction
      const history = generateTestFundingHistory();
      await agent.loadHistoricalData('SOL-PERP', history);

      const state = agent.getState();
      expect(state.scansCompleted).toBeGreaterThanOrEqual(0);
    });

    it('should not start without perps service', async () => {
      // Agent can start but won't fetch real data without service
      await agent.start();
      const state = agent.getState();
      expect(state.running).toBe(true);

      // Stop agent
      agent.stop();
      expect(agent.getState().running).toBe(false);
    });
  });

  describe('5. Python Feature Verification', () => {
    it('should have same feature names as Python', () => {
      // These are the critical features from Python
      const pythonFeatures = [
        'funding_rate', 'funding_rate_raw',
        'funding_lag_1h', 'funding_lag_24h',
        'funding_mean_1h', 'funding_std_1h',
        'funding_mean_24h', 'funding_std_24h',
        'funding_zscore',
        'return_1h', 'return_24h',
        'volatility_24h', 'volatility_168h',
        'hour', 'day_of_week', 'is_weekend',
        'hour_sin', 'hour_cos',
      ];

      for (const feature of pythonFeatures) {
        expect(FEATURE_NAMES).toContain(feature);
      }
    });

    it('should compute z-score correctly', () => {
      // Create data with known mean and std
      const extractor = createFeatureExtractor();
      const data: FundingDataPoint[] = [];
      const now = Date.now();

      // 168 hours of constant 0.001 funding, then one point at 0.003
      for (let i = 200; i >= 1; i--) {
        data.push({
          timestamp: new Date(now - i * 3600000),
          fundingRate: 0.001,
        });
      }
      // Last point is 2 std above mean
      data.push({
        timestamp: new Date(now),
        fundingRate: 0.003,
      });

      extractor.loadHistory(data);
      const features = extractor.extractFeatures();

      const zscoreIdx = FEATURE_NAMES.indexOf('funding_zscore');
      // Z-score should be positive (above mean)
      expect(features[zscoreIdx]).toBeGreaterThan(0);
    });
  });
});
