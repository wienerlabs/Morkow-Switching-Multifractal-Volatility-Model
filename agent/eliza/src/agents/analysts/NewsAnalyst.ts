/**
 * NewsAnalyst - Dedicated News Analysis Component
 * 
 * Separates news analysis from SpeculationAnalyst for:
 * - Better news impact analysis
 * - Real-time news monitoring
 * - Sentiment vs news separation
 * - News-based trade signals
 * 
 * Sources:
 * - CryptoPanic API (crypto-specific news)
 * - RSS Feeds (CoinDesk, CoinTelegraph, The Block, Decrypt)
 * - On-chain events (large transfers, whale alerts)
 */

import { BaseAnalyst, DEFAULT_ANALYST_CONFIG, type AnalystConfig } from './BaseAnalyst.js';
import { getCryptoPanicCollector, type NewsItem as CryptoPanicNewsItem } from '../../services/sentiment/cryptopanicCollector.js';
import { getNewsScorer, type NewsImpactScore, type TradingAction } from '../../services/news/newsScorer.js';
import { getNewsClassifier, type ClassificationResult, NewsType } from '../../services/news/newsClassifier.js';
import { logger } from '../../services/logger.js';

// ============= TYPES =============

export interface NewsAnalysisInput {
  /** Assets to analyze news for */
  assets: string[];
  /** Hours of news to fetch (default: 24) */
  lookbackHours?: number;
}

export interface NewsOpportunityResult {
  type: 'news';
  name: string;
  asset: string;
  impact: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL' | 'MIXED';
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  immediateImpact: number; // -100 to +100
  expectedReturn: number;
  riskScore: number;
  confidence: number;
  riskAdjustedReturn: number;
  approved: boolean;
  rejectReason?: string;
  warnings: string[];
  tradingAction: TradingAction;
  newsCount: number;
  topNews: NewsItemWithScore[];
  raw: {
    scores: NewsImpactScore[];
    aggregateScore: number;
  };
}

export interface NewsItemWithScore {
  title: string;
  description: string;
  url: string;
  publishedAt: string;
  score: NewsImpactScore;
}

export interface NewsAnalystConfig extends AnalystConfig {
  /** Minimum absolute impact score to generate opportunity (default: 20) */
  minImpactScore: number;
  /** Maximum news items to consider per asset (default: 10) */
  maxNewsItems: number;
  /** Hours of news to look back (default: 24) */
  defaultLookbackHours: number;
  /** Critical impact threshold for immediate alerts (default: -50) */
  criticalThreshold: number;
}

// ============= CONSTANTS =============

/**
 * DX-Research Task 1: Prospect Theory loss aversion multiplier.
 * Negative news impacts markets ~267x faster (0.03 min vs 8.03 min — DX Terminal finding).
 * Kahneman & Tversky empirical value: ~2.25. Conservative default: 1.75.
 */
const PROSPECT_THEORY_LOSS_AVERSION = parseFloat(
  process.env.PROSPECT_THEORY_LOSS_AVERSION ?? '1.75'
);

export const DEFAULT_NEWS_CONFIG: NewsAnalystConfig = {
  ...DEFAULT_ANALYST_CONFIG,
  minConfidence: 0.4,
  minImpactScore: 20,
  maxNewsItems: 10,
  defaultLookbackHours: 24,
  criticalThreshold: -50,
};

// ============= NEWS ANALYST =============

export class NewsAnalyst extends BaseAnalyst<NewsAnalysisInput, NewsOpportunityResult> {
  private newsConfig: NewsAnalystConfig;
  private cryptoPanic = getCryptoPanicCollector();
  private scorer = getNewsScorer();
  private classifier = getNewsClassifier();

  constructor(config: Partial<NewsAnalystConfig> = {}) {
    const mergedConfig = { ...DEFAULT_NEWS_CONFIG, ...config };
    super(mergedConfig);
    this.newsConfig = mergedConfig;
    logger.info('[NewsAnalyst] Initialized', { config: this.newsConfig });
  }

  getName(): string {
    return 'NewsAnalyst';
  }

  /**
   * Analyze news for given assets
   */
  async analyze(input: NewsAnalysisInput): Promise<NewsOpportunityResult[]> {
    const results: NewsOpportunityResult[] = [];
    const lookbackHours = input.lookbackHours || this.newsConfig.defaultLookbackHours;

    logger.info('[NewsAnalyst] Analyzing news', { 
      assets: input.assets, 
      lookbackHours 
    });

    // Fetch and analyze news for each asset in parallel
    const analysisPromises = input.assets.map(asset => 
      this.analyzeAsset(asset, lookbackHours).catch(error => {
        logger.warn('[NewsAnalyst] Failed to analyze asset', { asset, error: error.message });
        return null;
      })
    );

    const analysisResults = await Promise.all(analysisPromises);
    
    for (const result of analysisResults) {
      if (result) {
        results.push(result);
      }
    }

    // Sort by absolute impact (most impactful first)
    results.sort((a, b) => Math.abs(b.immediateImpact) - Math.abs(a.immediateImpact));

    logger.info('[NewsAnalyst] Analysis complete', {
      totalAssets: input.assets.length,
      resultsGenerated: results.length,
      criticalAlerts: results.filter(r => r.severity === 'CRITICAL').length,
    });

    return results;
  }

  /**
   * Analyze news for a single asset
   */
  private async analyzeAsset(asset: string, lookbackHours: number): Promise<NewsOpportunityResult | null> {
    // Fetch news from CryptoPanic using fetchPosts method
    const sentiment = await this.cryptoPanic.fetchPosts(asset);

    if (!sentiment || sentiment.newsCount === 0) {
      logger.debug('[NewsAnalyst] No news found', { asset });
      return null;
    }

    // Convert and score news items
    const newsItems = sentiment.topNews.slice(0, this.newsConfig.maxNewsItems);
    const scoredItems: NewsItemWithScore[] = [];

    for (const item of newsItems) {
      const score = this.scorer.scoreNews({
        title: item.title,
        description: item.description,
        url: item.url,
        publishedAt: item.publishedAt,
        assets: [asset],
      });

      scoredItems.push({
        title: item.title,
        description: item.description,
        url: item.url,
        publishedAt: item.publishedAt,
        score,
      });
    }

    // Aggregate scores
    const { scores, aggregate } = this.scorer.scoreBatch(
      newsItems.map(item => ({
        title: item.title,
        description: item.description,
        publishedAt: item.publishedAt,
        assets: [asset],
      }))
    );

    // Skip if impact below threshold
    if (Math.abs(aggregate) < this.newsConfig.minImpactScore) {
      logger.debug('[NewsAnalyst] Impact below threshold', {
        asset,
        aggregate,
        threshold: this.newsConfig.minImpactScore
      });
      return null;
    }

    // Determine overall impact direction
    const impact = this.determineImpact(aggregate);
    const severity = this.determineSeverity(aggregate, scores);

    // Calculate trading metrics
    const expectedReturn = this.calculateExpectedReturn(aggregate);
    const riskScore = this.calculateRiskScore(aggregate, scores);
    const confidence = this.calculateConfidence(scores, sentiment.newsCount);
    const riskAdjustedReturn = expectedReturn / riskScore;

    // Determine trading action
    const tradingAction = scores[0]?.tradingAction || 'HOLD';

    // Validation
    const warnings: string[] = [];
    let rejectReason: string | undefined;

    if (confidence < this.newsConfig.minConfidence) {
      rejectReason = `Confidence too low: ${(confidence * 100).toFixed(0)}%`;
    }

    if (severity === 'CRITICAL' && aggregate < 0) {
      warnings.push('CRITICAL: Negative news detected - consider immediate exit');
    }

    if (sentiment.newsCount < 3) {
      warnings.push('Low news volume - signal may be unreliable');
    }

    const approved = !rejectReason;

    return {
      type: 'news',
      name: `${impact} News Signal: ${asset}`,
      asset,
      impact,
      severity,
      immediateImpact: aggregate,
      expectedReturn,
      riskScore,
      confidence,
      riskAdjustedReturn,
      approved,
      rejectReason,
      warnings,
      tradingAction,
      newsCount: sentiment.newsCount,
      topNews: scoredItems.slice(0, 5),
      raw: { scores, aggregateScore: aggregate },
    };
  }

  /**
   * Determine impact direction from aggregate score
   */
  private determineImpact(score: number): 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL' | 'MIXED' {
    if (score >= 30) return 'POSITIVE';
    if (score <= -30) return 'NEGATIVE';
    if (Math.abs(score) < 10) return 'NEUTRAL';
    return 'MIXED';
  }

  /**
   * Determine severity from scores
   */
  private determineSeverity(
    aggregate: number,
    scores: NewsImpactScore[]
  ): 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' {
    // Check if any individual score is critical
    if (scores.some(s => s.severity === 'CRITICAL')) return 'CRITICAL';
    if (aggregate <= this.newsConfig.criticalThreshold) return 'CRITICAL';
    if (scores.some(s => s.severity === 'HIGH')) return 'HIGH';
    if (Math.abs(aggregate) >= 50) return 'HIGH';
    if (Math.abs(aggregate) >= 30) return 'MEDIUM';
    return 'LOW';
  }

  /**
   * Calculate expected return based on news impact
   */
  private calculateExpectedReturn(aggregate: number): number {
    // DX-Research Task 1: Prospect Theory asymmetric weighting.
    // Negative aggregate gets amplified by loss aversion lambda — markets react
    // to bad news ~267x faster than good news (DX Terminal finding).
    const adjusted = aggregate < 0
      ? aggregate * PROSPECT_THEORY_LOSS_AVERSION
      : aggregate;
    return adjusted / 10;
  }

  /**
   * Calculate risk score from news analysis
   */
  private calculateRiskScore(aggregate: number, scores: NewsImpactScore[]): number {
    // DX-Research Task 1: Prospect Theory — negative news carries amplified risk.
    const effectiveAggregate = aggregate < 0
      ? aggregate * PROSPECT_THEORY_LOSS_AVERSION
      : aggregate;

    let risk = 5; // Base risk

    if (effectiveAggregate < 0) risk += 2;
    if (effectiveAggregate < -50) risk += 2;

    // Check for high-risk news types
    const hasSecurityNews = scores.some(
      s => s.classification.type === NewsType.SECURITY
    );
    const hasRegulatoryNews = scores.some(
      s => s.classification.type === NewsType.REGULATORY
    );

    if (hasSecurityNews) risk += 2;
    if (hasRegulatoryNews) risk += 1;

    return Math.min(10, Math.max(1, risk));
  }

  /**
   * Calculate confidence based on news volume and consistency
   */
  private calculateConfidence(scores: NewsImpactScore[], newsCount: number): number {
    // Base confidence on news count
    let confidence = Math.min(newsCount / 10, 0.5); // Max 50% from volume

    // Add confidence from score consistency
    if (scores.length >= 2) {
      const allSameDirection = scores.every(s =>
        (s.immediateImpact > 0) === (scores[0].immediateImpact > 0)
      );
      if (allSameDirection) confidence += 0.3;
    }

    // Add confidence from classification confidence
    const avgClassConfidence = scores.reduce(
      (sum, s) => sum + s.classification.confidence, 0
    ) / scores.length;
    confidence += avgClassConfidence * 0.2;

    return Math.min(1, confidence);
  }
}

