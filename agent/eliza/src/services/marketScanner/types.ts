/**
 * Market Scanner Types
 */

import type { LendingMarketData } from '../lending/types.js';

export interface TokenPrice {
  symbol: string;
  address?: string;
  price: number;
  change24h?: number;
  volume24h?: number;
  source: string;
  timestamp: number;
}

export interface CEXPrice extends TokenPrice {
  exchange: 'binance' | 'coinbase' | 'kraken';
  bid?: number;
  ask?: number;
  spread?: number;
}

export interface DEXPrice extends TokenPrice {
  dex: 'raydium' | 'orca' | 'meteora' | 'jupiter' | 'birdeye' | 'dexscreener';
  poolAddress?: string;
  liquidity?: number;
}

// Multi-source price comparison
export interface MultiSourcePrice {
  symbol: string;
  cex: {
    binance?: number;
    coinbase?: number;
    kraken?: number;
  };
  dex: {
    birdeye?: number;
    jupiter?: number;
    dexscreener?: number;
  };
  coingecko?: number;
  cryptorank?: number;
  avgPrice: number;
  maxSpread: number;
  timestamp: number;
}

export interface ArbitrageOpportunity {
  symbol: string;
  buyExchange: string;
  sellExchange: string;
  buyPrice: number;
  sellPrice: number;
  spreadPct: number;
  estimatedProfit: number;
  fees: number;
  netProfit: number;
  confidence: 'high' | 'medium' | 'low';
  buyPoolAddress?: string;  // For DEX venues
  sellPoolAddress?: string; // For DEX venues
}

export interface LPPool {
  name: string;
  address: string;
  dex: string;
  token0: string;
  token1: string;
  apy: number;
  apr: number;
  tvl: number;
  volume24h: number;
  fees24h: number;
  feeRate: number;
  utilization: number;
  riskScore: number; // 1-10, lower is safer
}

export interface NewToken {
  symbol: string;
  name: string;
  address: string;
  launchTime: number;
  price: number;
  priceChange24h: number;
  liquidity: number;
  volume24h: number;
  holders?: number;
  socials?: {
    twitter?: string;
    telegram?: string;
    website?: string;
  };
  riskLevel: 'high' | 'medium' | 'low';
}

export interface ScannerConfig {
  tokens: string[];
  refreshInterval: number;
  minArbitrageSpread: number;
  minLiquidity: number;
  maxRiskScore: number;
}

/**
 * Perps funding rate opportunity
 * Represents a potential perps trade based on funding rates
 */
export interface PerpsOpportunity {
  market: string;                     // e.g., 'SOL-PERP'
  venue: 'drift' | 'jupiter' | 'flash' | 'adrena';
  side: 'long' | 'short';             // Direction to trade

  // Funding data
  fundingRate: number;                // Current hourly rate (e.g., -0.001 = -0.1%)
  annualizedRate: number;             // rate * 24 * 365
  nextFundingTime: number;            // Unix timestamp

  // Price data
  markPrice: number;
  indexPrice: number;
  perpSpotBasis: number;              // % diff between perp and spot

  // Opportunity metrics
  expectedReturnPct: number;          // Expected return from funding
  holdingPeriodHours: number;         // Recommended hold time
  estimatedProfitUsd: number;         // Profit for $1000 position

  // Risk metrics
  openInterest: number;               // Total OI in USD
  oiChangePct24h: number;             // OI change last 24h
  liquidityDepth: number;             // Available liquidity
  maxLeverage: number;                // Max allowed leverage

  // Scoring
  confidence: 'high' | 'medium' | 'low';
  riskScore: number;                  // 1-10
}

/**
 * Funding rate arbitrage between venues
 */
export interface FundingArbitrageOpportunity {
  market: string;
  longVenue: 'drift' | 'jupiter' | 'flash' | 'adrena';
  shortVenue: 'drift' | 'jupiter' | 'flash' | 'adrena';
  longRate: number;
  shortRate: number;
  netSpread: number;
  annualizedSpread: number;
  estimatedProfitBps: number;
  confidence: 'high' | 'medium' | 'low';
}

export interface GlobalMarketData {
  totalMarketCap: number;
  totalMarketCapChange: number;
  totalVolume24h: number;
  totalVolume24hChange: number;
  btcDominance: number;
  btcDominanceChange: number;
  ethDominance: number;
  ethDominanceChange: number;
  fearGreed: number;
  fearGreedChange: number;
  altcoinIndex: number;
  altcoinIndexChange: number;
  activeCurrencies: number;
}

export interface MarketSnapshot {
  timestamp: number;
  cexPrices: CEXPrice[];
  dexPrices: DEXPrice[];
  arbitrage: ArbitrageOpportunity[];
  lpPools: LPPool[];
  newTokens: NewToken[];
  perpsOpportunities: PerpsOpportunity[];           // Single-venue perps
  fundingArbitrage: FundingArbitrageOpportunity[];  // Cross-venue funding arb
  lendingMarkets: LendingMarketData[];              // Lending opportunities
  spotTokens: any[];                                 // Spot trading tokens (from Birdeye)
  cryptorankPrices: TokenPrice[];                    // CryptoRank price verification
  globalData: GlobalMarketData | null;               // Global market sentiment
  bestStrategy: {
    type: 'arbitrage' | 'lp' | 'token' | 'perps' | 'funding_arb' | 'lending' | 'spot';
    name: string;
    expectedReturn: number;
    risk: string;
    rationale: string;
  } | null;
}

