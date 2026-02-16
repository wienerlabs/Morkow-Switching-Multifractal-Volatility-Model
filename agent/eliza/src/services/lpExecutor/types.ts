/**
 * LP Executor Types
 * 
 * Common types for LP execution across all DEXs (Orca, Raydium, Meteora)
 */

import type { Keypair, PublicKey } from '@solana/web3.js';
import { config as prodConfig } from '../../config/production.js';

// Supported DEXs
export type SupportedDex = 'orca' | 'raydium' | 'meteora';

// Token information
export interface TokenInfo {
  mint: string;
  symbol: string;
  decimals: number;
}

// Pool information for LP execution
export interface LPPoolInfo {
  address: string;
  name: string;
  dex: SupportedDex;
  token0: {
    symbol: string;
    mint: string;
    decimals: number;
  };
  token1: {
    symbol: string;
    mint: string;
    decimals: number;
  };
  fee: number;        // Fee in basis points
  tvlUsd: number;
  apy: number;
}

// Deposit parameters
export interface DepositParams {
  pool: LPPoolInfo;
  amountUsd: number;          // USD amount to deposit
  slippageBps?: number;       // Slippage tolerance in basis points (default: 50 = 0.5%)
  wallet: Keypair;
  portfolioValueUsd?: number; // Total portfolio value for PM approval calculations
}

// Deposit result
export interface DepositResult {
  success: boolean;
  txSignature?: string;
  positionId?: string;        // Position NFT or LP token mint
  portfolioPositionId?: string; // PortfolioManager position ID for tracking
  lpTokenMint?: string;
  lpTokenBalance?: number;
  amountToken0?: number;
  amountToken1?: number;
  priceImpactPct?: number;
  feesUsd?: number;
  error?: string;
}

// Withdraw parameters
export interface WithdrawParams {
  positionId: string;         // Position NFT address or LP token mint
  portfolioPositionId?: string; // PortfolioManager position ID for tracking
  pool: LPPoolInfo;
  percentage?: number;        // Percentage to withdraw (default: 100%)
  slippageBps?: number;
  wallet: Keypair;
}

// Withdraw result
export interface WithdrawResult {
  success: boolean;
  txSignature?: string;
  amountToken0?: number;
  amountToken1?: number;
  amountUsd?: number;
  feesCollected?: number;
  error?: string;
}

// Position information
export interface PositionInfo {
  positionId: string;
  pool?: LPPoolInfo;
  poolAddress?: string;
  token0?: TokenInfo;
  token1?: TokenInfo;
  token0Amount: number;
  token1Amount: number;
  valueUsd?: number;
  liquidity?: string;
  feesEarnedUsd?: number;
  feesEarned?: { token0: number; token1: number };
  unrealizedPnlUsd?: number;
  entryTime?: number;
  // Concentrated liquidity specific
  priceLower?: number;
  priceUpper?: number;
  inRange?: boolean;
}

// Price impact calculation
export interface PriceImpactResult {
  impactPct: number;
  expectedOutput?: number;
  minimumOutput?: number;
  estimatedSlippage?: number;
  isAcceptable: boolean;    // true if impact < 1%
}

// DEX-specific executor interface
export interface IDexExecutor {
  readonly dex: SupportedDex;
  
  // Check if we can execute on this DEX
  isSupported(pool: LPPoolInfo): boolean;
  
  // Deposit liquidity
  deposit(params: DepositParams): Promise<DepositResult>;
  
  // Withdraw liquidity
  withdraw(params: WithdrawParams): Promise<WithdrawResult>;
  
  // Get position details
  getPosition(positionId: string, wallet: PublicKey): Promise<PositionInfo | null>;
  
  // Calculate price impact before execution
  calculatePriceImpact(pool: LPPoolInfo, amountUsd: number): Promise<PriceImpactResult>;
}

// Execution configuration
export interface ExecutorConfig {
  rpcUrl: string;
  defaultSlippageBps: number;
  depositSlippageBps?: number;   // Override for deposit (from SLIPPAGE_LP_DEPOSIT_BPS)
  withdrawSlippageBps?: number;  // Override for withdraw (from SLIPPAGE_LP_WITHDRAW_BPS)
  maxPriceImpactPct: number;
  confirmationTimeout: number;  // ms
  priorityFeeLamports?: number;
  dryRun?: boolean;  // If true, simulate trades without executing
}

// Default configuration — reads from centralized config (production.ts)
export const DEFAULT_EXECUTOR_CONFIG: ExecutorConfig = {
  rpcUrl: prodConfig.solanaRpcUrl,
  defaultSlippageBps: prodConfig.slippage.lpDeposit,
  depositSlippageBps: prodConfig.slippage.lpDeposit,    // 100 BPS from env
  withdrawSlippageBps: prodConfig.slippage.lpWithdraw,  // 50 BPS from env
  maxPriceImpactPct: 1.0,       // Max 1% price impact
  confirmationTimeout: 60000,   // 60 seconds
  priorityFeeLamports: 100000,  // 0.0001 SOL priority fee
};

// Token mints (common tokens)
export const TOKEN_MINTS = {
  SOL: 'So11111111111111111111111111111111111111112',
  BTC: '9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E',
  USDC: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
  USDT: 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',
  BONK: 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
  JUP: 'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN',
  WIF: 'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm',
  MSOL: 'mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So',
  STSOL: '7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj',
  JITOSOL: 'J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn',
  RAY: '4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R',
  ORCA: 'orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE',
} as const;

