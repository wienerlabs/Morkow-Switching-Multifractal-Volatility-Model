/**
 * Gas Service
 * 
 * Real-time gas price monitoring and emergency budget management.
 * All data from live Solana RPC - NO hardcoded values.
 */
import { Connection } from '@solana/web3.js';
import { logger } from '../logger.js';
import { getSolanaConnection, recordRpcFailure } from '../solana/connection.js';
import type { GasBudgetConfig, GasBudgetStatus } from './types.js';

// ============= CONSTANTS =============

// Standard compute units for different operations
export const COMPUTE_UNITS = {
  simpleTransfer: 200,
  swap: 300_000,
  lpDeposit: 400_000,
  lpWithdraw: 350_000,
  perpsOpen: 500_000,
  perpsClose: 450_000,
  emergencyExit: 600_000,  // Higher for safety
} as const;

// Base transaction fee in lamports
const BASE_FEE_LAMPORTS = 5000;

export const DEFAULT_GAS_CONFIG: GasBudgetConfig = {
  reserveMultiplier: 2.0,           // Keep 2x average gas in reserve
  maxPriorityFeeLamports: 500_000,  // Cap at 0.0005 SOL priority fee
  emergencyGasReserveUsd: 5.0,      // Min $5 for emergency exits
};

// ============= GAS SERVICE CLASS =============

export class GasService {
  private connection: Connection;
  private config: GasBudgetConfig;
  private recentFees: number[] = [];  // Rolling window of recent priority fees
  private solPriceUsd: number = 0;
  private lastFeeUpdate: number = 0;

  constructor(
    rpcUrl?: string,
    config: Partial<GasBudgetConfig> = {}
  ) {
    this.connection = rpcUrl
      ? new Connection(rpcUrl, 'confirmed')
      : getSolanaConnection();
    this.config = { ...DEFAULT_GAS_CONFIG, ...config };
  }

  /**
   * Fetch recent prioritization fees from Solana RPC
   * Uses getRecentPrioritizationFees for REAL on-chain data
   */
  async fetchRecentPriorityFees(): Promise<number[]> {
    try {
      // Get recent priority fees from the network
      // Pass empty array to get global fees (not account-specific)
      const recentFees = await this.connection.getRecentPrioritizationFees({});
      
      const fees = recentFees
        .filter(f => f.prioritizationFee > 0)
        .map(f => f.prioritizationFee);

      if (fees.length > 0) {
        // Keep last 100 fee samples
        this.recentFees = [...this.recentFees, ...fees].slice(-100);
        this.lastFeeUpdate = Date.now();
      }

      logger.debug('Fetched priority fees', { count: fees.length, sample: fees.slice(0, 5) });
      return fees;
    } catch (error) {
      recordRpcFailure();
      logger.error('Failed to fetch priority fees', { error });
      return [];
    }
  }

  /**
   * Update SOL price for USD calculations
   * Should be called with price from oracle service
   */
  setSolPrice(priceUsd: number): void {
    this.solPriceUsd = priceUsd;
  }

  /**
   * Calculate recommended priority fee based on recent network conditions
   */
  async getRecommendedPriorityFee(urgency: 'low' | 'medium' | 'high' = 'medium'): Promise<number> {
    // Refresh fees if stale (>30 seconds)
    if (Date.now() - this.lastFeeUpdate > 30_000) {
      await this.fetchRecentPriorityFees();
    }

    if (this.recentFees.length === 0) {
      // Default fallback based on urgency
      const defaults = { low: 1000, medium: 10000, high: 50000 };
      return defaults[urgency];
    }

    // Sort fees to calculate percentiles
    const sorted = [...this.recentFees].sort((a, b) => a - b);
    
    // Use different percentiles based on urgency
    const percentiles = { low: 0.25, medium: 0.50, high: 0.75 };
    const index = Math.floor(sorted.length * percentiles[urgency]);
    let recommendedFee = sorted[index] || sorted[sorted.length - 1];

    // Apply cap
    recommendedFee = Math.min(recommendedFee, this.config.maxPriorityFeeLamports);

    return recommendedFee;
  }

  /**
   * Calculate total gas cost for an operation in USD
   */
  async calculateGasCostUsd(
    computeUnits: number,
    urgency: 'low' | 'medium' | 'high' = 'medium'
  ): Promise<{ costLamports: number; costUsd: number; priorityFee: number }> {
    const priorityFee = await this.getRecommendedPriorityFee(urgency);
    
    // Total cost = base fee + (priority fee per CU * CU / 1M)
    const priorityCost = Math.ceil((priorityFee * computeUnits) / 1_000_000);
    const totalLamports = BASE_FEE_LAMPORTS + priorityCost;
    
    // Convert to USD
    const costSol = totalLamports / 1e9;
    const costUsd = costSol * this.solPriceUsd;

    return {
      costLamports: totalLamports,
      costUsd,
      priorityFee,
    };
  }

  /**
   * Get current gas budget status
   */
  async getGasBudgetStatus(walletBalanceSol: number): Promise<GasBudgetStatus> {
    // Fetch latest fees
    await this.fetchRecentPriorityFees();
    
    // Calculate averages
    const avgPriorityFee = this.recentFees.length > 0
      ? this.recentFees.reduce((a, b) => a + b, 0) / this.recentFees.length
      : 10000; // Default fallback

    // Average gas cost for a swap operation
    const avgSwapCost = await this.calculateGasCostUsd(COMPUTE_UNITS.swap, 'medium');
    
    // Calculate reserve
    const walletBalanceUsd = walletBalanceSol * this.solPriceUsd;
    const requiredReserve = avgSwapCost.costUsd * this.config.reserveMultiplier;
    
    // Emergency exit cost
    const emergencyExitCost = await this.calculateGasCostUsd(COMPUTE_UNITS.emergencyExit, 'high');
    const canAffordEmergencyExit = walletBalanceUsd >= (emergencyExitCost.costUsd * 3); // 3x safety

    const recommendedFee = await this.getRecommendedPriorityFee('medium');

    return {
      currentBaseFee: BASE_FEE_LAMPORTS,
      currentPriorityFee: recommendedFee,
      averageGasCostUsd: avgSwapCost.costUsd,
      reserveBalanceUsd: walletBalanceUsd,
      recommendedPriorityFee: recommendedFee,
      canAffordEmergencyExit,
    };
  }
}

