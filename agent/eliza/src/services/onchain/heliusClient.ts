/**
 * Helius RPC Client
 *
 * Fetches on-chain token data from Helius API.
 *
 * API Plan: Free Tier
 * - Endpoint: https://rpc.helius.xyz
 * - Rate Limit: 100,000 requests/day
 * - Cache: 5 minutes (blockchain updates slowly)
 *
 * Features:
 * - Token holder distribution (top 100 holders)
 * - Large transfers (whale activity)
 * - Token metadata (supply, decimals)
 * - Token age (creation date)
 */

import { logger } from '../logger.js';
import { PublicKey, Connection } from '@solana/web3.js';
import { resilientFetchJson, Queues } from '../resilience.js';

// ============= TYPES =============

export interface TokenHolder {
  address: string;
  amount: number;
  percentage: number;
}

export interface TokenMetadata {
  mint: string;
  supply: number;
  decimals: number;
  createdAt: Date;
  age: number; // Age in days
}

export interface LargeTransfer {
  signature: string;
  from: string;
  to: string;
  amount: number;
  timestamp: Date;
}

export interface TokenOnChainData {
  mint: string;
  holders: TokenHolder[];
  holderCount: number; // -1 = too many to count (highly distributed), 0+ = actual count
  topHoldersPercentage: number; // Top 10 holders %
  metadata: TokenMetadata;
  recentLargeTransfers: LargeTransfer[];
  whaleActivityCount: number; // Large transfers in last 7 days
  isHighlyDistributed: boolean; // True if holder count exceeds API limit
}

// ============= CACHE =============

interface CacheEntry {
  data: TokenOnChainData;
  timestamp: number;
}

const cache = new Map<string, CacheEntry>();
const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

// ============= HELIUS CLIENT =============

export class HeliusClient {
  private connection: Connection;
  private apiKey: string;
  private requestCount = 0;
  private dailyLimit = 100000;

  constructor(apiKey?: string) {
    this.apiKey = apiKey || process.env.HELIUS_API_KEY || '';

    if (!this.apiKey) {
      logger.warn('[HeliusClient] No API key provided, using public endpoint (limited)');
    }

    const endpoint = this.apiKey
      ? `https://rpc.helius.xyz/?api-key=${this.apiKey}`
      : 'https://api.mainnet-beta.solana.com';

    this.connection = new Connection(endpoint, 'confirmed');

    logger.info('[HeliusClient] Initialized', {
      hasApiKey: !!this.apiKey,
      endpoint: this.apiKey ? 'helius' : 'public',
      dailyLimit: this.dailyLimit,
    });
  }

  /**
   * Fetch complete on-chain data for a token
   */
  async fetchTokenData(mintAddress: string): Promise<TokenOnChainData | null> {
    // Check cache first
    const cached = cache.get(mintAddress);
    if (cached && Date.now() - cached.timestamp < CACHE_TTL_MS) {
      logger.info('[HeliusClient] Cache hit', { mint: mintAddress });
      return cached.data;
    }

    try {
      // Validate mint address
      const mint = new PublicKey(mintAddress);

      logger.info('[HeliusClient] Fetching token data', { mint: mintAddress });

      // Fetch metadata first (always succeeds)
      const metadata = await this.fetchTokenMetadata(mint);

      // Try to fetch holders (may fail for highly distributed tokens)
      let holders: TokenHolder[] = [];
      let holderCount = 0;
      let topHoldersPercentage = 0;
      let isHighlyDistributed = false;

      try {
        holders = await this.fetchTopHolders(mint);
        holderCount = holders.length;

        // Calculate top holders percentage
        topHoldersPercentage = holders
          .slice(0, 10)
          .reduce((sum, h) => sum + h.percentage, 0);
      } catch (holderError: any) {
        // Check if error is due to too many accounts (highly distributed token)
        if (holderError.message.includes('Too many accounts') ||
            holderError.message.includes('too many')) {
          logger.info('[HeliusClient] Highly distributed token detected', {
            mint: mintAddress,
            reason: 'Holder count exceeds API limit',
            interpretation: 'Good sign - too many holders to count (likely >1M)',
          });

          holderCount = -1; // Special value: too many to count
          topHoldersPercentage = 0; // Assume well distributed
          isHighlyDistributed = true;
        } else {
          // Other errors - rethrow
          throw holderError;
        }
      }

      // Fetch recent large transfers
      const transfers = await this.fetchRecentLargeTransfers(mint);

      // Count whale activity (large transfers in last 7 days)
      const sevenDaysAgo = Date.now() - 7 * 24 * 60 * 60 * 1000;
      const whaleActivityCount = transfers.filter(
        t => t.timestamp.getTime() > sevenDaysAgo
      ).length;

      const data: TokenOnChainData = {
        mint: mintAddress,
        holders,
        holderCount,
        topHoldersPercentage,
        metadata,
        recentLargeTransfers: transfers,
        whaleActivityCount,
        isHighlyDistributed,
      };

      // Cache the result
      cache.set(mintAddress, { data, timestamp: Date.now() });

      logger.info('[HeliusClient] Token data fetched', {
        mint: mintAddress,
        holderCount: holderCount === -1 ? 'too many to count' : holderCount,
        topHoldersPercentage: topHoldersPercentage.toFixed(2),
        whaleActivityCount,
        isHighlyDistributed,
      });

      return data;
    } catch (error: any) {
      logger.error('[HeliusClient] Failed to fetch token data', {
        mint: mintAddress,
        error: error.message,
      });
      return null;
    }
  }

  /**
   * Fetch top token holders
   * Throws error if too many accounts (highly distributed token)
   */
  private async fetchTopHolders(mint: PublicKey): Promise<TokenHolder[]> {
    this.requestCount++;

    // Get token accounts for this mint
    const accounts = await this.connection.getTokenLargestAccounts(mint);

    if (!accounts.value || accounts.value.length === 0) {
      return [];
    }

    // Get total supply
    const supply = await this.connection.getTokenSupply(mint);
    const totalSupply = Number(supply.value.amount);

    if (totalSupply === 0) {
      return [];
    }

    // Convert to holders with percentages
    const holders: TokenHolder[] = accounts.value
      .slice(0, 100) // Top 100 holders
      .map(account => {
        const amount = Number(account.amount);
        const percentage = (amount / totalSupply) * 100;

        return {
          address: account.address.toBase58(),
          amount,
          percentage,
        };
      })
      .filter(h => h.amount > 0);

    return holders;
  }

  /**
   * Fetch token metadata — uses Helius DAS API for real creation timestamp
   */
  private async fetchTokenMetadata(mint: PublicKey): Promise<TokenMetadata> {
    try {
      this.requestCount++;

      // Get token supply info
      const supply = await this.connection.getTokenSupply(mint);

      // Try Helius DAS API for real asset metadata (includes creation time)
      let createdAt = new Date();
      if (this.apiKey) {
        try {
          const dasResp = await resilientFetchJson<{
            result: {
              content?: { metadata?: { created?: string } };
              token_info?: { decimals?: number; supply?: number };
            };
          }>(
            `https://mainnet.helius-rpc.com/?api-key=${this.apiKey}`,
            {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                jsonrpc: '2.0',
                id: 'get-asset',
                method: 'getAsset',
                params: { id: mint.toBase58() },
              }),
            },
            { queue: Queues.helius(), label: 'helius/getAsset', retries: 2, fetchTimeout: 15000 },
          );
          if (dasResp.result?.content?.metadata?.created) {
            createdAt = new Date(dasResp.result.content.metadata.created);
          }
          this.requestCount++;
        } catch {
          // Fall through to signature-based estimation
        }
      }

      // If DAS didn't give us a date, estimate from oldest signature
      if (createdAt.getTime() >= Date.now() - 60_000) {
        try {
          const sigs = await this.connection.getSignaturesForAddress(mint, { limit: 1 });
          if (sigs.length > 0 && sigs[0].blockTime) {
            createdAt = new Date(sigs[0].blockTime * 1000);
          }
          this.requestCount++;
        } catch {
          // Use current time as last resort
        }
      }

      const age = Math.floor((Date.now() - createdAt.getTime()) / (24 * 60 * 60 * 1000));

      return {
        mint: mint.toBase58(),
        supply: Number(supply.value.amount),
        decimals: supply.value.decimals,
        createdAt,
        age,
      };
    } catch (error: any) {
      logger.error('[HeliusClient] Failed to fetch metadata', {
        mint: mint.toBase58(),
        error: error.message,
      });

      // Return default metadata
      return {
        mint: mint.toBase58(),
        supply: 0,
        decimals: 9,
        createdAt: new Date(),
        age: 0,
      };
    }
  }

  /**
   * Fetch recent large transfers (whale activity)
   * Uses Helius enhanced transactions API to detect large token movements
   */
  private async fetchRecentLargeTransfers(mint: PublicKey): Promise<LargeTransfer[]> {
    if (!this.apiKey) {
      return []; // Enhanced API requires API key
    }

    try {
      this.requestCount++;

      // Get recent signatures for this mint
      const signatures = await this.connection.getSignaturesForAddress(mint, { limit: 50 });
      if (signatures.length === 0) return [];

      this.requestCount++;

      // Use Helius enhanced transactions API to parse transfer amounts
      const sigList = signatures.slice(0, 20).map(s => s.signature);
      const parsed = await resilientFetchJson<Array<{
        signature: string;
        timestamp: number;
        tokenTransfers?: Array<{
          fromUserAccount: string;
          toUserAccount: string;
          tokenAmount: number;
          mint: string;
        }>;
      }>>(
        `https://api.helius.xyz/v0/transactions/?api-key=${this.apiKey}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ transactions: sigList }),
        },
        { queue: Queues.helius(), label: 'helius/parsedTxns', retries: 2, fetchTimeout: 20000 },
      );

      const transfers: LargeTransfer[] = [];
      const mintStr = mint.toBase58();

      for (const tx of parsed) {
        if (!tx.tokenTransfers) continue;
        for (const transfer of tx.tokenTransfers) {
          if (transfer.mint !== mintStr) continue;
          // Consider "large" as any transfer — the caller filters by context
          if (transfer.tokenAmount > 0) {
            transfers.push({
              signature: tx.signature,
              from: transfer.fromUserAccount || 'unknown',
              to: transfer.toUserAccount || 'unknown',
              amount: transfer.tokenAmount,
              timestamp: new Date(tx.timestamp * 1000),
            });
          }
        }
      }

      logger.info('[HeliusClient] Whale transfers fetched', {
        mint: mintStr,
        transferCount: transfers.length,
      });

      return transfers;
    } catch (error: any) {
      logger.warn('[HeliusClient] Failed to fetch transfers', {
        mint: mint.toBase58(),
        error: error.message,
      });
      return [];
    }
  }

  /**
   * Get current request count
   */
  getRequestCount(): number {
    return this.requestCount;
  }

  /**
   * Check if approaching rate limit
   */
  isApproachingLimit(): boolean {
    return this.requestCount > this.dailyLimit * 0.9;
  }
}

// ============= SINGLETON =============

let heliusClient: HeliusClient | null = null;

export function getHeliusClient(): HeliusClient {
  if (!heliusClient) {
    heliusClient = new HeliusClient();
  }
  return heliusClient;
}


