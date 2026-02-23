import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';

vi.mock('../services/logger.js', () => ({
  logger: {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
  },
}));

// Mock resilience layer
const mockResilientFetchJson = vi.fn();
vi.mock('../services/resilience.js', () => ({
  resilientFetchJson: (...args: any[]) => mockResilientFetchJson(...args),
  getQueue: vi.fn(() => ({ add: (fn: any) => fn() })),
}));

import {
  fetchCryptoRankPrices,
  fetchCryptoRankBulkPrices,
  fetchCryptoRankGlobal,
} from '../services/marketScanner/cryptorankFetcher.js';

describe('CryptoRank Fetcher', () => {
  const ORIGINAL_ENV = process.env;

  beforeEach(() => {
    vi.clearAllMocks();
    process.env = { ...ORIGINAL_ENV, CRYPTORANK_API_KEY: 'test-api-key-123' };
  });

  afterEach(() => {
    process.env = ORIGINAL_ENV;
  });

  describe('fetchCryptoRankPrices', () => {
    it('returns empty array when no API key is set', async () => {
      delete process.env.CRYPTORANK_API_KEY;
      const result = await fetchCryptoRankPrices(['SOL', 'BTC']);
      expect(result).toEqual([]);
      expect(mockResilientFetchJson).not.toHaveBeenCalled();
    });

    it('returns empty array for unknown symbols', async () => {
      const result = await fetchCryptoRankPrices(['UNKNOWN_TOKEN']);
      expect(result).toEqual([]);
    });

    it('fetches individual currency data with percentChange', async () => {
      mockResilientFetchJson.mockResolvedValue({
        data: {
          id: 5663,
          key: 'solana',
          symbol: 'SOL',
          name: 'Solana',
          rank: 7,
          price: '142.50',
          high24h: '148.00',
          low24h: '138.25',
          volume24h: '2500000000',
          marketCap: '65000000000',
          fullyDilutedValuation: '72000000000',
          circulatingSupply: '456000000',
          percentChange: {
            h24: '-3.5',
            d7: '12.8',
            d30: '-15.2',
            m3: '45.0',
            m6: '120.5',
          },
          ath: { date: 1737244800000, value: '293.64', percentChange: '-51.5' },
          atl: { date: 1589155200000, value: '0.50', percentChange: '28400.0' },
        },
        status: { usedCredits: 1 },
      });

      const result = await fetchCryptoRankPrices(['SOL']);
      expect(result).toHaveLength(1);

      const sol = result[0];
      expect(sol.symbol).toBe('SOL');
      expect(sol.price).toBe(142.5);
      expect(sol.high24h).toBe(148.0);
      expect(sol.low24h).toBe(138.25);
      expect(sol.volume24h).toBe(2500000000);
      expect(sol.marketCap).toBe(65000000000);
      expect(sol.rank).toBe(7);
      expect(sol.source).toBe('cryptorank');
      expect(sol.percentChange).toEqual({
        h24: -3.5,
        d7: 12.8,
        d30: -15.2,
        m3: 45.0,
        m6: 120.5,
      });
      expect(sol.ath).toEqual({
        date: 1737244800000,
        value: 293.64,
        percentChange: -51.5,
      });
    });

    it('handles partial failures gracefully', async () => {
      mockResilientFetchJson
        .mockResolvedValueOnce({
          data: {
            id: 1, symbol: 'BTC', rank: 1,
            price: '65000', high24h: '66000', low24h: '64000',
            volume24h: '10000000000', marketCap: '1300000000000',
            fullyDilutedValuation: '1365000000000', circulatingSupply: '20000000',
          },
          status: { usedCredits: 1 },
        })
        .mockRejectedValueOnce(new Error('Network timeout'));

      const result = await fetchCryptoRankPrices(['BTC', 'SOL']);
      expect(result).toHaveLength(1);
      expect(result[0].symbol).toBe('BTC');
      expect(result[0].price).toBe(65000);
    });

    it('passes correct headers and URL', async () => {
      mockResilientFetchJson.mockResolvedValue({
        data: {
          id: 5663, symbol: 'SOL', rank: 7,
          price: '142', high24h: '148', low24h: '138',
          volume24h: '2500000000', marketCap: '65000000000',
          fullyDilutedValuation: '72000000000', circulatingSupply: '456000000',
        },
        status: { usedCredits: 1 },
      });

      await fetchCryptoRankPrices(['SOL']);

      expect(mockResilientFetchJson).toHaveBeenCalledWith(
        'https://api.cryptorank.io/v2/currencies/5663',
        { headers: { 'X-API-Key': 'test-api-key-123' } },
        expect.objectContaining({
          retries: 2,
          fetchTimeout: 10000,
          label: 'cryptorank/SOL',
        }),
      );
    });
  });

  describe('fetchCryptoRankBulkPrices', () => {
    it('returns empty array when no API key', async () => {
      delete process.env.CRYPTORANK_API_KEY;
      const result = await fetchCryptoRankBulkPrices(['SOL']);
      expect(result).toEqual([]);
    });

    it('filters currencies by requested symbols', async () => {
      mockResilientFetchJson.mockResolvedValue({
        data: [
          { symbol: 'BTC', rank: 1, price: '65000', volume24h: '10B', marketCap: '1.3T', fullyDilutedValuation: '1.3T', high24h: '66000', low24h: '64000' },
          { symbol: 'ETH', rank: 2, price: '3500', volume24h: '5B', marketCap: '420B', fullyDilutedValuation: '420B', high24h: '3600', low24h: '3400' },
          { symbol: 'USDT', rank: 3, price: '1.00', volume24h: '20B', marketCap: '100B', fullyDilutedValuation: '100B', high24h: '1.001', low24h: '0.999' },
          { symbol: 'SOL', rank: 7, price: '142', volume24h: '2.5B', marketCap: '65B', fullyDilutedValuation: '72B', high24h: '148', low24h: '138' },
        ],
        status: { usedCredits: 1 },
      });

      const result = await fetchCryptoRankBulkPrices(['SOL', 'BTC']);
      expect(result).toHaveLength(2);
      expect(result.map(r => r.symbol).sort()).toEqual(['BTC', 'SOL']);
    });

    it('handles fetch errors', async () => {
      mockResilientFetchJson.mockRejectedValue(new Error('API down'));
      const result = await fetchCryptoRankBulkPrices(['SOL']);
      expect(result).toEqual([]);
    });
  });

  describe('fetchCryptoRankGlobal', () => {
    it('returns null when no API key', async () => {
      delete process.env.CRYPTORANK_API_KEY;
      const result = await fetchCryptoRankGlobal();
      expect(result).toBeNull();
    });

    it('parses global market data correctly', async () => {
      mockResilientFetchJson.mockResolvedValue({
        data: {
          allCurrencies: 36889,
          activeCurrencies: 4645,
          totalMarketCap: '2349544732519',
          totalMarketCapChange: -2.19,
          totalVolume24h: '35322252743',
          totalVolume24hChange: 63.36,
          btcDominance: 55.95,
          btcDominanceChange: -1.13,
          ethDominance: 9.66,
          ethDominanceChange: -2.72,
          fearGreed: 5,
          fearGreedChange: -4,
          altcoinIndex: 30,
          altcoinIndexChange: -2,
        },
        status: { usedCredits: 1 },
      });

      const result = await fetchCryptoRankGlobal();
      expect(result).not.toBeNull();
      expect(result!.btcDominance).toBe(55.95);
      expect(result!.fearGreed).toBe(5);
      expect(result!.altcoinIndex).toBe(30);
      expect(result!.totalMarketCap).toBe(2349544732519);
      expect(result!.totalVolume24h).toBe(35322252743);
      expect(result!.activeCurrencies).toBe(4645);
    });

    it('returns null on fetch error', async () => {
      mockResilientFetchJson.mockRejectedValue(new Error('500'));
      const result = await fetchCryptoRankGlobal();
      expect(result).toBeNull();
    });
  });
});
