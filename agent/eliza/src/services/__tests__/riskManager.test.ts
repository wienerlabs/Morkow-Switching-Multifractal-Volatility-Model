/**
 * Risk Manager Service Tests
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { 
  RiskManager, 
  DEFAULT_RISK_LIMITS,
  resetRiskManager,
  getRiskManager,
} from '../riskManager.js';

describe('RiskManager', () => {
  let manager: RiskManager;
  
  beforeEach(() => {
    resetRiskManager();
    manager = new RiskManager();
  });
  
  describe('DEFAULT_RISK_LIMITS', () => {
    it('should have data-driven values from backtest', () => {
      // Position sizing (configured for $100 capital)
      expect(DEFAULT_RISK_LIMITS.maxPositionPct).toBe(20.0);  // 20% max = $20 per position
      expect(DEFAULT_RISK_LIMITS.minPositionPct).toBe(5.0);   // 5% min position size

      // Daily limits
      expect(DEFAULT_RISK_LIMITS.maxDailyLossPct).toBe(5.0);
      expect(DEFAULT_RISK_LIMITS.maxDailyTrades).toBe(2);

      // Volatility
      expect(DEFAULT_RISK_LIMITS.maxVolatility24h).toBe(0.15);
      expect(DEFAULT_RISK_LIMITS.minVolatility24h).toBe(0.02);

      // Cooldown (1 hour for production)
      expect(DEFAULT_RISK_LIMITS.minCooldownHours).toBe(1);

      // Data quality
      expect(DEFAULT_RISK_LIMITS.dataQualityScore).toBe(0.56);
    });
  });
  
  describe('checkTradeAllowed', () => {
    it('should allow valid trade', () => {
      const result = manager.checkTradeAllowed({
        proposedPositionPct: 3.0,
        currentVolatility24h: 0.05,
      });
      
      expect(result.allowed).toBe(true);
      expect(result.reason).toBe('All risk checks passed');
    });
    
    it('should block trade when daily loss limit reached', () => {
      // Simulate big loss
      manager.recordTrade(-5.5);
      
      const result = manager.checkTradeAllowed({
        proposedPositionPct: 3.0,
        currentVolatility24h: 0.05,
      });
      
      expect(result.allowed).toBe(false);
      expect(result.reason).toContain('Daily loss limit reached');
    });
    
    it('should block trade when max daily trades reached', () => {
      manager.recordTrade(1.0);
      manager.recordTrade(0.5);
      
      // Reset lastTradeTime to bypass cooldown
      const state = manager.getState();
      state.lastTradeTime = null;
      
      const result = manager.checkTradeAllowed({
        proposedPositionPct: 3.0,
        currentVolatility24h: 0.05,
      });
      
      expect(result.allowed).toBe(false);
      expect(result.reason).toContain('Max daily trades reached');
    });
    
    it('should block trade during cooldown', () => {
      // Create manager with cooldown enabled
      const managerWithCooldown = new RiskManager({ minCooldownHours: 24 });
      managerWithCooldown.recordTrade(1.0);

      const result = managerWithCooldown.checkTradeAllowed({
        proposedPositionPct: 3.0,
        currentVolatility24h: 0.05,
      });

      expect(result.allowed).toBe(false);
      expect(result.reason).toContain('Cooldown active');
    });
    
    it('should block trade when volatility too high', () => {
      const result = manager.checkTradeAllowed({
        proposedPositionPct: 3.0,
        currentVolatility24h: 0.20, // 20% > 15% max
      });
      
      expect(result.allowed).toBe(false);
      expect(result.reason).toContain('Volatility too high');
    });
    
    it('should warn but allow trade with low volatility', () => {
      const result = manager.checkTradeAllowed({
        proposedPositionPct: 3.0,
        currentVolatility24h: 0.01, // 1% < 2% min
      });
      
      expect(result.allowed).toBe(true);
      expect(result.warnings.some(w => w.includes('Low volatility'))).toBe(true);
    });
    
    it('should cap position size to max', () => {
      const result = manager.checkTradeAllowed({
        proposedPositionPct: 25.0, // Over 20% max
        currentVolatility24h: 0.05,
      });

      expect(result.allowed).toBe(true);
      expect(result.suggestedPositionPct).toBe(20.0);  // Capped to max
      expect(result.warnings.some(w => w.includes('Position capped'))).toBe(true);
    });

    it('should raise position size to min', () => {
      const result = manager.checkTradeAllowed({
        proposedPositionPct: 1.0, // Below 5% min
        currentVolatility24h: 0.05,
      });

      expect(result.allowed).toBe(true);
      expect(result.suggestedPositionPct).toBe(5.0);  // Raised to min
      expect(result.warnings.some(w => w.includes('Position raised'))).toBe(true);
    });
    
    it('should warn about low data quality', () => {
      const result = manager.checkTradeAllowed({
        proposedPositionPct: 3.0,
        currentVolatility24h: 0.05,
      });
      
      // Default data quality is 56% < 70%
      expect(result.warnings.some(w => w.includes('Low data quality'))).toBe(true);
    });
  });
  
  describe('calculatePositionSize', () => {
    it('should calculate based on confidence and volatility', () => {
      const result = manager.calculatePositionSize({
        modelConfidence: 0.9,
        currentVolatility24h: 0.05,
        portfolioValueUsd: 10000,
      });

      expect(result.positionPct).toBeGreaterThan(0);
      expect(result.positionPct).toBeLessThanOrEqual(20.0);  // Max is 20%
      expect(result.positionUsd).toBeLessThanOrEqual(2000);  // 20% of $10K
      expect(result.rationale).toContain('Half-Kelly');
    });
    
    it('should reduce position for low confidence', () => {
      const highConfidence = manager.calculatePositionSize({
        modelConfidence: 0.95,
        currentVolatility24h: 0.05,
        portfolioValueUsd: 10000,
      });
      
      const lowConfidence = manager.calculatePositionSize({
        modelConfidence: 0.55,
        currentVolatility24h: 0.05,
        portfolioValueUsd: 10000,
      });
      
      expect(lowConfidence.positionPct).toBeLessThan(highConfidence.positionPct);
    });
  });
  
  describe('recordTrade', () => {
    it('should update state correctly', () => {
      manager.recordTrade(2.5);
      
      const state = manager.getState();
      expect(state.dailyPnL).toBe(2.5);
      expect(state.dailyTradeCount).toBe(1);
      expect(state.lastTradeTime).not.toBeNull();
    });
  });
  
  describe('resetDaily', () => {
    it('should reset daily counters', () => {
      manager.recordTrade(2.5);
      manager.resetDaily();
      
      const state = manager.getState();
      expect(state.dailyPnL).toBe(0);
      expect(state.dailyTradeCount).toBe(0);
    });
  });
  
  describe('getRiskManager singleton', () => {
    it('should return same instance', () => {
      resetRiskManager();
      const manager1 = getRiskManager();
      const manager2 = getRiskManager();
      expect(manager1).toBe(manager2);
    });
  });
});

