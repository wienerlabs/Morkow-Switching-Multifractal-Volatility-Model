/**
 * Global Risk Management Module
 * 
 * Exports all risk management components for cross-strategy risk control.
 * 
 * Features:
 * 1. Drawdown Circuit Breakers (Daily 5%, Weekly 10%, Monthly 15%)
 * 2. Correlation Risk Tracking (Asset/Protocol exposure limits)
 * 3. Dynamic Stop Loss (ATR-based or asset class)
 * 4. Protocol Concentration Limit (Max 50% per protocol)
 * 5. Oracle Staleness Protection (30s reject, 60s emergency)
 * 6. Emergency Gas Budget (Real-time gas monitoring)
 */

// Types
export type {
  CircuitBreakerState,
  DrawdownLimits,
  DrawdownStatus,
  ExposureLimits,
  AssetExposure,
  ProtocolExposure,
  CorrelationRiskStatus,
  AssetClass,
  DynamicStopLossConfig,
  AssetVolatilityData,
  OracleConfig,
  OracleStatus,
  GasBudgetConfig,
  GasBudgetStatus,
  GlobalRiskConfig,
  GlobalRiskStatus,
  RiskAlert,
  TrackedPosition,
  StrategyRiskTier,
  StrategyType,
  StrategyVaRThresholds,
} from './types.js';

export { STRATEGY_RISK_TIERS } from './types.js';

// Global Risk Manager
export {
  GlobalRiskManager,
  getGlobalRiskManager,
  resetGlobalRiskManager,
  DEFAULT_GLOBAL_RISK_CONFIG,
  DEFAULT_DRAWDOWN_LIMITS,
  DEFAULT_EXPOSURE_LIMITS,
  DEFAULT_STOP_LOSS_CONFIG,
  DEFAULT_STRATEGY_VAR_THRESHOLDS,
  getPositionScaleForRegime,
  getStrategyRiskTier,
} from './globalRiskManager.js';

// Oracle Service
export {
  OracleService,
  DEFAULT_ORACLE_CONFIG,
  PYTH_PRICE_FEEDS,
} from './oracleService.js';

// Gas Service
export {
  GasService,
  DEFAULT_GAS_CONFIG,
  COMPUTE_UNITS,
} from './gasService.js';

// P&L Attribution
export {
  PnLAttributionEngine,
  getPnLAttributionEngine,
  resetPnLAttributionEngine,
  type StrategyType as PnLStrategyType,
  type ComponentType as PnLComponentType,
  type AttributedTrade,
  type StrategyPnL,
  type ComponentPnL,
  type TokenPnL,
  type TimePeriodPnL,
  type PnLAttributionReport,
  type ComponentInfluence,
} from './pnlAttribution.js';

// Regime mapping (A-LAMS → TS)
export {
  mapAlamsRegimeToMarketRegime,
  ALAMS_REGIME_POSITION_SCALE,
} from '../analysis/regimeDetector.js';

// Debate Client
export {
  DebateClient,
  getDebateClient,
  resetDebateClient,
  type DebateTranscript,
  type DebateTranscriptResponse,
  type DebateStatsResponse,
  type DebateStorageStats,
} from './debateClient.js';

