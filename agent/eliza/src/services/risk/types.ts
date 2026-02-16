/**
 * Global Risk Manager Types
 * 
 * Type definitions for cross-strategy risk management.
 * All values are calculated from real on-chain data.
 */

// ============= DRAWDOWN CIRCUIT BREAKERS =============

export type CircuitBreakerState = 'ACTIVE' | 'PAUSED' | 'STOPPED' | 'LOCKDOWN';

export interface DrawdownLimits {
  daily: number;     // 5% - pause all strategies
  weekly: number;    // 10% - full stop
  monthly: number;   // 15% - system lockdown
}

export interface DrawdownStatus {
  dailyDrawdownPct: number;
  weeklyDrawdownPct: number;
  monthlyDrawdownPct: number;
  peakValueUsd: number;
  currentValueUsd: number;
  circuitBreakerState: CircuitBreakerState;
  lastTriggered?: Date;
  triggerReason?: string;
}

// ============= CORRELATION RISK =============

export interface ExposureLimits {
  maxBaseAssetPct: number;     // 40% max single base asset
  maxQuoteAssetPct: number;    // 60% max single quote asset  
  maxProtocolPct: number;      // 50% max single protocol
}

export interface AssetExposure {
  asset: string;
  exposureUsd: number;
  exposurePct: number;
  positions: string[];        // Position IDs contributing
}

export interface ProtocolExposure {
  protocol: string;
  exposureUsd: number;
  exposurePct: number;
  positions: string[];
}

export interface CorrelationRiskStatus {
  baseAssetExposures: AssetExposure[];
  quoteAssetExposures: AssetExposure[];
  protocolExposures: ProtocolExposure[];
  violations: string[];
  isCompliant: boolean;
}

// ============= DYNAMIC STOP LOSS =============

export type AssetClass = 'major' | 'midcap' | 'alt';

export interface DynamicStopLossConfig {
  majorStopPct: number;    // 3% for BTC, ETH, SOL
  midcapStopPct: number;   // 5% for mid-caps
  altStopPct: number;      // 7% for alts
  useATR: boolean;         // Whether to use ATR-based stops
  atrMultiplier: number;   // ATR multiplier for stop calculation
}

export interface AssetVolatilityData {
  asset: string;
  atr24h?: number;         // Average True Range 24h
  marketCapUsd?: number;
  volume24hUsd?: number;
  classification: AssetClass;
  recommendedStopPct: number;
}

// ============= ORACLE STALENESS =============

export interface OracleConfig {
  maxStalenessSeconds: number;        // 30s reject threshold
  emergencyExitSeconds: number;       // 60s emergency exit
  requiredConfirmations: number;      // Min oracle confirmations
}

export interface OracleStatus {
  source: string;
  price: number;
  timestamp: Date;
  slotNumber?: number;
  confidence?: number;
  stalenessSeconds: number;
  isStale: boolean;
  isEmergency: boolean;
}

// ============= GAS BUDGET =============

export interface GasBudgetConfig {
  reserveMultiplier: number;      // Multiplier for reserve (e.g., 2x average)
  maxPriorityFeeLamports: number; // Cap on priority fee
  emergencyGasReserveUsd: number; // Min reserve for emergency exits
}

export interface GasBudgetStatus {
  currentBaseFee: number;
  currentPriorityFee: number;
  averageGasCostUsd: number;
  reserveBalanceUsd: number;
  recommendedPriorityFee: number;
  canAffordEmergencyExit: boolean;
}

// ============= STRATEGY RISK TIERS =============

export type StrategyRiskTier = 'low' | 'medium' | 'high';

export type StrategyType = 'spot' | 'lp' | 'arb' | 'perps' | 'leverage' | 'margin';

export const STRATEGY_RISK_TIERS: Record<StrategyType, StrategyRiskTier> = {
  lp: 'low',
  spot: 'low',
  arb: 'medium',
  perps: 'high',
  leverage: 'high',
  margin: 'high',
};

// ============= A-LAMS VaR (Python API) =============

export interface ALAMSVaRConfig {
  apiUrl: string;               // Python API base URL
  timeoutMs: number;            // HTTP request timeout
  maxAcceptableVarPct: number;  // Max VaR before blocking (e.g. 0.05 = 5%)
  cacheTtlMs: number;           // Cache TTL for VaR results
}

export interface ALAMSVaRResult {
  var_pure: number;
  slippage_component: number;
  var_total: number;
  confidence: number;
  current_regime: number;
  regime_probs: number[];
  delta: number;
  regime_means: number[];
  regime_sigmas: number[];
}

export interface ALAMSVaRStatus {
  available: boolean;
  result: ALAMSVaRResult | null;
  error?: string;
  fetchedAt?: number;
  regimePositionScale?: number;
}

// ============= GLOBAL RISK STATUS =============

export interface StrategyVaRThresholds {
  spot: number;
  lp: number;
  arb: number;
  perps: number;
  leverage: number;
  margin: number;
}

export interface GlobalRiskConfig {
  drawdownLimits: DrawdownLimits;
  exposureLimits: ExposureLimits;
  stopLossConfig: DynamicStopLossConfig;
  oracleConfig: OracleConfig;
  gasBudgetConfig: GasBudgetConfig;
  alamsVarConfig?: ALAMSVaRConfig;
  strategyVarThresholds?: StrategyVaRThresholds;
}

export interface GlobalRiskStatus {
  timestamp: Date;
  circuitBreakerState: CircuitBreakerState;
  drawdown: DrawdownStatus;
  correlationRisk: CorrelationRiskStatus;
  oracleStatuses: Map<string, OracleStatus>;
  gasBudget: GasBudgetStatus;
  alamsVar?: ALAMSVaRStatus;
  canTrade: boolean;
  blockReasons: string[];
}

export interface RiskAlert {
  id: string;
  timestamp: Date;
  severity: 'warning' | 'critical' | 'emergency';
  category: 'drawdown' | 'exposure' | 'oracle' | 'gas' | 'position';
  message: string;
  data?: Record<string, unknown>;
}

// ============= ADVERSARIAL DEBATE =============

export interface DebateEvidence {
  source: string;
  claim: string;
  value: number;
  threshold: number | null;
  severity: 'low' | 'medium' | 'high' | 'critical';
  supports_approval: boolean;
}

export interface DebateAgentArgument {
  role: string;
  position: string;
  confidence: number;
  arguments: string[];
  evidence: DebateEvidence[];
  suggested_action: string;
  suggested_size_pct: number | null;
  bayesian_posterior: number;
}

export interface DebateRound {
  round: number;
  trader: DebateAgentArgument;
  risk_manager: DebateAgentArgument;
  devils_advocate: DebateAgentArgument;
  arbitrator: {
    role: string;
    decision: 'approve' | 'reject';
    confidence: number;
    reasoning: string[];
    approval_score: number;
    rejection_score: number;
    recommended_size_pct: number;
    trader_weight: number;
    risk_weight: number;
    da_weight: number;
    strategy_profile: string;
  };
}

export interface DebateResult {
  final_decision: 'approve' | 'reject';
  final_confidence: number;
  recommended_size_pct: number;
  approval_score: number;
  rounds: DebateRound[];
  num_rounds: number;
  elapsed_ms: number;
  original_approved: boolean;
  decision_changed: boolean;
  strategy: string;
  evidence_summary: {
    total: number;
    bullish: number;
    bearish: number;
    bullish_items: DebateEvidence[];
    bearish_items: DebateEvidence[];
  };
}

// ============= OUTCOME CIRCUIT BREAKERS =============

export interface OutcomeCircuitBreakerStatus {
  name: string;
  strategy: string;
  state: 'closed' | 'open' | 'half_open';
  consecutive_losses: number;
  loss_limit: number;
  loss_type: string;
  cooldown_seconds: number;
  opened_at: number | null;
  cooldown_remaining: number | null;
  total_trades: number;
  total_wins: number;
  total_losses: number;
  win_rate: number;
  description: string;
}

// Position types for tracking
export interface TrackedPosition {
  id: string;
  type: 'lp' | 'perps' | 'arbitrage';
  protocol: string;
  baseAsset: string;
  quoteAsset: string;
  sizeUsd: number;
  entryPrice: number;
  currentPrice?: number;
  unrealizedPnlUsd: number;
  entryTime: Date;
  stopLossPct?: number;
}

