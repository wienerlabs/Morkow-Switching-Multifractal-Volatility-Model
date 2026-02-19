from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CalibrationMethod(str, Enum):
    MLE = "mle"
    GRID = "grid"
    EMPIRICAL = "empirical"
    HYBRID = "hybrid"


class DataSource(str, Enum):
    SOLANA = "solana"
    YFINANCE = "yfinance"


class CalibrateRequest(BaseModel):
    token: str = Field(..., description="Token symbol (SOL, RAY) or mint address, or yfinance ticker")
    data_source: DataSource = DataSource.SOLANA
    start_date: str = Field(..., description="ISO date string, e.g. '2025-01-01'")
    end_date: str = Field(..., description="ISO date string, e.g. '2026-02-10'")
    num_states: int = Field(5, ge=2, le=10)
    method: CalibrationMethod = CalibrationMethod.MLE
    target_var_breach: float = Field(0.05, gt=0.0, lt=1.0)
    interval: str = Field("1D", description="Candle interval for Solana data")
    use_student_t: bool = Field(False, description="Use Student-t distribution for VaR")
    nu: float = Field(5.0, gt=2.0, description="Student-t degrees of freedom (must be > 2)")
    leverage_gamma: float | str | None = Field(
        None,
        description=(
            "Asymmetric leverage parameter (γ). "
            "None = no leverage, float (≤ 0) = fixed value, "
            "'estimate' = estimate via MLE."
        ),
    )


class CalibrationMetrics(BaseModel):
    var_breach_rate: float
    vol_correlation: float
    log_likelihood: float
    aic: float
    bic: float


class CalibrateResponse(BaseModel):
    token: str
    method: str
    num_states: int
    sigma_low: float
    sigma_high: float
    p_stay: float | list[float]
    sigma_states: list[float]
    leverage_gamma: float = Field(0.0, description="Asymmetric leverage parameter used in calibration")
    metrics: CalibrationMetrics
    calibrated_at: datetime


class RegimeResponse(BaseModel):
    timestamp: datetime
    regime_state: int = Field(..., description="Most probable state index (1-based)")
    regime_name: str = Field(..., description="Human-readable regime label")
    regime_probabilities: list[float]
    volatility_filtered: float
    volatility_forecast: float
    var_95: float
    transition_matrix: list[list[float]]


class VaRResponse(BaseModel):
    timestamp: datetime
    confidence: float
    var_value: float
    sigma_forecast: float
    z_alpha: float
    regime_probabilities: list[float]
    distribution: str = "normal"


class VolatilityForecastResponse(BaseModel):
    timestamp: datetime
    sigma_forecast: float
    sigma_filtered: float
    regime_probabilities: list[float]
    sigma_states: list[float]


class BacktestSummaryResponse(BaseModel):
    token: str
    num_observations: int
    var_alpha: float
    breach_count: int
    breach_rate: float
    kupiec_lr: Optional[float]
    kupiec_pvalue: Optional[float]
    kupiec_pass: bool
    christoffersen_lr: Optional[float]
    christoffersen_pvalue: Optional[float]
    christoffersen_pass: bool


class TailProbResponse(BaseModel):
    l1_threshold: float
    p1_day: float
    horizon_probs: dict[int, float]
    distribution: str


class RegimeStreamMessage(BaseModel):
    timestamp: datetime
    regime_state: int
    regime_name: str
    regime_probabilities: list[float]
    volatility_forecast: float
    var_95: float


class ErrorResponse(BaseModel):
    detail: str
    error_code: str = "INTERNAL_ERROR"


REGIME_NAMES: dict[int, str] = {
    1: "Very Low Vol",
    2: "Low Vol",
    3: "Normal",
    4: "High Vol",
    5: "Crisis",
}


def get_regime_name(state_idx: int, num_states: int) -> str:
    """Map 1-based state index to human-readable name."""
    if num_states in (4, 5, 6):
        return REGIME_NAMES.get(state_idx, f"State {state_idx}")
    return f"State {state_idx}/{num_states}"



# ── News Intelligence Models ──

class NewsSentimentModel(BaseModel):
    score: float = Field(..., description="Continuous sentiment [-1, 1]")
    confidence: float = Field(..., description="Confidence [0, 1]")
    label: str = Field(..., description="Bullish / Bearish / Neutral")
    bull_weight: float
    bear_weight: float
    entropy: float = Field(..., description="Information entropy of sentiment distribution")


class NewsItemModel(BaseModel):
    id: str
    source: str
    api_source: str
    title: str
    body: str
    url: str
    timestamp: float
    assets: list[str]
    sentiment: NewsSentimentModel
    impact: float = Field(..., description="Impact score [0, 10]")
    novelty: float = Field(..., description="Novelty [0, 1]")
    source_credibility: float
    time_decay: float
    regime_multiplier: float


class NewsMarketSignalModel(BaseModel):
    sentiment_ewma: float = Field(..., description="EWMA sentiment [-1, 1]")
    sentiment_momentum: float = Field(..., description="Sentiment momentum ΔS")
    entropy: float = Field(..., description="Consensus entropy")
    confidence: float = Field(..., description="Aggregate confidence [0, 1]")
    direction: str = Field(..., description="LONG / SHORT / NEUTRAL")
    strength: float = Field(..., description="Signal strength [0, 1]")
    n_sources: int
    n_items: int
    bull_pct: float
    bear_pct: float
    neutral_pct: float


class NewsSourceCounts(BaseModel):
    cryptocompare: int = 0
    newsdata: int = 0
    cryptopanic: int = 0


class NewsMeta(BaseModel):
    errors: list[str] = []
    elapsed_ms: int = 0
    total: int = 0
    regime_state: Optional[int] = None


class NewsFeedResponse(BaseModel):
    items: list[NewsItemModel]
    signal: NewsMarketSignalModel
    source_counts: NewsSourceCounts
    meta: NewsMeta


# ── Regime Analytics Models ──


class RegimeDurationsResponse(BaseModel):
    token: str
    p_stay: float | list[float]
    num_states: int
    durations: dict[int, float] = Field(..., description="Expected duration per regime (days)")
    timestamp: datetime


class RegimePeriod(BaseModel):
    start: datetime
    end: datetime
    regime: int
    duration: int
    cumulative_return: float
    volatility: float
    max_drawdown: float


class RegimeHistoryResponse(BaseModel):
    token: str
    num_periods: int
    periods: list[RegimePeriod]
    timestamp: datetime


class TransitionAlertResponse(BaseModel):
    token: str
    alert: bool
    current_regime: int
    transition_probability: float
    most_likely_next_regime: int
    next_regime_probability: float
    threshold: float
    timestamp: datetime


class RegimeStatRow(BaseModel):
    regime: int
    mean_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    days_in_regime: int
    frequency: float


class RegimeStatisticsResponse(BaseModel):
    token: str
    num_states: int
    total_observations: int
    statistics: list[RegimeStatRow]
    timestamp: datetime


# ── Model Comparison Models ──


class CompareRequest(BaseModel):
    token: str = Field(..., description="Token key from _model_store (must be calibrated)")
    alpha: float = Field(0.05, gt=0.0, lt=1.0)
    models: Optional[list[str]] = Field(
        None,
        description="Subset of: msm, garch, egarch, gjr, rolling_20, rolling_60, ewma. None = all.",
    )


class ModelMetricsRow(BaseModel):
    model: str
    log_likelihood: Optional[float]
    aic: Optional[float]
    bic: Optional[float]
    breach_rate: Optional[float]
    breach_count: int
    kupiec_lr: Optional[float]
    kupiec_pvalue: Optional[float]
    kupiec_pass: Optional[bool]
    christoffersen_lr: Optional[float]
    christoffersen_pvalue: Optional[float]
    christoffersen_pass: Optional[bool]
    mae_volatility: float
    correlation: Optional[float]
    num_params: int


class CompareResponse(BaseModel):
    token: str
    alpha: float
    num_observations: int
    models_compared: list[str]
    results: list[ModelMetricsRow]
    timestamp: datetime


class ComparisonReportResponse(BaseModel):
    token: str
    alpha: float
    summary_table: str = Field(..., description="Markdown table")
    winners: dict[str, str]
    pass_fail: dict[str, dict[str, Optional[bool]]]
    ranking: list[str]
    timestamp: datetime


# --- Portfolio VaR models ---

class PortfolioCalibrateRequest(BaseModel):
    tokens: list[str] = Field(..., min_length=2, description="Ticker symbols")
    weights: dict[str, float] = Field(..., description="Asset weights summing to ~1.0")
    num_states: int = Field(5, ge=2, le=10)
    method: str = Field("mle", pattern="^(mle|grid|empirical|hybrid)$")
    period: str = "2y"
    data_source: DataSource = DataSource.YFINANCE
    copula_family: str | None = Field(None, description="Copula family: gaussian|student_t|clayton|gumbel|frank|auto")


class RegimeBreakdownItem(BaseModel):
    regime: int
    probability: float
    portfolio_sigma: float
    portfolio_var: float


class PortfolioVaRResponse(BaseModel):
    portfolio_var: float
    portfolio_sigma: float
    z_alpha: float
    weights: dict[str, float]
    regime_breakdown: list[RegimeBreakdownItem]
    timestamp: datetime


class AssetDecompositionItem(BaseModel):
    asset: str
    weight: float
    marginal_var: float
    component_var: float
    pct_contribution: float


class MarginalVaRResponse(BaseModel):
    portfolio_var: float
    portfolio_sigma: float
    decomposition: list[AssetDecompositionItem]
    timestamp: datetime


class AssetStressItem(BaseModel):
    asset: str
    normal_sigma: float
    stressed_sigma: float


class StressVaRResponse(BaseModel):
    forced_regime: int
    stressed_var: float
    stressed_sigma: float
    normal_var: float
    normal_sigma: float
    stress_multiplier: float
    regime_correlation: list[list[float]]
    asset_stress: list[AssetStressItem]
    timestamp: datetime


# --- Copula Portfolio VaR models ---


class CopulaFamily(str, Enum):
    GAUSSIAN = "gaussian"
    STUDENT_T = "student_t"
    CLAYTON = "clayton"
    GUMBEL = "gumbel"
    FRANK = "frank"
    AUTO = "auto"


class TailDependence(BaseModel):
    lambda_lower: float
    lambda_upper: float


class CopulaFitResult(BaseModel):
    family: str
    params: dict
    log_likelihood: float
    aic: float
    bic: float
    n_obs: int
    n_assets: int
    n_params: int
    tail_dependence: TailDependence


class CopulaPortfolioVaRResponse(BaseModel):
    copula_var: float
    gaussian_var: float
    var_ratio: float
    copula_family: str
    tail_dependence: TailDependence
    n_simulations: int
    alpha: float
    timestamp: datetime


class RegimeCopulaItem(BaseModel):
    regime: int
    n_obs: int
    copula: CopulaFitResult


class CopulaDiagnosticsResponse(BaseModel):
    portfolio_key: str
    copula_family: str
    fit: CopulaFitResult
    regime_copulas: list[RegimeCopulaItem]
    timestamp: datetime


class CopulaCompareItem(BaseModel):
    family: str
    log_likelihood: float
    aic: float
    bic: float
    tail_dependence: TailDependence
    rank: int
    best: bool


class CopulaCompareResponse(BaseModel):
    portfolio_key: str
    results: list[CopulaCompareItem]
    timestamp: datetime


class RegimeTailDependenceItem(BaseModel):
    regime: int
    family: str
    lambda_lower: float
    lambda_upper: float


class RegimeDependentCopulaVaRResponse(BaseModel):
    regime_dependent_var: float
    static_var: float
    var_difference_pct: float
    current_regime_copula: CopulaFitResult
    regime_tail_dependence: list[RegimeTailDependenceItem]
    dominant_regime: int
    regime_probs: list[float]
    n_simulations: int
    alpha: float
    timestamp: datetime


# --- EVT (Extreme Value Theory) models ---


class ThresholdMethod(str, Enum):
    PERCENTILE = "percentile"
    MEAN_EXCESS = "mean_excess"
    VARIANCE_STABILITY = "variance_stability"


class EVTCalibrateRequest(BaseModel):
    token: str = Field(..., description="Token key from _model_store (must be calibrated)")
    threshold_method: ThresholdMethod = ThresholdMethod.VARIANCE_STABILITY
    min_exceedances: int = Field(50, ge=10, le=500)


class EVTCalibrateResponse(BaseModel):
    token: str
    xi: float = Field(..., description="GPD shape parameter (ξ). >0 = heavy tail")
    beta: float = Field(..., description="GPD scale parameter (β)")
    threshold: float
    n_total: int
    n_exceedances: int
    log_likelihood: float
    aic: float
    bic: float
    threshold_method: str
    timestamp: datetime


class EVTVaRResponse(BaseModel):
    timestamp: datetime
    confidence: float
    var_value: float = Field(..., description="VaR in return space (negative)")
    cvar_value: float = Field(..., description="CVaR / Expected Shortfall (negative)")
    distribution: str = "gpd"
    xi: float
    beta: float
    threshold: float


class EVTBacktestRow(BaseModel):
    alpha: float
    confidence: float
    evt_var: float
    breach_count: int
    breach_rate: float
    expected_rate: float
    kupiec_lr: Optional[float]
    kupiec_pvalue: Optional[float]
    kupiec_pass: Optional[bool]


class EVTBacktestResponse(BaseModel):
    token: str
    results: list[EVTBacktestRow]
    timestamp: datetime


class VaRComparisonRow(BaseModel):
    method: str
    alpha: float
    confidence: float
    var_value: float
    breach_count: int
    breach_rate: float
    expected_rate: float


class EVTDiagnosticsResponse(BaseModel):
    token: str
    xi: float
    beta: float
    threshold: float
    threshold_method: str
    n_exceedances: int
    backtest: list[EVTBacktestRow]
    comparison: list[VaRComparisonRow]
    timestamp: datetime


# ── Hawkes Process Models ──────────────────────────────────────────────


class HawkesCalibrateRequest(BaseModel):
    token: str = Field(..., description="Token key from _model_store (must be calibrated)")
    threshold_percentile: float = Field(5.0, gt=0.0, lt=50.0)
    use_absolute: bool = Field(True, description="Use |returns| for both tails")


class HawkesCalibrateResponse(BaseModel):
    token: str
    mu: float = Field(..., description="Baseline intensity (background event rate)")
    alpha: float = Field(..., description="Excitation magnitude per event")
    beta: float = Field(..., description="Decay rate of excitation")
    branching_ratio: float = Field(..., description="α/β — must be < 1 for stationarity")
    half_life: float = Field(..., description="Time for excitation to halve (ln2/β)")
    stationary: bool
    n_events: int
    log_likelihood: float
    aic: float
    bic: float
    threshold: float
    timestamp: datetime


class HawkesIntensityResponse(BaseModel):
    token: str
    current_intensity: float
    baseline: float
    intensity_ratio: float
    peak_intensity: float
    mean_intensity: float
    contagion_risk_score: float = Field(..., ge=0.0, le=1.0, description="Flash crash clustering risk [0-1]")
    excitation_level: float = Field(..., description="Current intensity minus baseline")
    risk_level: str = Field(..., description="low/medium/high/critical")
    timestamp: datetime


class HawkesClusterItem(BaseModel):
    cluster_id: int
    start_time: float
    end_time: float
    n_events: int
    duration: float
    peak_intensity: float


class HawkesClustersResponse(BaseModel):
    token: str
    clusters: list[HawkesClusterItem]
    n_clusters: int
    timestamp: datetime


class HawkesVaRRequest(BaseModel):
    token: str = Field(..., description="Token key from _model_store")
    confidence: float = Field(95.0, gt=50.0, le=99.99)
    max_multiplier: float = Field(3.0, gt=1.0, le=10.0)


class HawkesVaRResponse(BaseModel):
    adjusted_var: float
    base_var: float
    multiplier: float
    intensity_ratio: float
    capped: bool
    confidence: float
    recent_events: int = Field(..., description="Number of extreme events in lookback window")
    timestamp: datetime


class HawkesSimulateRequest(BaseModel):
    token: str | None = Field(None, description="Token key — use stored params if provided")
    mu: float | None = Field(None, gt=0, description="Baseline intensity (required if no token)")
    alpha: float | None = Field(None, gt=0, description="Excitation magnitude (required if no token)")
    beta: float | None = Field(None, gt=0, description="Decay rate (required if no token)")
    T: float = Field(500.0, gt=0, description="Simulation horizon")
    seed: int = Field(42, description="Random seed")


class HawkesSimulateResponse(BaseModel):
    n_events: int
    T: float
    mean_intensity: float
    peak_intensity: float
    timestamp: datetime


# ── Multifractal / Hurst ──────────────────────────────────────────────


class HurstResponse(BaseModel):
    token: str
    H: float = Field(..., description="Hurst exponent estimate")
    H_se: float = Field(..., description="Standard error of H")
    r_squared: float = Field(..., description="R² of log-log regression")
    interpretation: str
    method: str = Field(..., description="rs or dfa")
    timestamp: datetime


class MultifractalSpectrumResponse(BaseModel):
    token: str
    width: float = Field(..., description="Spectrum width (degree of multifractality)")
    peak_alpha: float = Field(..., description="α at peak f(α)")
    is_multifractal: bool
    q_values: list[float]
    tau_q: list[float]
    H_q: list[float] = Field(..., description="Generalized Hurst exponent H(q)")
    alpha: list[float]
    f_alpha: list[float]
    timestamp: datetime


class RegimeHurstItem(BaseModel):
    regime: int
    sigma: float
    n_obs: int
    fraction: float
    H: float | None = None
    H_se: float | None = None
    interpretation: str


class RegimeHurstResponse(BaseModel):
    token: str
    per_regime: list[RegimeHurstItem]
    n_states: int
    summary: str
    timestamp: datetime


class FractalDiagnosticsResponse(BaseModel):
    token: str
    H_rs: float
    H_dfa: float
    spectrum_width: float
    is_multifractal: bool
    is_long_range_dependent: bool
    confidence_z: float
    timestamp: datetime


# ── Rough Volatility ─────────────────────────────────────────────────


class RoughModel(str, Enum):
    BERGOMI = "rough_bergomi"
    HESTON = "rough_heston"


class RoughCalibrateRequest(BaseModel):
    token: str = Field(..., description="Token identifier (must be calibrated first via /calibrate)")
    model: RoughModel = Field(RoughModel.BERGOMI, description="Rough volatility model variant")
    window: int = Field(5, ge=2, le=60, description="Rolling window for realized vol")
    max_lag: int = Field(50, ge=10, le=200, description="Max lag for variogram")


class RoughCalibrationMetrics(BaseModel):
    H_se: float
    H_r_squared: float
    vol_correlation: float
    mae: float
    is_rough: bool
    optimization_success: bool | None = None
    optimization_nit: int | None = None


class RoughCalibrateResponse(BaseModel):
    token: str
    model: str
    H: float = Field(..., description="Hurst exponent (H < 0.3 = rough)")
    nu: float | None = Field(None, description="Vol-of-vol (rBergomi)")
    lambda_: float | None = Field(None, description="Mean-reversion speed (rHeston)")
    theta: float | None = Field(None, description="Long-run variance (rHeston)")
    xi: float | None = Field(None, description="Vol-of-vol (rHeston)")
    V0: float = Field(..., description="Initial variance")
    metrics: RoughCalibrationMetrics
    method: str
    timestamp: datetime


class RoughForecastResponse(BaseModel):
    token: str
    model: str
    horizon: int
    current_vol: float
    point_forecast: list[float]
    lower_95: list[float]
    upper_95: list[float]
    mean_forecast: list[float]
    timestamp: datetime


class RoughDiagnosticsResponse(BaseModel):
    token: str
    H_variogram: float = Field(..., description="H from variogram method")
    H_se: float
    r_squared: float
    is_rough: bool
    lags: list[float]
    variogram: list[float]
    interpretation: str
    timestamp: datetime


class RoughModelMetrics(BaseModel):
    mae: float
    rmse: float
    correlation: float


class RoughCompareMSMResponse(BaseModel):
    token: str
    rough_H: float
    rough_nu: float
    rough_is_rough: bool
    rough_metrics: RoughModelMetrics
    msm_num_states: int
    msm_metrics: RoughModelMetrics
    winner: str
    mae_ratio: float
    rmse_ratio: float
    corr_diff: float
    timestamp: datetime


# ── SVJ (Stochastic Volatility with Jumps) ──────────────────────────


class SVJCalibrateRequest(BaseModel):
    token: str
    use_hawkes: bool = Field(default=False, description="Use Hawkes process for jump clustering")
    jump_threshold_multiplier: float = Field(default=3.0, ge=1.5, le=6.0)


class SVJHawkesParams(BaseModel):
    mu: float
    alpha: float
    beta: float
    branching_ratio: float
    current_intensity: float
    baseline_intensity: float
    intensity_ratio: float


class SVJCalibrateResponse(BaseModel):
    token: str
    kappa: float
    theta: float
    sigma: float
    rho: float
    lambda_: float = Field(alias="lambda_")
    mu_j: float
    sigma_j: float
    feller_ratio: float
    feller_satisfied: bool
    log_likelihood: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    n_obs: int
    n_jumps_detected: int
    jump_fraction: float
    bns_statistic: float
    bns_pvalue: float
    optimization_success: bool
    use_hawkes: bool
    hawkes_params: Optional[SVJHawkesParams] = None
    timestamp: datetime

    class Config:
        populate_by_name = True


class SVJVaRResponse(BaseModel):
    token: str
    var_svj: float
    var_diffusion_only: float
    var_jump_component: float
    expected_shortfall: float
    jump_contribution_pct: float
    alpha: float
    confidence: float
    n_simulations: int
    current_variance: float
    avg_jumps_per_day: float
    timestamp: datetime


class SVJJumpRiskResponse(BaseModel):
    token: str
    diffusion_variance: float
    jump_variance: float
    total_model_variance: float
    empirical_variance: float
    jump_share_pct: float
    diffusion_share_pct: float
    daily_diffusion_vol: float
    daily_jump_vol: float
    daily_total_vol: float
    annualized_diffusion_vol: float
    annualized_jump_vol: float
    annualized_total_vol: float
    timestamp: datetime


class SVJJumpStats(BaseModel):
    n_jumps: int
    jump_fraction: float
    avg_jump_size: float
    jump_vol: float
    bns_statistic: float
    bns_pvalue: float
    jumps_significant: bool


class SVJParameterQuality(BaseModel):
    feller_satisfied: bool
    feller_ratio: float
    half_life_years: float
    mean_reversion_days: float
    optimization_success: bool


class SVJMomentComparison(BaseModel):
    empirical_skewness: float
    empirical_kurtosis: float
    model_variance: float
    model_skew_approx: float


class SVJEVTTail(BaseModel):
    gpd_xi: float
    gpd_beta: float
    threshold: float
    n_exceedances: int
    tail_index: float


class SVJClustering(BaseModel):
    branching_ratio: float
    half_life_days: float
    n_clusters: int
    avg_cluster_size: float
    stationarity: bool


class SVJDiagnosticsResponse(BaseModel):
    token: str
    jump_stats: SVJJumpStats
    parameter_quality: SVJParameterQuality
    moment_comparison: SVJMomentComparison
    evt_tail: Optional[SVJEVTTail] = None
    clustering: Optional[SVJClustering] = None
    timestamp: datetime


# ── Guardian (Unified Risk Veto) ─────────────────────────────────────


class GuardianAssessRequest(BaseModel):
    token: str = Field(..., description="Token symbol (must be calibrated)")
    trade_size_usd: float = Field(..., gt=0, description="Proposed trade size in USD")
    direction: str = Field(..., pattern="^(long|short)$", description="Trade direction")
    urgency: bool = Field(default=False, description="Bypass cache for urgent assessment")
    max_slippage_pct: float = Field(default=1.0, ge=0.0, le=10.0, description="Max acceptable slippage %")
    strategy: str | None = Field(default=None, description="Trading strategy: lp, arb, perp")
    run_debate: bool = Field(default=False, description="Run adversarial debate on this trade")
    agent_confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="Agent ONNX model confidence (0-1)")


class GuardianComponentScore(BaseModel):
    component: str = Field(..., description="Component name: evt, svj, hawkes, regime, news")
    score: float = Field(..., ge=0, le=100, description="Risk score 0-100")
    details: dict


class GuardianAssessResponse(BaseModel):
    approved: bool
    risk_score: float = Field(..., ge=0, le=100, description="Composite risk score 0-100")
    veto_reasons: list[str]
    recommended_size: float = Field(..., ge=0, description="Suggested position size in USD")
    regime_state: int
    confidence: float = Field(..., ge=0, le=1, description="Model agreement level")
    calibrated_confidence: float | None = None
    effective_threshold: float = Field(75.0, description="Approval threshold (may be regime-scaled)")
    hawkes_deferred: bool = False
    copula_gate_triggered: bool = False
    expires_at: str
    component_scores: list[GuardianComponentScore]
    circuit_breaker: dict | None = None
    portfolio_limits: dict | None = None
    debate: dict | None = None
    from_cache: bool = False


# ── Liquidity-Adjusted VaR (LVaR) ─────────────────────────────────


class LVaREstimateRequest(BaseModel):
    token: str = Field(..., description="Token key from _model_store (must be calibrated)")
    window: Optional[int] = Field(None, ge=5, le=200, description="Rolling window for spread estimation. None = full-sample.")
    position_value: float = Field(100_000.0, gt=0, description="Position notional in USD")
    holding_period: int = Field(1, ge=1, le=30, description="Holding period in days")
    confidence: float = Field(95.0, gt=50.0, le=99.99, description="VaR confidence level (%)")


class SpreadEstimate(BaseModel):
    spread_pct: float = Field(..., description="Estimated bid-ask spread as % of mid-price")
    spread_abs: float = Field(..., description="Absolute spread in price units")
    spread_vol_pct: float = Field(..., description="Spread volatility as % of mid-price")
    method: str = Field(..., description="Estimation method (roll)")
    n_obs: int


class LVaREstimateResponse(BaseModel):
    token: str
    lvar: float = Field(..., description="Liquidity-adjusted VaR (%)")
    base_var: float = Field(..., description="Base VaR without liquidity adjustment (%)")
    liquidity_cost_pct: float = Field(..., description="Liquidity cost component (%)")
    liquidity_cost_abs: float = Field(..., description="Liquidity cost in USD")
    lvar_abs: float = Field(..., description="LVaR in USD")
    lvar_ratio: float = Field(..., description="LVaR / VaR ratio (>1 means liquidity worsens risk)")
    spread: SpreadEstimate
    alpha: float
    holding_period: int
    position_value: float
    timestamp: datetime


class RegimeLiquidityItem(BaseModel):
    regime: int
    n_obs: int
    spread_pct: Optional[float] = None
    spread_abs: Optional[float] = None
    mean_volume: Optional[float] = None
    liquidity_score: Optional[float] = None
    insufficient_data: bool = False


class RegimeLiquidityProfileResponse(BaseModel):
    token: str
    num_states: int
    profiles: list[RegimeLiquidityItem]
    weighted_avg_spread_pct: float
    n_total: int
    timestamp: datetime


class RegimeLVaRBreakdownItem(BaseModel):
    regime: int
    probability: float
    spread_pct: float
    lvar: float
    liquidity_cost_pct: float


class RegimeLVaRResponse(BaseModel):
    token: str
    lvar: float = Field(..., description="Regime-weighted LVaR (%)")
    base_var: float
    liquidity_cost_pct: float
    regime_weighted_spread_pct: float
    regime_breakdown: list[RegimeLVaRBreakdownItem]
    alpha: float
    holding_period: int
    position_value: float
    timestamp: datetime


class MarketImpactRequest(BaseModel):
    token: str = Field(..., description="Token key from _model_store (must be calibrated)")
    trade_size_usd: float = Field(..., gt=0, description="Proposed trade size in USD")
    adv_usd: Optional[float] = Field(None, gt=0, description="Average daily volume in USD. If None, uses stored volume data.")
    participation_rate: float = Field(0.10, gt=0.0, le=1.0, description="Max acceptable participation rate")


class MarketImpactResponse(BaseModel):
    token: str
    impact_pct: float = Field(..., description="Estimated price impact (%)")
    impact_usd: float = Field(..., description="Estimated impact cost in USD")
    participation_rate: float = Field(..., description="Trade size / ADV ratio")
    participation_warning: bool = Field(..., description="True if participation exceeds threshold")
    sigma_daily: float
    trade_size_usd: float
    adv_usd: float
    timestamp: datetime



# ── Oracle (Pyth) models ──


class PythFeedAttributes(BaseModel):
    asset_type: str = ""
    base: str = ""
    description: str = ""
    display_symbol: str = ""
    generic_symbol: str = ""
    quote_currency: str = ""
    symbol: str = ""


class PythFeedItem(BaseModel):
    id: str
    attributes: PythFeedAttributes


class PythFeedListResponse(BaseModel):
    feeds: list[PythFeedItem]
    total: int
    query: str | None = None
    timestamp: datetime


class OraclePriceItem(BaseModel):
    price: float
    confidence: float
    ema_price: float
    expo: int = 0
    publish_time: int
    feed_id: str
    symbol: str = ""
    description: str = ""
    timestamp: float


class OraclePricesResponse(BaseModel):
    prices: list[OraclePriceItem]
    count: int
    source: str = "pyth"
    timestamp: datetime


class OracleHistoricalResponse(BaseModel):
    prices: list[OraclePriceItem]
    query_timestamp: int
    count: int
    timestamp: datetime


class OracleStreamStatus(BaseModel):
    active: bool
    feed_ids: list[str]
    events_received: int
    started_at: float | None


class OracleStatusResponse(BaseModel):
    hermes_url: str
    total_feeds_known: int
    feed_cache_age_s: float | None
    prices_cached: int
    buffers_active: int
    buffer_depth: int
    stream: OracleStreamStatus
    timestamp: float


# ── Stream (Helius) models ──


class StreamEvent(BaseModel):
    event_type: str
    severity: str
    signature: str
    slot: int
    timestamp: float
    details: dict = Field(default_factory=dict)


class StreamEventsResponse(BaseModel):
    events: list[StreamEvent]
    total: int
    timestamp: datetime


class StreamStatusResponse(BaseModel):
    connected: bool
    last_event_time: float | None
    events_received: int
    started_at: float | None


# ── Social sentiment models ──


class SocialSourceItem(BaseModel):
    source: str
    sentiment: float
    count: int


class SocialSentimentResponse(BaseModel):
    token: str
    overall_sentiment: float
    sources: list[SocialSourceItem]
    timestamp: float


# ── Macro indicator models ──


class FearGreedItem(BaseModel):
    value: int
    classification: str
    timestamp: int


class BtcDominanceItem(BaseModel):
    btc_dominance: float
    eth_dominance: float
    sol_dominance: float = 0.0
    total_market_cap_usd: float
    total_volume_24h_usd: float
    active_cryptocurrencies: int


class MacroIndicatorsResponse(BaseModel):
    fear_greed: FearGreedItem
    btc_dominance: BtcDominanceItem
    risk_level: str
    avg_gas_sol: float | None = None
    timestamp: float



# ── Kelly Criterion models ──


class KellyStatsResponse(BaseModel):
    active: bool
    n_trades: int = 0
    win_rate: float | None = None
    win_loss_ratio: float | None = None
    kelly_full: float | None = None
    kelly_fraction: float | None = None
    fraction_used: float | None = None
    reason: str | None = None


class TradeOutcomeRequest(BaseModel):
    pnl: float = Field(..., description="Profit/loss of the trade in USD")
    size: float = Field(..., description="Trade size in USD")
    token: str = Field("", description="Token symbol")
    regime: int = Field(-1, description="MSM regime at trade time (-1 = unknown)")
    component_scores: list[dict] | None = Field(None, description="Guardian component scores at trade time")
    risk_score: float = Field(0.0, description="Composite risk score at trade time")
    strategy: str = Field("", description="Trading strategy (spot, arbitrage, lp, perp)")


# ── Circuit Breaker models ──


class CircuitBreakerItem(BaseModel):
    name: str
    state: str
    fail_count: int
    threshold: float
    consecutive_required: int
    cooldown_seconds: float
    opened_at: float | None = None
    cooldown_remaining: float | None = None
    history_len: int = 0


class CircuitBreakersResponse(BaseModel):
    breakers: list[CircuitBreakerItem]
    timestamp: float


# ── Portfolio Risk models ──


class PositionItem(BaseModel):
    token: str
    size_usd: float
    direction: str
    entry_price: float = 0.0
    opened_at: float = 0.0


class DrawdownResponse(BaseModel):
    daily_pnl: float
    weekly_pnl: float
    daily_drawdown_pct: float
    weekly_drawdown_pct: float
    daily_limit_pct: float
    weekly_limit_pct: float
    daily_breached: bool
    weekly_breached: bool
    portfolio_value: float


class CorrelationExposure(BaseModel):
    group: str | None = None
    group_tokens: list[str] = Field(default_factory=list)
    group_exposure_usd: float = 0.0
    exposure_pct: float = 0.0
    limit_pct: float = 0.0
    breached: bool = False


class PortfolioLimitsResponse(BaseModel):
    blocked: bool
    blockers: list[str]
    drawdown: DrawdownResponse
    correlation: CorrelationExposure


class PositionsResponse(BaseModel):
    positions: list[PositionItem]
    total_exposure_usd: float
    portfolio_value: float
    timestamp: float


# ── Adversarial Debate models ──


class DebateAgentOutput(BaseModel):
    role: str
    position: str | None = None
    decision: str | None = None
    confidence: float
    arguments: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
    suggested_action: str | None = None
    trader_weight: float | None = None
    risk_weight: float | None = None


class DebateRound(BaseModel):
    round: int
    trader: DebateAgentOutput
    risk_manager: DebateAgentOutput
    arbitrator: DebateAgentOutput


class DebateResponse(BaseModel):
    final_decision: str
    final_confidence: float
    rounds: list[DebateRound]
    num_rounds: int
    elapsed_ms: float
    original_approved: bool
    decision_changed: bool


# --- Async calibration task models ---

class AsyncTaskResponse(BaseModel):
    task_id: str
    status: str
    endpoint: str
    created_at: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    endpoint: str
    result: dict | list | None = None
    error: str | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None


# ── On-Chain Liquidity (Wave 10.1) ─────────────────────────────────


class OnchainDepthRequest(BaseModel):
    pool_address: str = Field(..., description="CLMM pool address (Raydium/Orca)")
    num_ticks: int = Field(50, ge=5, le=200, description="Number of ticks per side")


class OnchainDepthResponse(BaseModel):
    pool: str
    current_price: float
    bid_prices: list[float]
    ask_prices: list[float]
    bid_depth: list[float]
    ask_depth: list[float]
    total_bid_liquidity: float
    total_ask_liquidity: float
    depth_imbalance: float
    timestamp: datetime


class DexSpreadItem(BaseModel):
    dex: str
    mean_spread_pct: float
    std_spread_pct: float
    n_swaps: int


class RealizedSpreadRequest(BaseModel):
    token_address: str = Field(..., description="Solana token mint address")
    limit: int = Field(100, ge=10, le=500, description="Number of swaps to analyze")


class RealizedSpreadResponse(BaseModel):
    token_address: str
    realized_spread_pct: float
    realized_spread_vol_pct: float
    n_swaps: int
    by_dex: list[DexSpreadItem]
    vwas_pct: float = Field(..., description="Volume-weighted average spread")
    total_volume: float
    timestamp: datetime


class OnchainLVaRRequest(BaseModel):
    token: str = Field(..., description="Token key from _model_store (must be calibrated)")
    token_address: str | None = Field(None, description="Solana token mint for on-chain spread")
    pair_address: str | None = Field(None, description="DEX pair address for live spread")
    confidence: float = Field(95.0, gt=50.0, le=99.99)
    position_value: float = Field(100_000.0, gt=0)
    holding_period: int = Field(1, ge=1, le=30)


class OnchainLVaRResponse(BaseModel):
    token: str
    lvar: float
    base_var: float
    liquidity_cost_pct: float
    spread_pct: float
    spread_source: str = Field(..., description="onchain|dexscreener|roll|default")
    by_dex: list[DexSpreadItem] | None = None
    confidence: float
    holding_period: int
    position_value: float
    timestamp: datetime


# ── Tick-Level Backtesting (Wave 10.2) ─────────────────────────────


class TickDataRequest(BaseModel):
    token_address: str = Field(..., description="Solana token mint address")
    lookback_days: int = Field(7, ge=1, le=90)
    bar_type: str = Field("time", description="time|volume|tick|imbalance")
    bar_size: int = Field(300, ge=1, description="Bar size (seconds for time, units for others)")
    limit: int = Field(1000, ge=100, le=10000)


class TickBar(BaseModel):
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    n_ticks: int
    vwap: float


class TickDataResponse(BaseModel):
    token_address: str
    bar_type: str
    bar_size: int
    n_bars: int
    bars: list[TickBar]
    timestamp: datetime


class BacktestRequest(BaseModel):
    token: str = Field(..., description="Token key from _model_store")
    token_address: str | None = Field(None, description="Solana token mint for tick data")
    horizons: list[int] = Field([60, 240], description="Backtest horizons in minutes")
    confidence: float = Field(95.0, gt=50.0, le=99.99)
    lookback_days: int = Field(30, ge=7, le=90)


class BacktestHorizonResult(BaseModel):
    horizon_minutes: int
    n_observations: int
    n_violations: int
    violation_rate: float
    expected_rate: float
    kupiec_stat: float
    kupiec_pvalue: float
    kupiec_pass: bool
    christoffersen_stat: float | None = None
    christoffersen_pvalue: float | None = None


class BacktestResponse(BaseModel):
    token: str
    confidence: float
    horizons: list[BacktestHorizonResult]
    overall_pass: bool
    timestamp: datetime


# ── On-Chain Events for Hawkes (Wave 10.3) ─────────────────────────


class OnchainEventItem(BaseModel):
    event_type: str
    slot: int
    timestamp: float
    magnitude: float
    details: dict = Field(default_factory=dict)


class OnchainEventsResponse(BaseModel):
    token_address: str
    events: list[OnchainEventItem]
    n_events: int
    event_type_counts: dict[str, int]
    timestamp: datetime


class HawkesOnchainCalibrateRequest(BaseModel):
    token_address: str = Field(..., description="Solana token mint address")
    event_types: list[str] = Field(
        ["large_swap", "oracle_jump", "liquidation"],
        description="Event types to include",
    )
    lookback_slots: int = Field(216000, ge=1000, description="Lookback window in slots (~24h = 216000)")


class CrossExcitationEntry(BaseModel):
    source: str
    target: str
    alpha: float = Field(..., description="Excitation magnitude")
    beta: float = Field(..., description="Decay rate")


class HawkesOnchainCalibrateResponse(BaseModel):
    token_address: str
    event_types: list[str]
    n_events_per_type: dict[str, int]
    mu: dict[str, float] = Field(..., description="Baseline intensity per event type")
    cross_excitation: list[CrossExcitationEntry]
    branching_matrix: list[list[float]]
    spectral_radius: float
    stationary: bool
    timestamp: datetime


class HawkesOnchainRiskResponse(BaseModel):
    token_address: str
    flash_crash_score: float = Field(..., description="0-100 risk score")
    current_intensities: dict[str, float]
    baseline_intensities: dict[str, float]
    dominant_event_type: str
    risk_level: str
    timestamp: datetime


# ── Token Supply / Tokenomics ─────────────────────────────────────


class TokenSupplyStaking(BaseModel):
    total_staked: str = "0"
    total_staked_formatted: float = 0.0
    reward_rate: str = "0"
    reward_rate_formatted: float = 0.0


class TokenSupplyTreasury(BaseModel):
    sol_balance: float = 0.0
    address: str = ""


class TokenSupplyResponse(BaseModel):
    symbol: str = "CRTX"
    decimals: int = 9
    total_supply: str = "100000000000000000"
    total_supply_formatted: float = 100_000_000.0
    mint: str = ""
    staking: TokenSupplyStaking = TokenSupplyStaking()
    treasury: TokenSupplyTreasury = TokenSupplyTreasury()
    programs: dict[str, str] = {}
    timestamp: datetime


# ── Token Info (Token Card) ───────────────────────────────────────


class TokenInfoResponse(BaseModel):
    address: str
    name: str
    symbol: str
    logo_uri: str = ""
    decimals: int = 0
    price_usd: float = 0.0
    price_change_24h_pct: float = 0.0
    market_cap: float = 0.0
    volume_24h_usd: float = 0.0
    liquidity_usd: float = 0.0
    holder_count: int = 0
    deployer: str = ""
    created_at: str | int | None = None
    dex_platform: str = ""
    timestamp: datetime


# ── Vine Copula Models ────────────────────────────────────────────


class VineCopulaFitRequest(BaseModel):
    structure: str = Field("rvine", pattern="^(rvine|cvine|dvine)$")
    family_set: list[str] | None = Field(None, description="Copula families to consider. None = all.")


class VineCopulaFitResponse(BaseModel):
    engine: str
    structure: str
    n_obs: int
    n_assets: int
    families_used: list[str]
    log_likelihood: float
    n_params: int
    aic: float
    bic: float
    timestamp: datetime


class VineCopulaVaRResponse(BaseModel):
    vine_var: float
    gaussian_var: float
    var_ratio: float
    engine: str
    structure: str
    n_params: int
    n_simulations: int
    alpha: float
    timestamp: datetime


class VineCopulaSimulateResponse(BaseModel):
    n_samples: int
    n_assets: int
    sample_mean: list[float]
    sample_std: list[float]
    timestamp: datetime


# ── ccxt Data Feed Models ─────────────────────────────────────────


class CcxtOHLCVRequest(BaseModel):
    symbol: str = Field(..., description="Trading pair, e.g. BTC/USDT")
    timeframe: str = Field("1d", description="Candle interval")
    limit: int = Field(500, ge=1, le=2000)
    exchange: str | None = Field(None, description="Exchange ID. None = default.")


class CcxtOHLCVResponse(BaseModel):
    symbol: str
    exchange: str
    timeframe: str
    n_candles: int
    first_timestamp: str
    last_timestamp: str
    last_close: float
    timestamp: datetime


class CcxtOrderBookResponse(BaseModel):
    symbol: str
    exchange: str
    best_bid: float
    best_ask: float
    mid_price: float
    spread: float
    spread_bps: float
    bid_depth: float
    ask_depth: float
    timestamp: datetime


class CcxtTickerResponse(BaseModel):
    symbol: str
    exchange: str
    last: float | None
    high: float | None
    low: float | None
    volume: float | None
    quote_volume: float | None
    change_pct: float | None
    vwap: float | None
    bid: float | None
    ask: float | None
    timestamp: datetime


class CcxtExchangesResponse(BaseModel):
    exchanges: list[str]
    count: int


# ── Portfolio Optimization Models ─────────────────────────────────


class PortfolioOptRequest(BaseModel):
    tokens: list[str] = Field(..., min_length=2, description="Ticker symbols or trading pairs")
    period: str = Field("2y", description="Data period (e.g. '1y', '2y')")
    data_source: DataSource = DataSource.YFINANCE
    cvar_beta: float = Field(0.95, gt=0.5, le=0.999)
    max_weight: float = Field(0.40, gt=0.0, le=1.0)


class PortfolioOptWeights(BaseModel):
    method: str
    engine: str
    weights: dict[str, float]
    expected_return: float
    cvar: float | None = None
    cvar_beta: float | None = None
    variance: float | None = None
    n_assets: int


class PortfolioOptCompareResponse(BaseModel):
    strategies: list[PortfolioOptWeights]
    n_assets: int
    n_observations: int
    timestamp: datetime


# ── Walk-Forward Backtesting ──────────────────────────────────────


class WalkForwardRequest(BaseModel):
    token: str = Field(..., description="Token key from _model_store (must be calibrated)")
    min_train_window: int = Field(120, ge=30, le=500, description="Minimum training window in days")
    step_size: int = Field(1, ge=1, le=20, description="Steps between out-of-sample points")
    refit_interval: int = Field(20, ge=1, le=100, description="Steps between model recalibrations")
    expanding: bool = Field(True, description="True=expanding window, False=rolling")
    max_train_window: Optional[int] = Field(None, ge=50, description="Max window size for rolling mode")
    confidence: float = Field(95.0, gt=50.0, le=99.99, description="VaR confidence level (%)")
    num_states: int = Field(5, ge=2, le=10)
    method: str = Field("empirical", pattern="^(mle|grid|empirical|hybrid)$")
    use_student_t: bool = False
    nu: float = Field(5.0, gt=2.0)


class WalkForwardKupiecResult(BaseModel):
    statistic: float
    p_value: float
    pass_: bool = Field(alias="pass")
    violation_rate: Optional[float] = None
    expected_rate: Optional[float] = None

    class Config:
        populate_by_name = True


class WalkForwardRegimeResult(BaseModel):
    regime: int
    regime_name: str
    n_obs: int
    n_violations: int
    violation_rate: float
    mean_return: Optional[float] = None
    volatility: Optional[float] = None
    sharpe: Optional[float] = None
    kupiec: Optional[WalkForwardKupiecResult] = None
    insufficient_data: bool = False


class WalkForwardParameterStability(BaseModel):
    n_refits: int
    stable: bool
    sigma_low: Optional[dict] = None
    sigma_high: Optional[dict] = None


class WalkForwardHealthCheck(BaseModel):
    pass_: bool = Field(alias="pass")
    flags: list[str]

    class Config:
        populate_by_name = True


class WalkForwardReportResponse(BaseModel):
    token: str
    overall: dict
    per_regime: list[WalkForwardRegimeResult]
    parameter_stability: WalkForwardParameterStability
    health: WalkForwardHealthCheck
    n_calibration_snapshots: int
    elapsed_ms: float
    timestamp: datetime


class HistoricalExportResponse(BaseModel):
    token: str
    n_observations: int
    regime_timeline: list[dict]
    regime_statistics: list[dict]
    calibration: dict
    calibrated_at: str
    timestamp: str


# ── DexScreener Response Models ──


class DexTokenPriceResponse(BaseModel):
    source: str = "dexscreener"
    token_address: str
    price_usd: Optional[float] = None
    price_native: Optional[float] = None
    pair_address: Optional[str] = None
    dex_id: Optional[str] = None
    liquidity_usd: float = 0.0
    volume_24h: float = 0.0
    error: Optional[str] = None
    timestamp: float
    timestamp_iso: Optional[str] = None


class DexPairLiquidityResponse(BaseModel):
    source: str = "dexscreener"
    pair_address: str
    price_usd: Optional[float] = None
    liquidity_usd: float = 0.0
    liquidity_base: float = 0.0
    liquidity_quote: float = 0.0
    volume_24h: float = 0.0
    volume_6h: float = 0.0
    volume_1h: float = 0.0
    txns_24h: Optional[dict] = None
    price_change_24h: float = 0.0
    dex_id: Optional[str] = None
    error: Optional[str] = None
    timestamp: float


class DexLiquidityMetricsResponse(BaseModel):
    source: str = "dexscreener"
    pair_address: Optional[str] = None
    spread_pct: Optional[float] = None
    spread_vol_pct: Optional[float] = None
    volume_24h: Optional[float] = None
    liquidity: Optional[float] = None
    error: Optional[str] = None
    timestamp: Optional[float] = None


class DexNewToken(BaseModel):
    token_address: str
    pair_address: Optional[str] = None
    dex_id: Optional[str] = None
    price_usd: Optional[float] = None
    liquidity_usd: float = 0.0
    volume_24h: float = 0.0
    pair_created_at: Optional[int] = None
    base_token: Optional[dict] = None
    timestamp: float
    meets_min_liquidity: bool = False


class DexNewTokensResponse(BaseModel):
    tokens: list[DexNewToken]


class DexStatusResponse(BaseModel):
    available: bool
    timestamp: str


# ── Narrator (LLM-powered narrative engine) ────────────────────────


class NarratorExplainRequest(BaseModel):
    token: str = Field("", description="Token symbol")
    direction: str = Field("long", description="Trade direction")
    trade_size_usd: float = Field(0.0, ge=0, description="Proposed trade size in USD")
    strategy: str | None = Field(None, description="Trading strategy")
    assessment: dict | None = Field(None, description="Full guardian assessment dict. If None, runs live assessment.")


class NarratorExplainResponse(BaseModel):
    enabled: bool
    narrative: str | None = None
    model: str | None = None
    latency_ms: float = 0
    token: str = ""
    direction: str = ""
    error: str | None = None


class NarratorNewsRequest(BaseModel):
    news_items: list[dict] | None = Field(None, description="News item dicts. If None, reads from buffer.")
    news_signal: dict | None = Field(None, description="Aggregate signal dict. If None, reads from buffer.")


class NarratorNewsResponse(BaseModel):
    enabled: bool
    interpretation: str | None = None
    model: str | None = None
    latency_ms: float = 0
    n_items: int = 0
    error: str | None = None


class NarratorBriefingResponse(BaseModel):
    enabled: bool
    briefing: str | None = None
    model: str | None = None
    latency_ms: float = 0
    error: str | None = None


class NarratorAskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Operator question about the system")
    context: dict | None = Field(None, description="Optional additional context to inject")


class NarratorAskResponse(BaseModel):
    enabled: bool
    answer: str | None = None
    model: str | None = None
    latency_ms: float = 0
    question: str = ""
    error: str | None = None


class NarratorStatusResponse(BaseModel):
    enabled: bool
    model: str
    api_key_set: bool
    call_count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0
    total_latency_ms: float = 0