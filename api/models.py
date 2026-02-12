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
    leverage_gamma: float | str | None = Field(
        None,
        description="Asymmetric leverage parameter. None=no leverage, float=fixed value, 'estimate'=MLE estimation",
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
    expires_at: str
    component_scores: list[GuardianComponentScore]
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


class OraclePriceItem(BaseModel):
    token: str
    price: float
    confidence: float
    ema_price: float
    publish_time: int
    feed_id: str
    timestamp: float


class OraclePricesResponse(BaseModel):
    prices: dict[str, OraclePriceItem]
    source: str = "pyth"
    timestamp: datetime


class OracleStatusResponse(BaseModel):
    tracked_tokens: list[str]
    buffer_depth: int
    prices_cached: int
    buffer_sizes: dict[str, int]
    hermes_url: str
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
    total_market_cap_usd: float
    total_volume_24h_usd: float
    active_cryptocurrencies: int


class MacroIndicatorsResponse(BaseModel):
    fear_greed: FearGreedItem
    btc_dominance: BtcDominanceItem
    risk_level: str
    timestamp: float