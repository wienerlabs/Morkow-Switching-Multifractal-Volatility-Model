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
    p_stay: float
    sigma_states: list[float]
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
    p_stay: float
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