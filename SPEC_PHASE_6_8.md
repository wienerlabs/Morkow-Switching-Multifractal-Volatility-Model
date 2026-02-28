# Phase 6-8: Multi-Agent Alpha Engine

## Phase 6: Sharpe Contribution Weights
Direction-correctness Sharpe MVO for agent weight optimization.
- `SignalPnLTracker` with min_samples=5
- Two-layer: sharpe_base × regime_multiplier × confidence

## Phase 7: HMM Regime Detection
3-state GaussianHMM regime detector.
- Calm (×1.0), Trending (×1.2), Volatile (×0.6)
- Lazy hmmlearn import

## Phase 8: SOL/BTC Cointegration
Johansen cointegration z-score signal.
- Multiplier -10.0, clip [-25, 25]
- Direction override when |z_score| >= 2.0
- coint_lookback=168 (1 week)

All features default OFF, independently toggleable.
