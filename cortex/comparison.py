"""
Volatility model benchmarking framework.

Compares MSM-VaR against GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1,1),
Rolling Window, and EWMA (RiskMetrics) using standardized backtesting.
"""

import logging
import math
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm

from cortex import msm as msm_module

logger = logging.getLogger(__name__)

MIN_OBS = 30


def garch_vol_forecast(returns: pd.Series) -> pd.Series:
    """GARCH(1,1) one-step-ahead volatility via expanding window."""
    return _arch_expanding_forecast(returns, vol="Garch", p=1, o=0, q=1)


def egarch_vol_forecast(returns: pd.Series) -> pd.Series:
    """EGARCH(1,1) one-step-ahead volatility via expanding window."""
    return _arch_expanding_forecast(returns, vol="EGARCH", p=1, o=0, q=1)


def gjr_garch_vol_forecast(returns: pd.Series) -> pd.Series:
    """GJR-GARCH(1,1,1) one-step-ahead volatility via expanding window."""
    return _arch_expanding_forecast(returns, vol="Garch", p=1, o=1, q=1)


def _arch_expanding_forecast(
    returns: pd.Series, vol: str, p: int, o: int, q: int
) -> pd.Series:
    """Shared expanding-window forecast for arch-family models."""
    r = returns.values.astype(float)
    n = len(r)
    forecasts = np.full(n, np.nan)

    for t in range(MIN_OBS, n):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                am = arch_model(
                    r[:t], vol=vol, p=p, o=o, q=q, mean="Zero", rescale=False
                )
                res = am.fit(disp="off", show_warning=False)
                fcast = res.forecast(horizon=1)
                forecasts[t] = float(np.sqrt(fcast.variance.values[-1, 0]))
        except Exception as e:
            logger.debug("GARCH fit failed at t=%d: %s", t, e)
            if t > MIN_OBS and np.isfinite(forecasts[t - 1]):
                forecasts[t] = forecasts[t - 1]

    return pd.Series(forecasts, index=returns.index, name=f"{vol}_forecast")


def rolling_vol_forecast(returns: pd.Series, window: int = 20) -> pd.Series:
    """Historical rolling-window volatility, shifted for out-of-sample."""
    vol = returns.rolling(window).std().shift(1)
    return vol.rename(f"rolling_{window}_forecast")


def ewma_vol_forecast(
    returns: pd.Series, lambda_decay: float = 0.94
) -> pd.Series:
    """EWMA (RiskMetrics) volatility forecast with decay λ."""
    r = returns.values.astype(float)
    n = len(r)
    var_series = np.full(n, np.nan)

    init_var = np.var(r[:MIN_OBS]) if n >= MIN_OBS else np.var(r)
    var_series[MIN_OBS - 1] = init_var

    for t in range(MIN_OBS, n):
        var_series[t] = lambda_decay * var_series[t - 1] + (1 - lambda_decay) * r[t - 1] ** 2

    sigma = np.sqrt(var_series)
    return pd.Series(sigma, index=returns.index, name="ewma_forecast")


def _gaussian_ll(returns: np.ndarray, sigma: np.ndarray) -> float:
    """Gaussian log-likelihood for a volatility forecast series."""
    mask = np.isfinite(sigma) & (sigma > 1e-12)
    r, s = returns[mask], sigma[mask]
    if len(r) == 0:
        return -np.inf
    return float(np.sum(-0.5 * np.log(2 * np.pi) - np.log(s) - 0.5 * (r / s) ** 2))


_MODEL_REGISTRY: dict[str, tuple[str, int]] = {
    "msm": ("MSM", 3),
    "garch": ("GARCH(1,1)", 3),
    "egarch": ("EGARCH(1,1)", 4),
    "gjr": ("GJR-GARCH(1,1,1)", 4),
    "rolling_20": ("Rolling-20d", 0),
    "rolling_60": ("Rolling-60d", 0),
    "ewma": ("EWMA", 1),
    "rough_bergomi": ("Rough-Bergomi", 3),
    "svj": ("SVJ-Bates", 7),
}


def compare_models(
    returns: pd.Series,
    alpha: float = 0.05,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run all specified volatility models and produce a comparison table.

    Args:
        returns: Daily log-returns in % with DatetimeIndex.
        alpha: VaR confidence level (default 5%).
        models: List of model keys from _MODEL_REGISTRY. None = all.

    Returns:
        DataFrame with one row per model and standardized quality metrics.
    """
    if models is None:
        models = list(_MODEL_REGISTRY.keys())

    forecasts = _generate_all_forecasts(returns, models)
    z = norm.ppf(alpha)
    r = returns.values.astype(float)
    rows: list[dict] = []

    for key in models:
        display_name, num_params = _MODEL_REGISTRY[key]
        sigma = forecasts[key].values.astype(float)
        var_line = z * sigma
        breaches = (r < var_line).astype(int)

        valid = np.isfinite(sigma) & (sigma > 1e-12)
        breach_valid = breaches[valid]
        n_valid = int(valid.sum())
        breach_count = int(breach_valid.sum())
        breach_rate = breach_count / n_valid if n_valid > 0 else np.nan

        kup_lr, kup_p, _, _ = msm_module.kupiec_test(breach_valid, alpha)
        chr_lr, chr_p, _ = msm_module.christoffersen_independence_test(breach_valid)

        ll = _gaussian_ll(r, sigma)
        n_obs = int(np.sum(np.isfinite(sigma) & (sigma > 1e-12)))
        aic = 2 * num_params - 2 * ll if np.isfinite(ll) else np.nan
        bic = num_params * math.log(n_obs) - 2 * ll if np.isfinite(ll) and n_obs > 0 else np.nan

        realized = np.abs(r)
        mae = float(np.nanmean(np.abs(realized[valid] - sigma[valid])))
        corr = float(np.corrcoef(realized[valid], sigma[valid])[0, 1]) if n_valid > 2 else np.nan

        rows.append({
            "model": display_name,
            "log_likelihood": round(ll, 2) if np.isfinite(ll) else None,
            "aic": round(aic, 2) if np.isfinite(aic) else None,
            "bic": round(bic, 2) if np.isfinite(bic) else None,
            "breach_rate": round(breach_rate, 4) if np.isfinite(breach_rate) else None,
            "breach_count": breach_count,
            "kupiec_lr": round(float(kup_lr), 4) if np.isfinite(kup_lr) else None,
            "kupiec_pvalue": round(float(kup_p), 4) if np.isfinite(kup_p) else None,
            "kupiec_pass": bool(kup_p > 0.05) if np.isfinite(kup_p) else None,
            "christoffersen_lr": round(float(chr_lr), 4) if np.isfinite(chr_lr) else None,
            "christoffersen_pvalue": round(float(chr_p), 4) if np.isfinite(chr_p) else None,
            "christoffersen_pass": bool(chr_p > 0.05) if np.isfinite(chr_p) else None,
            "mae_volatility": round(mae, 4),
            "correlation": round(corr, 4) if np.isfinite(corr) else None,
            "num_params": num_params,
        })

    df = pd.DataFrame(rows)
    df.index = df["model"]
    return df


def _generate_all_forecasts(
    returns: pd.Series, models: list[str]
) -> dict[str, pd.Series]:
    """Dispatch each model key to its forecast function."""
    cal = None
    out: dict[str, pd.Series] = {}

    for key in models:
        if key == "msm":
            cal = msm_module.calibrate_msm_advanced(
                returns, num_states=5, method="hybrid", verbose=False
            )
            sigma_f, _, _, _, _ = msm_module.msm_vol_forecast(
                returns,
                num_states=cal["num_states"],
                sigma_low=cal["sigma_low"],
                sigma_high=cal["sigma_high"],
                p_stay=cal["p_stay"],
            )
            out[key] = sigma_f
        elif key == "garch":
            out[key] = garch_vol_forecast(returns)
        elif key == "egarch":
            out[key] = egarch_vol_forecast(returns)
        elif key == "gjr":
            out[key] = gjr_garch_vol_forecast(returns)
        elif key == "rolling_20":
            out[key] = rolling_vol_forecast(returns, window=20)
        elif key == "rolling_60":
            out[key] = rolling_vol_forecast(returns, window=60)
        elif key == "ewma":
            out[key] = ewma_vol_forecast(returns)
        elif key == "rough_bergomi":
            from cortex.rough_vol import rough_bergomi_vol_series
            out[key] = rough_bergomi_vol_series(returns)
        elif key == "svj":
            from cortex.svj import svj_vol_series
            out[key] = svj_vol_series(returns)
        else:
            raise ValueError(f"Unknown model key: {key}")

    return out


def generate_comparison_report(
    comparison_df: pd.DataFrame, alpha: float = 0.05
) -> dict:
    """
    Analyze comparison results and produce a structured report.

    Args:
        comparison_df: Output of compare_models().
        alpha: Target VaR level used in the comparison.

    Returns:
        Dict with keys: summary_table, winners, pass_fail, ranking.
    """
    df = comparison_df.copy()

    # --- Summary table (markdown) ---
    header = "| Model | LL | AIC | BIC | Breach% | Kupiec p | Chris p | MAE | Corr |"
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['model']} "
            f"| {_fmt(row['log_likelihood'])} "
            f"| {_fmt(row['aic'])} "
            f"| {_fmt(row['bic'])} "
            f"| {_pct(row['breach_rate'])} "
            f"| {_fmt(row['kupiec_pvalue'])} "
            f"| {_fmt(row['christoffersen_pvalue'])} "
            f"| {_fmt(row['mae_volatility'])} "
            f"| {_fmt(row['correlation'])} |"
        )
    summary_table = "\n".join(lines)

    # --- Winners per metric ---
    winners = {
        "best_aic": _best(df, "aic", lower=True),
        "best_bic": _best(df, "bic", lower=True),
        "best_breach": _closest_to(df, "breach_rate", alpha),
        "best_kupiec": _best(df, "kupiec_pvalue", lower=False),
        "best_christoffersen": _best(df, "christoffersen_pvalue", lower=False),
        "best_mae": _best(df, "mae_volatility", lower=True),
        "best_correlation": _best(df, "correlation", lower=False),
    }

    # --- Pass/Fail summary ---
    pass_fail = {}
    for _, row in df.iterrows():
        pass_fail[row["model"]] = {
            "kupiec": row["kupiec_pass"],
            "christoffersen": row["christoffersen_pass"],
        }

    # --- Composite ranking ---
    ranking = _compute_ranking(df, alpha)

    return {
        "summary_table": summary_table,
        "winners": winners,
        "pass_fail": pass_fail,
        "ranking": ranking,
    }


def _compute_ranking(df: pd.DataFrame, alpha: float) -> list[str]:
    """Weighted composite score ranking (lower = better)."""
    scores = pd.Series(0.0, index=df.index)
    metrics = {
        "aic": (0.15, True),
        "bic": (0.15, True),
        "breach_rate": (0.25, None),  # special: distance to alpha
        "kupiec_pvalue": (0.15, False),
        "christoffersen_pvalue": (0.15, False),
        "correlation": (0.15, False),
    }

    for col, (weight, lower_better) in metrics.items():
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().all():
            continue
        if col == "breach_rate":
            vals = (vals - alpha).abs()
            normed = _min_max_normalize(vals)
        elif lower_better:
            normed = _min_max_normalize(vals)
        else:
            normed = 1.0 - _min_max_normalize(vals)
        scores += weight * normed.fillna(1.0)

    return list(df.loc[scores.sort_values().index, "model"])


def _min_max_normalize(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=s.index)


def _best(df: pd.DataFrame, col: str, lower: bool) -> str:
    vals = pd.to_numeric(df[col], errors="coerce")
    idx = vals.idxmin() if lower else vals.idxmax()
    return str(df.loc[idx, "model"]) if pd.notna(idx) else "N/A"


def _closest_to(df: pd.DataFrame, col: str, target: float) -> str:
    vals = pd.to_numeric(df[col], errors="coerce")
    idx = (vals - target).abs().idxmin()
    return str(df.loc[idx, "model"]) if pd.notna(idx) else "N/A"


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def _pct(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.2%}"

