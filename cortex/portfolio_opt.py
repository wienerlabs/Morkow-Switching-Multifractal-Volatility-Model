"""Portfolio optimization via skfolio (scikit-portfolio).

Provides Mean-CVaR, Hierarchical Risk Parity, and other advanced portfolio
optimization methods through an sklearn-compatible API. Falls back to
simple equal-weight or numpy-based optimization when skfolio is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SKFOLIO_AVAILABLE = False
try:
    from skfolio import RiskMeasure, ObjectiveFunction
    from skfolio.optimization import MeanRisk, HierarchicalRiskParity, EqualWeighted
    from skfolio.preprocessing import prices_to_returns
    _SKFOLIO_AVAILABLE = True
except ImportError:
    pass


def optimize_mean_cvar(
    returns: pd.DataFrame,
    cvar_beta: float = 0.95,
    max_weight: float = 0.40,
) -> dict[str, Any]:
    """Optimize portfolio weights to maximize return/CVaR ratio.

    Args:
        returns: DataFrame of asset returns (columns = assets).
        cvar_beta: CVaR confidence level (default 95%).
        max_weight: Maximum weight per asset.

    Returns:
        Dict with weights, expected_return, cvar, sharpe_ratio, method.

    Raises:
        RuntimeError: If skfolio is not installed.
    """
    if not _SKFOLIO_AVAILABLE:
        raise RuntimeError("skfolio not installed. Install with: pip install skfolio")

    model = MeanRisk(
        risk_measure=RiskMeasure.CVAR,
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        cvar_beta=cvar_beta,
        max_weights=max_weight,
    )
    model.fit(returns)
    weights = dict(zip(returns.columns, model.weights_))
    portfolio = model.predict(returns)

    return {
        "method": "mean_cvar",
        "engine": "skfolio",
        "weights": weights,
        "expected_return": float(portfolio.mean),
        "cvar": float(portfolio.cvar),
        "cvar_beta": cvar_beta,
        "n_assets": len(weights),
    }


def optimize_hrp(
    returns: pd.DataFrame,
) -> dict[str, Any]:
    """Hierarchical Risk Parity — distribution-free portfolio optimization.

    Uses hierarchical clustering to build a diversified portfolio without
    assuming any specific return distribution.

    Args:
        returns: DataFrame of asset returns.

    Returns:
        Dict with weights, method.
    """
    if not _SKFOLIO_AVAILABLE:
        raise RuntimeError("skfolio not installed. Install with: pip install skfolio")

    model = HierarchicalRiskParity()
    model.fit(returns)
    weights = dict(zip(returns.columns, model.weights_))
    portfolio = model.predict(returns)

    return {
        "method": "hrp",
        "engine": "skfolio",
        "weights": weights,
        "expected_return": float(portfolio.mean),
        "n_assets": len(weights),
    }


def optimize_min_variance(
    returns: pd.DataFrame,
    max_weight: float = 0.40,
) -> dict[str, Any]:
    """Minimum variance portfolio.

    Args:
        returns: DataFrame of asset returns.
        max_weight: Maximum weight per asset.

    Returns:
        Dict with weights, variance, method.
    """
    if not _SKFOLIO_AVAILABLE:
        raise RuntimeError("skfolio not installed. Install with: pip install skfolio")

    model = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        max_weights=max_weight,
    )
    model.fit(returns)
    weights = dict(zip(returns.columns, model.weights_))
    portfolio = model.predict(returns)

    return {
        "method": "min_variance",
        "engine": "skfolio",
        "weights": weights,
        "expected_return": float(portfolio.mean),
        "variance": float(portfolio.variance),
        "n_assets": len(weights),
    }


def optimize_equal_weight(
    returns: pd.DataFrame,
) -> dict[str, Any]:
    """Equal weight benchmark (1/N portfolio). No external dependencies needed."""
    n = len(returns.columns)
    w = 1.0 / n
    weights = {col: w for col in returns.columns}

    port_ret = returns.mean(axis=1)
    return {
        "method": "equal_weight",
        "engine": "native",
        "weights": weights,
        "expected_return": float(port_ret.mean()),
        "n_assets": n,
    }


def compare_strategies(
    returns: pd.DataFrame,
    cvar_beta: float = 0.95,
    max_weight: float = 0.40,
) -> list[dict[str, Any]]:
    """Run all available optimization strategies and compare results.

    Always includes equal_weight. Adds skfolio strategies when available.

    Args:
        returns: DataFrame of asset returns.
        cvar_beta: CVaR confidence level.
        max_weight: Maximum weight per asset.

    Returns:
        List of dicts, one per strategy, sorted by expected_return desc.
    """
    results = [optimize_equal_weight(returns)]

    if _SKFOLIO_AVAILABLE:
        for fn, kwargs in [
            (optimize_mean_cvar, {"cvar_beta": cvar_beta, "max_weight": max_weight}),
            (optimize_hrp, {}),
            (optimize_min_variance, {"max_weight": max_weight}),
        ]:
            try:
                results.append(fn(returns, **kwargs))
            except Exception as e:
                logger.warning("Strategy %s failed: %s", fn.__name__, e)

    results.sort(key=lambda r: r.get("expected_return", 0), reverse=True)
    return results
