"""SOL/BTC Johansen cointegration z-score signal."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CointegrationSignal:
    z_score: float
    direction: str | None
    confidence: float
    is_valid: bool
    half_life: float = 0.0


class CointegrationModule:
    """Compute SOL/BTC cointegration z-score signal using Johansen test."""

    def __init__(
        self,
        entry_zscore: float = 2.0,
        max_half_life: float = 72.0,
    ) -> None:
        self._entry_zscore = entry_zscore
        self._max_half_life = max_half_life

    def compute(
        self,
        sol_prices: np.ndarray | list,
        btc_prices: np.ndarray | list,
        lookback: int = 168,
    ) -> CointegrationSignal:
        sol = np.asarray(sol_prices, dtype=float)
        btc = np.asarray(btc_prices, dtype=float)

        min_len = min(len(sol), len(btc))
        if min_len < lookback:
            return CointegrationSignal(
                z_score=0.0, direction=None, confidence=0.0, is_valid=False
            )

        sol_window = sol[-lookback:]
        btc_window = btc[-lookback:]

        try:
            from statsmodels.tsa.vector_ar.vecm import coint_johansen

            data = np.column_stack([sol_window, btc_window])
            result = coint_johansen(data, det_order=0, k_ar_diff=1)

            trace_stat = float(result.lr1[0])
            critical_value = float(result.cvt[0, 1])  # 95% critical value

            if trace_stat <= critical_value:
                return CointegrationSignal(
                    z_score=0.0, direction=None, confidence=0.0, is_valid=False
                )

            coint_vec = result.evec[:, 0]
            spread = sol_window * coint_vec[0] + btc_window * coint_vec[1]

            spread_mean = spread.mean()
            spread_std = spread.std()
            if spread_std < 1e-10:
                return CointegrationSignal(
                    z_score=0.0, direction=None, confidence=0.0, is_valid=False
                )

            z_score = float((spread[-1] - spread_mean) / spread_std)

            # Half-life estimation via OLS
            spread_lag = spread[:-1] - spread_mean
            spread_diff = np.diff(spread)
            if len(spread_lag) > 0 and np.std(spread_lag) > 1e-10:
                beta = float(np.sum(spread_lag * spread_diff) / np.sum(spread_lag ** 2))
                half_life = -np.log(2) / beta if beta < 0 else 999.0
            else:
                half_life = 999.0

            is_valid = half_life < self._max_half_life

            direction = None
            if abs(z_score) >= self._entry_zscore:
                direction = "long" if z_score < 0 else "short"

            confidence = min(0.9, abs(z_score) / 4.0)

            return CointegrationSignal(
                z_score=z_score,
                direction=direction,
                confidence=confidence,
                is_valid=is_valid,
                half_life=half_life,
            )

        except ImportError:
            logger.warning("statsmodels not installed, cointegration disabled")
            return CointegrationSignal(
                z_score=0.0, direction=None, confidence=0.0, is_valid=False
            )
        except Exception as e:
            logger.warning("cointegration_compute_failed: %s", e)
            return CointegrationSignal(
                z_score=0.0, direction=None, confidence=0.0, is_valid=False
            )
