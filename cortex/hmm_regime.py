"""3-state GaussianHMM regime detector with probability-weighted multipliers."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class HMMRegimeDetector:
    """Detect market regimes using a 3-state Hidden Markov Model."""

    def __init__(
        self,
        n_states: int = 3,
        min_bars: int = 100,
        retrain_interval: int = 24,
    ) -> None:
        self._n_states = n_states
        self._min_bars = min_bars
        self._retrain_interval = retrain_interval
        self._model = None
        self._fitted = False
        self._fit_count = 0
        self._samples_since_fit = 0
        self._state_multipliers: dict[int, float] = {}
        self._current_state: int | None = None
        self._state_probs: np.ndarray | None = None

    def _build_features(self, returns: np.ndarray, volumes: np.ndarray) -> np.ndarray | None:
        min_len = min(len(returns), len(volumes))
        if min_len < self._min_bars:
            return None
        returns = returns[:min_len]
        volumes = volumes[:min_len]
        log_vol = np.log1p(np.abs(volumes))
        return np.column_stack([returns, log_vol])

    def fit(self, returns: np.ndarray, volumes: np.ndarray) -> bool:
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.warning("hmmlearn not installed, HMM regime detection disabled")
            return False

        X = self._build_features(returns, volumes)
        if X is None:
            return False

        try:
            model = GaussianHMM(
                n_components=self._n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42,
            )
            model.fit(X)
            self._model = model
            self._fitted = True
            self._fit_count += 1
            self._samples_since_fit = 0
            self._classify_states()
            return True
        except Exception as e:
            logger.warning("hmm_fit_failed: %s", e)
            return False

    def _classify_states(self) -> None:
        if self._model is None:
            return
        vols = []
        for i in range(self._n_states):
            vols.append(np.sqrt(self._model.covars_[i][0, 0]))
        order = np.argsort(vols)
        self._state_multipliers = {}
        multiplier_map = {0: 1.0, 1: 1.2, 2: 0.6}  # calm, trending, volatile
        for rank, state_idx in enumerate(order):
            self._state_multipliers[int(state_idx)] = multiplier_map.get(rank, 1.0)

    def predict(self, returns: np.ndarray, volumes: np.ndarray) -> int | None:
        if not self._fitted or self._model is None:
            return None
        X = self._build_features(returns, volumes)
        if X is None:
            return None
        try:
            states = self._model.predict(X)
            self._current_state = int(states[-1])
            proba = self._model.predict_proba(X)
            self._state_probs = proba[-1]
            self._samples_since_fit += 1
            return self._current_state
        except Exception as e:
            logger.warning("hmm_predict_failed: %s", e)
            return None

    def get_regime_multipliers(self) -> dict[str, float] | None:
        if not self._fitted or self._state_probs is None:
            return None
        weighted = 0.0
        for state_idx, prob in enumerate(self._state_probs):
            mult = self._state_multipliers.get(state_idx, 1.0)
            weighted += prob * mult
        return {"regime_multiplier": weighted}

    def needs_retrain(self) -> bool:
        return self._samples_since_fit >= self._retrain_interval

    @property
    def current_state(self) -> int | None:
        return self._current_state

    @property
    def state_multipliers(self) -> dict[int, float]:
        return self._state_multipliers
