"""
A-LAMS-VaR: Asymmetric Liquidity-Adjusted Markov-Switching Value-at-Risk

5-regime Markov-switching model with:
- Hamilton filter for regime probability estimation
- Asymmetric transition matrix (negative returns -> higher P of high-vol regimes)
- AMM liquidity adjustment (regime-dependent slippage)
- Two-stage MLE estimation via scipy.optimize

Mathematical framework:
  VaR_α(t) = Σ_k P(S_t=k | Ω_{t-1}) · [μ_k + σ_k · Φ^{-1}(α)] + λ(k, V_t) · Slippage(k)

References:
  - Hamilton (1989) - Regime switching models
  - Diebold, Lee, Weinbach (1994) - Asymmetric transitions
  - Cortex whitepaper - Liquidity adjustment for AMM slippage
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import optimize, stats

import structlog

logger = structlog.get_logger(__name__)

# ============= CONFIGURATION =============


@dataclass
class ALAMSConfig:
    """Configuration for the A-LAMS-VaR model."""

    n_regimes: int = 5
    confidence_levels: list[float] = field(default_factory=lambda: [0.95, 0.99])
    min_observations: int = 100
    max_iter: int = 200
    tol: float = 1e-6
    asymmetry_prior: float = 0.15
    regularization: float = 1e-4


@dataclass
class LiquidityConfig:
    """Configuration for AMM liquidity adjustment."""

    base_slippage_bps: float = 5.0
    regime_multipliers: list[float] = field(
        default_factory=lambda: [0.5, 0.8, 1.0, 1.5, 3.0]
    )
    impact_exponent: float = 1.5
    max_slippage_bps: float = 500.0


# ============= MODEL =============


class ALAMSVaRModel:
    """
    Asymmetric Liquidity-Adjusted Markov-Switching VaR.

    Estimates a K-regime model where each regime k has its own
    mean μ_k and variance σ²_k. Transition probabilities depend
    on the sign of past returns (asymmetry parameter δ).
    """

    def __init__(
        self,
        config: Optional[ALAMSConfig] = None,
        liquidity_config: Optional[LiquidityConfig] = None,
    ):
        self.config = config or ALAMSConfig()
        self.liquidity_config = liquidity_config or LiquidityConfig()

        K = self.config.n_regimes
        self.K = K

        # Regime parameters (estimated by fit)
        self.mu: np.ndarray = np.zeros(K)
        self.sigma: np.ndarray = np.ones(K)

        # Transition matrix P[i,j] = P(S_t=j | S_{t-1}=i, r > 0)
        self.P_base: np.ndarray = np.eye(K) * 0.9 + np.ones((K, K)) * 0.1 / K

        # Asymmetry parameter: shifts toward higher regimes after negative returns
        self.delta: float = self.config.asymmetry_prior

        # Filtered probabilities from last filter() call
        self.filtered_probs: Optional[np.ndarray] = None
        self.log_likelihood: float = -np.inf
        self.is_fitted: bool = False
        self.n_obs: int = 0

    # ============= ESTIMATION =============

    def fit(self, returns: np.ndarray) -> dict[str, float]:
        """
        Estimate model parameters via two-stage MLE.

        Stage 1: Estimate regime means and variances
        Stage 2: Estimate transition matrix and asymmetry parameter δ

        Args:
            returns: Array of log returns (T,)

        Returns:
            Dict with estimation diagnostics (log_likelihood, delta, n_obs, aic, bic)
        """
        returns = np.asarray(returns, dtype=np.float64)
        T = len(returns)

        if T < self.config.min_observations:
            raise ValueError(
                f"Need at least {self.config.min_observations} observations, got {T}"
            )

        self.n_obs = T
        K = self.K

        # Initialize regime params via quantile-based clustering
        self._initialize_params(returns)

        # Stage 1: Estimate μ_k, σ_k and P_base jointly
        logger.info("alams_var.fit.stage1", n_obs=T, n_regimes=K)
        result1 = self._estimate_stage1(returns)

        # Stage 2: Estimate asymmetry parameter δ
        logger.info("alams_var.fit.stage2", initial_delta=self.delta)
        result2 = self._estimate_stage2(returns)

        # Final filter pass with estimated parameters
        self.filtered_probs, self.log_likelihood = self._hamilton_filter(returns)

        self.is_fitted = True

        n_params = K * 2 + K * (K - 1) + 1  # mu, sigma, transitions, delta
        aic = -2 * self.log_likelihood + 2 * n_params
        bic = -2 * self.log_likelihood + n_params * np.log(T)

        diagnostics = {
            "log_likelihood": self.log_likelihood,
            "delta": self.delta,
            "n_obs": T,
            "n_regimes": K,
            "aic": aic,
            "bic": bic,
            "stage1_success": result1.success,
            "stage2_success": result2.success,
        }

        logger.info("alams_var.fit.complete", **diagnostics)
        return diagnostics

    def _initialize_params(self, returns: np.ndarray) -> None:
        """Initialize regime parameters via volatility quantiles."""
        K = self.K
        T = len(returns)

        # Sort absolute returns to define volatility regimes
        rolling_vol = np.array(
            [np.std(returns[max(0, i - 20) : i + 1]) for i in range(T)]
        )
        quantiles = np.linspace(0, 1, K + 1)[1:-1]
        thresholds = np.quantile(rolling_vol, quantiles)

        # Assign each observation to a regime based on local volatility
        assignments = np.digitize(rolling_vol, thresholds)

        for k in range(K):
            mask = assignments == k
            if mask.sum() > 1:
                self.mu[k] = np.mean(returns[mask])
                self.sigma[k] = max(np.std(returns[mask]), 1e-6)
            else:
                self.mu[k] = np.mean(returns)
                self.sigma[k] = np.std(returns) * (0.5 + k * 0.5)

        # Ensure sigmas are monotonically increasing
        self.sigma = np.sort(self.sigma)

        # Initialize transition matrix: high persistence on diagonal
        self.P_base = np.eye(K) * 0.85
        for i in range(K):
            off_diag = 0.15 / (K - 1)
            for j in range(K):
                if i != j:
                    self.P_base[i, j] = off_diag

    def _estimate_stage1(self, returns: np.ndarray) -> optimize.OptimizeResult:
        """Stage 1: Estimate regime means, variances, and base transition matrix."""
        K = self.K

        # Pack parameters: [mu_0..mu_K-1, log_sigma_0..log_sigma_K-1, P_logit_flat]
        # Transition matrix is parameterized via softmax rows
        x0 = self._pack_stage1()

        def neg_log_likelihood(x: np.ndarray) -> float:
            self._unpack_stage1(x)
            _, ll = self._hamilton_filter(returns)
            reg = self.config.regularization * np.sum(x**2)
            return -(ll - reg)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = optimize.minimize(
                neg_log_likelihood,
                x0,
                method="L-BFGS-B",
                options={"maxiter": self.config.max_iter, "ftol": self.config.tol},
            )

        self._unpack_stage1(result.x)
        # Enforce sigma ordering after optimization
        order = np.argsort(self.sigma)
        self.sigma = self.sigma[order]
        self.mu = self.mu[order]
        self.P_base = self.P_base[order][:, order]

        return result

    def _estimate_stage2(self, returns: np.ndarray) -> optimize.OptimizeResult:
        """Stage 2: Estimate asymmetry parameter δ with fixed mu/sigma."""
        x0 = np.array([self.delta])

        def neg_log_likelihood(x: np.ndarray) -> float:
            self.delta = float(np.clip(x[0], 0.0, 0.5))
            _, ll = self._hamilton_filter(returns)
            return -ll

        result = optimize.minimize(
            neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=[(0.001, 0.5)],
            options={"maxiter": 50, "ftol": self.config.tol},
        )

        self.delta = float(np.clip(result.x[0], 0.001, 0.5))
        return result

    def _pack_stage1(self) -> np.ndarray:
        """Pack model params into optimization vector."""
        K = self.K
        parts = [
            self.mu,
            np.log(np.clip(self.sigma, 1e-8, None)),
            self._transition_to_logit(self.P_base).ravel(),
        ]
        return np.concatenate(parts)

    def _unpack_stage1(self, x: np.ndarray) -> None:
        """Unpack optimization vector into model params."""
        K = self.K
        idx = 0
        self.mu = x[idx : idx + K].copy()
        idx += K
        self.sigma = np.exp(x[idx : idx + K])
        self.sigma = np.clip(self.sigma, 1e-8, 10.0)
        idx += K
        logits = x[idx:].reshape(K, K)
        self.P_base = self._logit_to_transition(logits)

    @staticmethod
    def _transition_to_logit(P: np.ndarray) -> np.ndarray:
        """Convert transition matrix to unconstrained logit space."""
        return np.log(np.clip(P, 1e-10, None))

    @staticmethod
    def _logit_to_transition(logits: np.ndarray) -> np.ndarray:
        """Convert logits back to valid transition matrix (rows sum to 1)."""
        # Softmax per row
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # ============= HAMILTON FILTER =============

    def _get_transition_matrix(self, r_prev: float) -> np.ndarray:
        """
        Get asymmetric transition matrix conditioned on previous return sign.

        When r_prev < 0, increase probability of transitioning to higher
        volatility regimes by δ. This captures the empirical finding that
        negative returns make high-volatility states more likely.
        """
        K = self.K
        P = self.P_base.copy()

        if r_prev < 0:
            # Shift probability mass toward higher regimes
            for i in range(K):
                shift = np.zeros(K)
                for j in range(K):
                    # Higher regimes get positive shift, lower get negative
                    direction = (j - i) / max(K - 1, 1)
                    shift[j] = self.delta * direction

                P[i] = P[i] + shift
                P[i] = np.clip(P[i], 1e-10, None)
                P[i] /= P[i].sum()

        return P

    def _regime_density(self, r: float) -> np.ndarray:
        """
        Compute P(r_t | S_t=k) for each regime k.
        Each regime is a Gaussian with mean μ_k and std σ_k.
        """
        densities = np.zeros(self.K)
        for k in range(self.K):
            densities[k] = stats.norm.pdf(r, loc=self.mu[k], scale=self.sigma[k])
        return densities

    def _hamilton_filter(
        self, returns: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Hamilton (1989) filter: forward recursion for regime probabilities.

        Returns:
            (filtered_probs, log_likelihood)
            filtered_probs: (T, K) array of P(S_t=k | Ω_t)
            log_likelihood: scalar log-likelihood of the data
        """
        T = len(returns)
        K = self.K

        # Storage
        filtered = np.zeros((T, K))
        log_lik = 0.0

        # Initial state: ergodic distribution of base transition matrix
        xi = self._ergodic_distribution(self.P_base)

        for t in range(T):
            # Prediction step: P(S_t=k | Ω_{t-1})
            if t == 0:
                predicted = xi
            else:
                P_t = self._get_transition_matrix(returns[t - 1])
                predicted = filtered[t - 1] @ P_t

            predicted = np.clip(predicted, 1e-10, None)
            predicted /= predicted.sum()

            # Update step: P(S_t=k | Ω_t) ∝ P(r_t|S_t=k) * P(S_t=k|Ω_{t-1})
            densities = self._regime_density(returns[t])
            joint = densities * predicted
            marginal = joint.sum()

            if marginal < 1e-300:
                filtered[t] = predicted
            else:
                filtered[t] = joint / marginal
                log_lik += np.log(marginal)

        return filtered, log_lik

    @staticmethod
    def _ergodic_distribution(P: np.ndarray) -> np.ndarray:
        """Compute stationary distribution of transition matrix."""
        K = P.shape[0]
        A = np.vstack([P.T - np.eye(K), np.ones(K)])
        b = np.zeros(K + 1)
        b[-1] = 1.0
        try:
            result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            result = np.clip(result, 1e-10, None)
            return result / result.sum()
        except np.linalg.LinAlgError:
            return np.ones(K) / K

    # ============= FILTERING (ONLINE) =============

    def filter(self, returns: np.ndarray) -> np.ndarray:
        """
        Run Hamilton filter on new data using fitted parameters.

        Args:
            returns: Array of log returns

        Returns:
            (T, K) array of filtered regime probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before filtering")

        self.filtered_probs, _ = self._hamilton_filter(np.asarray(returns))
        return self.filtered_probs

    def get_regime_probabilities(self) -> np.ndarray:
        """Return current regime probabilities (last time step)."""
        if self.filtered_probs is None:
            raise RuntimeError("No filtered probabilities available. Call fit() or filter() first.")
        return self.filtered_probs[-1].copy()

    def get_current_regime(self) -> int:
        """Return most likely current regime index (0 = lowest vol, K-1 = highest)."""
        probs = self.get_regime_probabilities()
        return int(np.argmax(probs))

    # ============= VAR CALCULATION =============

    def calculate_var(
        self,
        confidence: float = 0.95,
        returns: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculate regime-weighted VaR (without liquidity adjustment).

        VaR_α = Σ_k P(S_t=k) · [μ_k + σ_k · Φ^{-1}(α)]

        Args:
            confidence: VaR confidence level (e.g. 0.95 for 95% VaR)
            returns: Optional new returns to filter first

        Returns:
            VaR as a positive number (loss magnitude at given confidence)
        """
        if returns is not None:
            self.filter(returns)

        probs = self.get_regime_probabilities()
        quantile = stats.norm.ppf(1 - confidence)

        var = 0.0
        for k in range(self.K):
            regime_var = self.mu[k] + self.sigma[k] * quantile
            var += probs[k] * regime_var

        # VaR is reported as positive loss
        return -var

    def calculate_liquidity_adjusted_var(
        self,
        confidence: float = 0.95,
        trade_size_usd: float = 0.0,
        pool_depth_usd: float = 1e9,
        returns: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """
        Full A-LAMS-VaR: regime VaR + liquidity adjustment.

        VaR_α(t) = Σ_k P(S_t=k) · [μ_k + σ_k · Φ^{-1}(α)] + λ(k) · Slippage(k)

        Args:
            confidence: VaR confidence level
            trade_size_usd: Planned trade size in USD
            pool_depth_usd: AMM pool depth in USD
            returns: Optional new returns to filter first

        Returns:
            Dict with var_pure, slippage_component, var_total, regime, regime_probs
        """
        if returns is not None:
            self.filter(returns)

        probs = self.get_regime_probabilities()
        pure_var = self.calculate_var(confidence)

        # Liquidity adjustment
        slippage = self._estimate_slippage(trade_size_usd, pool_depth_usd, probs)

        return {
            "var_pure": pure_var,
            "slippage_component": slippage,
            "var_total": pure_var + slippage,
            "confidence": confidence,
            "current_regime": self.get_current_regime(),
            "regime_probs": probs.tolist(),
            "delta": self.delta,
            "regime_means": self.mu.tolist(),
            "regime_sigmas": self.sigma.tolist(),
        }

    def _estimate_slippage(
        self,
        trade_size_usd: float,
        pool_depth_usd: float,
        regime_probs: np.ndarray,
    ) -> float:
        """
        Estimate regime-weighted AMM slippage.

        Slippage scales with:
        - Trade size relative to pool depth (non-linear, exponent > 1)
        - Current regime (high-vol = worse liquidity = more slippage)
        """
        if trade_size_usd <= 0 or pool_depth_usd <= 0:
            return 0.0

        lc = self.liquidity_config
        size_ratio = trade_size_usd / pool_depth_usd

        # Non-linear price impact: slippage ~ (size/depth)^exponent
        base_impact = size_ratio**lc.impact_exponent

        # Regime-weighted slippage
        slippage_bps = 0.0
        for k in range(self.K):
            multiplier = (
                lc.regime_multipliers[k]
                if k < len(lc.regime_multipliers)
                else lc.regime_multipliers[-1]
            )
            regime_slippage = lc.base_slippage_bps * multiplier * (1 + base_impact * 100)
            slippage_bps += regime_probs[k] * regime_slippage

        slippage_bps = min(slippage_bps, lc.max_slippage_bps)

        # Convert bps to decimal
        return slippage_bps / 10000.0

    # ============= DIAGNOSTICS =============

    def summary(self) -> dict:
        """Return model summary for logging/reporting."""
        if not self.is_fitted:
            return {"is_fitted": False}

        probs = self.get_regime_probabilities()
        return {
            "is_fitted": True,
            "n_regimes": self.K,
            "n_obs": self.n_obs,
            "delta": round(self.delta, 4),
            "log_likelihood": round(self.log_likelihood, 2),
            "regime_means": [round(m, 6) for m in self.mu],
            "regime_sigmas": [round(s, 6) for s in self.sigma],
            "current_regime": self.get_current_regime(),
            "regime_probs": [round(p, 4) for p in probs],
            "var_95": round(self.calculate_var(0.95), 6),
            "var_99": round(self.calculate_var(0.99), 6),
        }
