"""
Copula-based portfolio VaR with regime-conditional tail dependence.

Implements Clayton, Gumbel, Frank, Gaussian, and Student-t copulas
for multi-asset dependence modeling. Replaces Gaussian correlation
assumption in portfolio_var.py with realistic tail co-movements.

Mathematics:
- Copula C(u1,...,ud) links marginal CDFs to joint distribution
- Clayton: C(u,v) = (u^{-θ} + v^{-θ} - 1)^{-1/θ}, θ > 0
  → Lower tail dependence λ_L = 2^{-1/θ}
- Gumbel: C(u,v) = exp(-[(-ln u)^θ + (-ln v)^θ]^{1/θ}), θ ≥ 1
  → Upper tail dependence λ_U = 2 - 2^{1/θ}
- Frank: C(u,v) = -1/θ * ln(1 + (e^{-θu}-1)(e^{-θv}-1)/(e^{-θ}-1))
  → No tail dependence (symmetric, light tails)
- Gaussian: C(u,v) = Φ_R(Φ^{-1}(u), Φ^{-1}(v))
  → No tail dependence
- Student-t: C(u,v) = t_{ν,R}(t_ν^{-1}(u), t_ν^{-1}(v))
  → Symmetric tail dependence λ = 2*t_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ)))
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import kendalltau, multivariate_normal, norm, t as student_t

logger = logging.getLogger(__name__)

COPULA_FAMILIES = ("gaussian", "student_t", "clayton", "gumbel", "frank")


def _to_uniform(data: np.ndarray) -> np.ndarray:
    """Convert data to pseudo-uniform margins using empirical CDF (rank transform)."""
    n, d = data.shape
    u = np.zeros_like(data)
    for j in range(d):
        ranks = np.argsort(np.argsort(data[:, j])) + 1
        u[:, j] = ranks / (n + 1)  # Weibull plotting position
    return u


def _kendall_tau_matrix(u: np.ndarray) -> np.ndarray:
    """Compute Kendall's tau correlation matrix from uniform margins."""
    d = u.shape[1]
    tau = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            tau_ij, _ = kendalltau(u[:, i], u[:, j])
            tau[i, j] = tau[j, i] = tau_ij
    return tau


def _tau_to_pearson(tau: np.ndarray) -> np.ndarray:
    """Convert Kendall's tau to Pearson correlation (for Gaussian/Student-t copulas)."""
    return np.sin(np.pi / 2 * tau)


def _ensure_positive_definite(R: np.ndarray) -> np.ndarray:
    """Nearest positive-definite correlation matrix via eigenvalue clipping."""
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, 1e-8)
    R_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(R_pd))
    R_pd = R_pd / np.outer(d, d)
    np.fill_diagonal(R_pd, 1.0)
    return R_pd


# --- Log-likelihood functions for each copula family ---

def _ll_gaussian(u: np.ndarray, R: np.ndarray) -> float:
    """Gaussian copula log-likelihood."""
    d = u.shape[1]
    z = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
    R_inv = np.linalg.inv(R)
    det_R = np.linalg.det(R)
    if det_R <= 0:
        return -1e15
    ll = -0.5 * np.log(det_R)
    diff = z @ (R_inv - np.eye(d))
    ll_per_obs = -0.5 * np.sum(z * diff, axis=1)
    return float(np.sum(ll_per_obs))


def _ll_student_t(u: np.ndarray, R: np.ndarray, nu: float) -> float:
    """Student-t copula log-likelihood."""
    n, d = u.shape
    z = student_t.ppf(np.clip(u, 1e-10, 1 - 1e-10), df=nu)
    R_inv = np.linalg.inv(R)
    det_R = np.linalg.det(R)
    if det_R <= 0:
        return -1e15

    ll = 0.0
    ll += n * (gammaln((nu + d) / 2) - gammaln(nu / 2)
               - (d - 1) * gammaln((nu + 1) / 2)
               + (d - 1) * gammaln(nu / 2))
    ll -= 0.5 * n * np.log(det_R)

    quad = np.sum(z * (z @ R_inv), axis=1)
    ll += np.sum(-(nu + d) / 2 * np.log(1 + quad / nu))
    ll -= np.sum(-(nu + 1) / 2 * np.log(1 + z**2 / nu))

    return float(ll)


def _ll_clayton_bivariate(u1: np.ndarray, u2: np.ndarray, theta: float) -> float:
    """Clayton copula log-likelihood (bivariate)."""
    if theta <= 0:
        return -1e15
    u1c = np.clip(u1, 1e-10, 1 - 1e-10)
    u2c = np.clip(u2, 1e-10, 1 - 1e-10)
    s = u1c**(-theta) + u2c**(-theta) - 1
    s = np.maximum(s, 1e-10)
    ll = np.sum(
        np.log(1 + theta)
        - (1 + theta) * np.log(u1c)
        - (1 + theta) * np.log(u2c)
        - (2 + 1 / theta) * np.log(s)
    )
    return float(ll)


def _ll_gumbel_bivariate(u1: np.ndarray, u2: np.ndarray, theta: float) -> float:
    """Gumbel copula log-likelihood (bivariate)."""
    if theta < 1:
        return -1e15
    u1c = np.clip(u1, 1e-10, 1 - 1e-10)
    u2c = np.clip(u2, 1e-10, 1 - 1e-10)
    lu1 = -np.log(u1c)
    lu2 = -np.log(u2c)
    A = (lu1**theta + lu2**theta)**(1 / theta)
    C = np.exp(-A)
    # log density
    log_c = (np.log(C) + np.log(A + theta - 1)
             + (theta - 1) * (np.log(lu1) + np.log(lu2))
             - lu1 - lu2
             - (2 - 1 / theta) * np.log(lu1**theta + lu2**theta))
    return float(np.sum(log_c))


def _ll_frank_bivariate(u1: np.ndarray, u2: np.ndarray, theta: float) -> float:
    """Frank copula log-likelihood (bivariate)."""
    if abs(theta) < 1e-10:
        return -1e15
    u1c = np.clip(u1, 1e-10, 1 - 1e-10)
    u2c = np.clip(u2, 1e-10, 1 - 1e-10)
    et = np.exp(-theta)
    eu1 = np.exp(-theta * u1c)
    eu2 = np.exp(-theta * u2c)
    numer = -theta * (et - 1) * np.exp(-theta * (u1c + u2c))
    denom = ((et - 1) + (eu1 - 1) * (eu2 - 1))**2
    denom = np.maximum(denom, 1e-30)
    ll = np.sum(np.log(np.maximum(numer / denom, 1e-30)))
    return float(ll)


def _tail_dependence(family: str, params: dict) -> dict:
    """Compute lower and upper tail dependence coefficients."""
    if family == "clayton":
        theta = params["theta"]
        return {"lambda_lower": 2**(-1 / theta), "lambda_upper": 0.0}
    elif family == "gumbel":
        theta = params["theta"]
        return {"lambda_lower": 0.0, "lambda_upper": 2 - 2**(1 / theta)}
    elif family == "student_t":
        nu = params["nu"]
        R = params["R"]
        d = R.shape[0]
        # Average pairwise tail dependence
        lambdas = []
        for i in range(d):
            for j in range(i + 1, d):
                rho = R[i, j]
                arg = np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
                lam = 2 * student_t.cdf(-arg, df=nu + 1)
                lambdas.append(lam)
        avg_lam = float(np.mean(lambdas)) if lambdas else 0.0
        return {"lambda_lower": avg_lam, "lambda_upper": avg_lam}
    else:
        return {"lambda_lower": 0.0, "lambda_upper": 0.0}


def fit_copula(
    returns: np.ndarray | pd.DataFrame,
    family: str = "gaussian",
) -> dict:
    """
    Fit a copula to multivariate return data using IFM (Inference Functions for Margins).

    Args:
        returns: (T, d) array or DataFrame of returns.
        family: One of 'gaussian', 'student_t', 'clayton', 'gumbel', 'frank'.

    Returns:
        Dict with family, parameters, log_likelihood, aic, bic, tail_dependence.
    """
    if family not in COPULA_FAMILIES:
        raise ValueError(f"Unknown copula family '{family}'. Choose from {COPULA_FAMILIES}")

    if isinstance(returns, pd.DataFrame):
        returns = returns.values
    n, d = returns.shape
    u = _to_uniform(returns)

    if family == "gaussian":
        tau = _kendall_tau_matrix(u)
        R = _ensure_positive_definite(_tau_to_pearson(tau))
        ll = _ll_gaussian(u, R)
        n_params = d * (d - 1) // 2
        params = {"R": R}

    elif family == "student_t":
        tau = _kendall_tau_matrix(u)
        R = _ensure_positive_definite(_tau_to_pearson(tau))

        def neg_ll_nu(log_nu):
            nu = np.exp(log_nu) + 2.01
            return -_ll_student_t(u, R, nu)

        result = minimize(neg_ll_nu, x0=np.log(5.0), method="Nelder-Mead",
                          options={"maxiter": 200})
        nu = float(np.exp(result.x[0]) + 2.01)
        ll = _ll_student_t(u, R, nu)
        n_params = d * (d - 1) // 2 + 1
        params = {"R": R, "nu": nu}

    elif family in ("clayton", "gumbel", "frank"):
        # For d > 2, fit pairwise and average (nested Archimedean approximation)
        pair_thetas = []
        pair_lls = []
        for i in range(d):
            for j in range(i + 1, d):
                u_i, u_j = u[:, i], u[:, j]
                if family == "clayton":
                    def neg_ll(log_th):
                        return -_ll_clayton_bivariate(u_i, u_j, np.exp(log_th))
                    res = minimize(neg_ll, x0=np.log(1.0), method="Nelder-Mead")
                    theta = float(np.exp(res.x[0]))
                    pair_lls.append(-res.fun)
                elif family == "gumbel":
                    def neg_ll(log_th_m1):
                        return -_ll_gumbel_bivariate(u_i, u_j, np.exp(log_th_m1) + 1.0)
                    res = minimize(neg_ll, x0=np.log(0.5), method="Nelder-Mead")
                    theta = float(np.exp(res.x[0]) + 1.0)
                    pair_lls.append(-res.fun)
                else:  # frank
                    def neg_ll(th):
                        return -_ll_frank_bivariate(u_i, u_j, th[0])
                    res = minimize(neg_ll, x0=[2.0], method="Nelder-Mead")
                    theta = float(res.x[0])
                    pair_lls.append(-res.fun)
                pair_thetas.append(theta)

        theta = float(np.mean(pair_thetas))
        ll = float(np.sum(pair_lls))
        n_params = 1
        params = {"theta": theta}
    else:
        raise ValueError(f"Unknown family: {family}")

    aic = 2 * n_params - 2 * ll
    bic = n_params * np.log(n) - 2 * ll
    tail_dep = _tail_dependence(family, params)

    # Serialize R matrix for JSON compatibility
    serializable_params = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            serializable_params[k] = v.tolist()
        else:
            serializable_params[k] = v

    return {
        "family": family,
        "params": serializable_params,
        "log_likelihood": ll,
        "aic": aic,
        "bic": bic,
        "n_obs": n,
        "n_assets": d,
        "n_params": n_params,
        "tail_dependence": tail_dep,
    }



def _sample_copula(family: str, params: dict, n_samples: int, d: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate uniform samples from a fitted copula via simulation."""
    if family == "gaussian":
        R = np.array(params["R"]) if isinstance(params["R"], list) else params["R"]
        z = rng.multivariate_normal(np.zeros(d), R, size=n_samples)
        return norm.cdf(z)
    elif family == "student_t":
        R = np.array(params["R"]) if isinstance(params["R"], list) else params["R"]
        nu = params["nu"]
        z = rng.multivariate_normal(np.zeros(d), R, size=n_samples)
        chi2 = rng.chisquare(nu, size=(n_samples, 1))
        t_samples = z / np.sqrt(chi2 / nu)
        return student_t.cdf(t_samples, df=nu)
    elif family == "clayton":
        theta = params["theta"]
        # Marshall-Olkin algorithm for bivariate, extend pairwise for d > 2
        u = np.zeros((n_samples, d))
        u[:, 0] = rng.uniform(size=n_samples)
        for j in range(1, d):
            v = rng.uniform(size=n_samples)
            u[:, j] = (u[:, 0]**(-theta) * (v**(-theta / (1 + theta)) - 1) + 1)**(-1 / theta)
            u[:, j] = np.clip(u[:, j], 1e-10, 1 - 1e-10)
        return u
    elif family == "gumbel":
        theta = params["theta"]
        # Stable subordinator method
        from scipy.stats import levy_stable
        u = np.zeros((n_samples, d))
        alpha_s = 1.0 / theta
        s = levy_stable.rvs(alpha_s, 1.0, size=n_samples, random_state=rng)
        s = np.maximum(s, 1e-10)
        for j in range(d):
            e = rng.exponential(size=n_samples)
            u[:, j] = np.exp(-(e / s)**alpha_s)
            u[:, j] = np.clip(u[:, j], 1e-10, 1 - 1e-10)
        return u
    else:  # frank — conditional inversion
        theta = params["theta"]
        u = np.zeros((n_samples, d))
        u[:, 0] = rng.uniform(size=n_samples)
        for j in range(1, d):
            v = rng.uniform(size=n_samples)
            et = np.exp(-theta)
            eu = np.exp(-theta * u[:, 0])
            u[:, j] = -np.log(1 + v * (eu - 1) / (v * (eu - 1) - (et - 1))) / theta
            u[:, j] = np.clip(u[:, j], 1e-10, 1 - 1e-10)
        return u


def copula_portfolio_var(
    model: dict,
    weights: dict[str, float],
    copula_fit: dict,
    alpha: float = 0.05,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Portfolio VaR using copula-based Monte Carlo simulation.

    Instead of assuming multivariate normal, simulates joint returns
    using the fitted copula and per-asset MSM marginal distributions.

    Args:
        model: Output of calibrate_multivariate() from portfolio_var.py.
        weights: Asset weights dict.
        copula_fit: Output of fit_copula().
        alpha: VaR confidence level.
        n_simulations: Number of Monte Carlo draws.
        seed: Random seed.

    Returns:
        Dict with copula_var, gaussian_var, var_ratio, and simulation details.
    """
    assets = model["assets"]
    d = len(assets)
    w = np.array([weights.get(a, 0.0) for a in assets])
    rng = np.random.RandomState(seed)

    # Simulate uniform samples from copula
    u_sim = _sample_copula(copula_fit["family"], copula_fit["params"], n_simulations, d, rng)

    # Convert uniform margins to return space using per-asset MSM marginals
    probs = model["current_probs"]
    returns_sim = np.zeros_like(u_sim)
    for i, asset in enumerate(assets):
        # Regime-weighted sigma for this asset
        sigma_i = sum(
            probs[k] * model["per_asset"][asset]["sigma_states"][k]
            for k in range(model["num_states"])
        )
        returns_sim[:, i] = norm.ppf(u_sim[:, i]) * sigma_i

    # Portfolio returns
    port_returns = returns_sim @ w
    copula_var = float(np.percentile(port_returns, alpha * 100))

    # Gaussian VaR for comparison
    from portfolio_var import portfolio_var as pvar_fn
    gauss_result = pvar_fn(model, weights, alpha=alpha)
    gaussian_var = gauss_result["portfolio_var"]

    var_ratio = copula_var / gaussian_var if abs(gaussian_var) > 1e-12 else 1.0

    return {
        "copula_var": copula_var,
        "gaussian_var": gaussian_var,
        "var_ratio": var_ratio,
        "copula_family": copula_fit["family"],
        "tail_dependence": copula_fit["tail_dependence"],
        "n_simulations": n_simulations,
        "alpha": alpha,
    }


def regime_conditional_copulas(
    model: dict,
    family: str = "student_t",
) -> list[dict]:
    """
    Fit separate copulas per MSM regime.

    Crisis regimes should show stronger tail dependence than calm regimes.

    Args:
        model: Output of calibrate_multivariate() from portfolio_var.py.
        family: Copula family to fit per regime.

    Returns:
        List of dicts (one per regime) with copula fit + regime info.
    """
    K = model["num_states"]
    returns_df = model["returns_df"]
    returns_arr = returns_df.values
    n, d = returns_arr.shape

    # Get average regime probabilities
    all_fp = np.zeros((n, K))
    for asset in model["assets"]:
        all_fp += model["per_asset"][asset]["filter_probs"].values
    avg_probs = all_fp / len(model["assets"])

    results = []
    for k in range(K):
        w_k = avg_probs[:, k]
        # Select observations where this regime has high probability
        threshold = np.percentile(w_k, 50)
        mask = w_k >= threshold
        n_obs = int(mask.sum())

        if n_obs < max(30, d + 5):
            # Not enough data — use full sample with regime weights
            regime_returns = returns_arr
        else:
            regime_returns = returns_arr[mask]

        try:
            fit = fit_copula(regime_returns, family=family)
        except Exception as e:
            logger.warning("Copula fit failed for regime %d: %s", k + 1, e)
            fit = fit_copula(returns_arr, family=family)

        results.append({
            "regime": k + 1,
            "n_obs": n_obs,
            "copula": fit,
        })

    return results


def compare_copulas(
    returns: np.ndarray | pd.DataFrame,
    families: list[str] | None = None,
) -> list[dict]:
    """
    Fit all copula families and rank by AIC/BIC.

    Args:
        returns: (T, d) array or DataFrame of returns.
        families: List of families to compare. Defaults to all 5.

    Returns:
        List of dicts sorted by AIC (best first), each with fit results.
    """
    if families is None:
        families = list(COPULA_FAMILIES)

    results = []
    for fam in families:
        try:
            fit = fit_copula(returns, family=fam)
            results.append(fit)
        except Exception as e:
            logger.warning("Failed to fit %s copula: %s", fam, e)

    results.sort(key=lambda r: r["aic"])

    for rank, r in enumerate(results):
        r["rank"] = rank + 1
        r["best"] = rank == 0

    return results