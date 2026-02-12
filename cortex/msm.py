# -*- coding: utf-8 -*-
"""
MSM-based volatility forecast + VaR(5%) + Kupiec & Christoffersen backtests
+ Tail probabilities (1, 3, 5 zile) condiționat pe regimul curent.
"""

import logging

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from scipy.stats import norm, chi2, t as student_t

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# 1. Helper: extragere prețuri din yfinance
# -------------------------------------------------------

def _extract_close(df: pd.DataFrame, symbol: str) -> pd.Series:
    """
    Extrage coloana de prețuri de închidere (Adj Close sau Close)
    din DataFrame-ul yfinance, indiferent dacă are MultiIndex sau nu.
    """
    if isinstance(df.columns, pd.MultiIndex):
        price_level = df.columns.get_level_values(0)
        target_level = 'Adj Close' if 'Adj Close' in price_level else 'Close'
        if target_level not in price_level:
            raise KeyError(
                "Neither 'Adj Close' nor 'Close' present in downloaded data."
                f" Available top-level columns: {sorted(set(price_level))}"
            )
        values = df.xs(target_level, axis=1, level=0)
        if isinstance(values, pd.Series):
            return values.dropna()
        if symbol in values.columns:
            return values[symbol].dropna()
        return values.iloc[:, 0].dropna()

    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    if price_col not in df.columns:
        raise KeyError(
            f"Missing both 'Adj Close' and 'Close' columns. Available columns: {list(df.columns)}"
        )
    return df[price_col].dropna()


# -------------------------------------------------------
# 2. Kupiec & Christoffersen
# -------------------------------------------------------

def kupiec_test(breach_series, alpha=0.05):
    """
    Kupiec Unconditional Coverage Test
    breach_series: 0/1 (1 = VaR breach)
    alpha: nivel VaR (ex: 0.05)
    """
    b = np.asarray(breach_series, dtype=int)
    n = len(b)
    x = b.sum()  # număr de breach-uri observate
    if n == 0:
        return np.nan, np.nan, x, n

    pi_hat = x / n        # frecvența empirică
    pi_0 = alpha          # frecvența teoretică

    if pi_hat in (0, 1):
        return np.nan, np.nan, x, n

    logL0 = (n - x) * np.log(1 - pi_0) + x * np.log(pi_0)
    logL1 = (n - x) * np.log(1 - pi_hat) + x * np.log(pi_hat)

    LR = -2 * (logL0 - logL1)
    p_value = chi2.sf(LR, df=1)

    return LR, p_value, x, n


def christoffersen_independence_test(breach_series):
    """
    Christoffersen Independence Test.
    Testează dacă breach-urile sunt independente în timp (nu vin în clustere).
    """
    b = np.asarray(breach_series, dtype=int)
    if len(b) < 2:
        return np.nan, np.nan, (0, 0, 0, 0)

    b_t   = b[:-1]
    b_tp1 = b[1:]

    n00 = np.sum((b_t == 0) & (b_tp1 == 0))
    n01 = np.sum((b_t == 0) & (b_tp1 == 1))
    n10 = np.sum((b_t == 1) & (b_tp1 == 0))
    n11 = np.sum((b_t == 1) & (b_tp1 == 1))

    n0_ = n00 + n01
    n1_ = n10 + n11
    n_1 = n01 + n11
    N   = n00 + n01 + n10 + n11

    if n0_ == 0 or n1_ == 0 or n_1 == 0 or N == 0:
        return np.nan, np.nan, (n00, n01, n10, n11)

    pi01 = n01 / n0_
    pi11 = n11 / n1_
    pi   = n_1 / N

    # H0: iid cu p = pi
    logL0 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)

    logL1 = 0.0
    if 0 < pi01 < 1:
        logL1 += n00 * np.log(1 - pi01) + n01 * np.log(pi01)
    if 0 < pi11 < 1:
        logL1 += n10 * np.log(1 - pi11) + n11 * np.log(pi11)

    LR = -2 * (logL0 - logL1)
    p_value = chi2.sf(LR, df=1)

    return LR, p_value, (n00, n01, n10, n11)


# -------------------------------------------------------
# 3. MSM: model simplu cu K stări + filter bayesian
# -------------------------------------------------------


def _build_transition_matrix(p_stay, K: int) -> np.ndarray:
    """Build K×K transition matrix from scalar or per-regime p_stay."""
    p_arr = np.atleast_1d(np.asarray(p_stay, dtype=float))
    if p_arr.size == 1:
        p_arr = np.full(K, p_arr[0])
    if p_arr.size != K:
        raise ValueError(f"p_stay must be scalar or length K={K}, got {p_arr.size}")
    P = np.zeros((K, K))
    for k in range(K):
        off_diag = (1.0 - p_arr[k]) / max(K - 1, 1)
        P[k, :] = off_diag
        P[k, k] = p_arr[k]
    return P


def msm_vol_forecast(
    returns: pd.Series,
    num_states: int = 5,
    sigma_low: float = 0.1,
    sigma_high: float = 1.0,
    p_stay=0.97,
):
    """
    MSM Bayesian filter with K volatility states.

    p_stay: float or array-like of length K. If scalar, all states share
            the same persistence. If array, each state k has its own p_stay[k].

    Returns (sigma_forecast, sigma_filtered, filter_probs_df, sigma_states, P_matrix).
    """
    r = np.asarray(returns.values, dtype=float)
    n = len(r)
    K = num_states

    sigmas = np.exp(
        np.linspace(np.log(sigma_low), np.log(sigma_high), num_states)
    )

    P = _build_transition_matrix(p_stay, K)

    # prior inițial: uniform
    pi_t = np.full(K, 1.0 / K)

    # Storage pentru ambele tipuri de volatilitate
    sigma_forecast = np.zeros(n)  # σ_{t|t-1} - FORECAST (out-of-sample)
    sigma_filtered = np.zeros(n)  # σ_t - FILTERED (in-sample)
    filter_probs = np.zeros((n, K))
    
    eps = 1e-12

    for t in range(n):
        # 1) FORECAST: σ_{t|t-1} = E[σ | info_{t-1}]
        # Aceasta e volatilitatea prezisă ÎNAINTE de a vedea r_t
        sigma_forecast[t] = np.sum(pi_t * sigmas)
        
        # 2) UPDATE cu observația r_t (likelihood)
        like = (1.0 / (sigmas + eps)) * np.exp(
            -0.5 * (r[t] / (sigmas + eps)) ** 2
        )
        
        # 3) Actualizare posterior: π_t ∝ π_{t|t-1} * likelihood
        pi_unnorm = pi_t * like
        s = pi_unnorm.sum()
        if s > eps and np.isfinite(s):
            pi_t = pi_unnorm / s
        else:
            pi_t = np.full(K, 1.0 / K)
        
        # 4) FILTERED: σ_t = E[σ | info_t] (după ce vedem r_t)
        sigma_filtered[t] = np.sum(pi_t * sigmas)
        filter_probs[t, :] = pi_t
        
        # 5) PREDICT pentru următorul pas: π_{t+1|t} = π_t @ P
        pi_t = pi_t @ P

    sigma_forecast_series = pd.Series(sigma_forecast, index=returns.index, name="sigma_forecast")
    sigma_filtered_series = pd.Series(sigma_filtered, index=returns.index, name="sigma_filtered")
    filter_probs_df = pd.DataFrame(
        filter_probs,
        index=returns.index,
        columns=[f"state_{k+1}" for k in range(K)]
    )

    return sigma_forecast_series, sigma_filtered_series, filter_probs_df, sigmas, P


# -------------------------------------------------------
# 3b. CALIBRARE AVANSATĂ MSM - Maximum Likelihood Estimation
# -------------------------------------------------------

def msm_log_likelihood(params, returns, num_states=5):
    """
    Negative log-likelihood for MSM.

    params: [sigma_low, sigma_high, p_stay_1, ..., p_stay_K]
            OR [sigma_low, sigma_high, p_stay] (scalar — backward compat).
    """
    if len(params) == 3:
        sigma_low, sigma_high, p_stay = params
        p_stay_arr = np.full(num_states, p_stay)
    elif len(params) == num_states + 2:
        sigma_low, sigma_high = params[0], params[1]
        p_stay_arr = np.array(params[2:])
    else:
        return 1e10

    if sigma_low <= 0 or sigma_high <= sigma_low:
        return 1e10
    if np.any(p_stay_arr <= 0) or np.any(p_stay_arr >= 1):
        return 1e10

    r = np.asarray(returns, dtype=float)
    n = len(r)
    K = num_states

    sigmas = np.exp(np.linspace(np.log(sigma_low), np.log(sigma_high), K))

    P = _build_transition_matrix(p_stay_arr, K)
    
    # Prior uniform
    pi_t = np.full(K, 1.0 / K)
    
    log_likelihood = 0.0
    eps = 1e-12
    
    for t in range(n):
        # Likelihood pentru observația curentă
        like = (1.0 / (sigmas + eps)) * np.exp(-0.5 * (r[t] / (sigmas + eps)) ** 2)
        
        # Probabilitatea marginală a observației
        p_obs = np.dot(pi_t, like)
        
        if p_obs > eps:
            log_likelihood += np.log(p_obs)
        else:
            log_likelihood += np.log(eps)
        
        # Update posterior
        pi_unnorm = pi_t * like
        s = pi_unnorm.sum()
        if s > eps:
            pi_t = pi_unnorm / s
        else:
            pi_t = np.full(K, 1.0 / K)
        
        # Predict next step
        pi_t = pi_t @ P
    
    return -log_likelihood  # Returnăm negativul pentru minimizare


def calibrate_msm_advanced(
    returns: pd.Series,
    num_states: int = 5,
    method: str = 'mle',
    target_var_breach: float = 0.05,
    verbose: bool = True
):
    """
    Calibrare avansată a parametrilor MSM.
    
    Metode disponibile:
    -------------------
    1. 'mle' - Maximum Likelihood Estimation (optimizare numerică)
    2. 'grid' - Grid search cu cross-validation
    3. 'empirical' - Bazat pe quantile empirice + optimizare VaR breach
    4. 'hybrid' - Combinație MLE + ajustare VaR breach
    
    Parametri:
    ----------
    returns : pd.Series - randamente în %
    num_states : int - numărul de stări MSM
    method : str - metoda de calibrare ('mle', 'grid', 'empirical', 'hybrid')
    target_var_breach : float - rata țintă de VaR breach (default 5%)
    verbose : bool - afișează progresul
    
    Returnează:
    -----------
    dict cu parametrii calibrați și metrici de calitate
    """
    from scipy.optimize import minimize, differential_evolution
    
    r = np.asarray(returns.values, dtype=float)
    std_r = np.std(r)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"   MSM ADVANCED CALIBRATION - Method: {method.upper()}")
        print(f"{'='*60}")
        print(f"   Returns: {len(r)} observations")
        print(f"   Empirical std: {std_r:.3f}%")
        print(f"   Target VaR breach: {target_var_breach*100:.1f}%")
    
    # =========================================================================
    # METODA 1: Maximum Likelihood Estimation
    # =========================================================================
    K = num_states

    if method == 'mle':
        # K+2 params: [sigma_low, sigma_high, p_stay_1, ..., p_stay_K]
        bounds = [
            (std_r * 0.1, std_r * 0.8),
            (std_r * 1.5, std_r * 5.0),
        ] + [(0.90, 0.995)] * K

        best_result = None
        best_ll = float('inf')

        base_ps = [0.95, 0.97, 0.93, 0.96]
        start_points = [
            [std_r * 0.3, std_r * 2.5] + [ps] * K
            for ps in base_ps
        ]

        if verbose:
            print(f"\n   Running MLE optimization ({K+2} params, {len(start_points)} starts)...")

        for i, x0 in enumerate(start_points):
            try:
                result = minimize(
                    msm_log_likelihood,
                    x0=x0,
                    args=(r, num_states),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'disp': False}
                )
                if result.fun < best_ll:
                    best_ll = result.fun
                    best_result = result
            except Exception as e:
                if verbose:
                    print(f"      Start point {i+1} failed: {e}")

        if best_result is not None:
            sigma_low = best_result.x[0]
            sigma_high = best_result.x[1]
            p_stay = best_result.x[2:].tolist()
            if verbose:
                print(f"\n   ✓ MLE Converged!")
                print(f"     Log-likelihood: {-best_ll:.2f}")
                print(f"     p_stay per regime: {[f'{p:.4f}' for p in p_stay]}")
        else:
            sigma_low = std_r * 0.35
            sigma_high = std_r * 3.0
            p_stay = [0.97] * K
            if verbose:
                print(f"\n   ⚠ MLE failed, using empirical fallback")
    
    # =========================================================================
    # METODA 2: Grid Search cu evaluare VaR
    # =========================================================================
    elif method == 'grid':
        if verbose:
            print(f"\n   Running grid search...")
        
        # Grid de parametri
        sigma_low_grid = np.array([0.2, 0.3, 0.4, 0.5]) * std_r
        sigma_high_grid = np.array([2.0, 2.5, 3.0, 3.5, 4.0]) * std_r
        p_stay_grid = [0.93, 0.95, 0.97, 0.98]
        
        best_params = None
        best_score = float('inf')
        
        total_combos = len(sigma_low_grid) * len(sigma_high_grid) * len(p_stay_grid)
        combo_count = 0
        
        for sl in sigma_low_grid:
            for sh in sigma_high_grid:
                for ps in p_stay_grid:
                    combo_count += 1
                    
                    # Run MSM - folosim sigma_forecast (out-of-sample)
                    sigma_forecast, _, _, _, _ = msm_vol_forecast(
                        returns, num_states, sl, sh, ps
                    )
                    
                    # Calculate VaR breach rate (OUT-OF-SAMPLE corect)
                    z_alpha = norm.ppf(target_var_breach)
                    var_5 = z_alpha * sigma_forecast
                    breaches = (returns < var_5).mean()
                    
                    # Score: distanța de la target + penalizare pentru volatilitate prea mică
                    breach_error = abs(breaches - target_var_breach)
                    vol_correlation = np.corrcoef(np.abs(r), sigma_forecast.values)[0, 1]
                    
                    # Score combinat: breach error + (1 - correlation)
                    score = breach_error + 0.1 * (1 - vol_correlation)
                    
                    if score < best_score:
                        best_score = score
                        best_params = (sl, sh, ps)
                        best_breach = breaches
                        best_corr = vol_correlation
        
        sigma_low, sigma_high, p_stay = best_params
        
        if verbose:
            print(f"   ✓ Grid search complete ({total_combos} combinations)")
            print(f"     Best breach rate: {best_breach*100:.2f}%")
            print(f"     Best correlation: {best_corr:.3f}")
    
    # =========================================================================
    # METODA 3: Empirical (bazat pe quantile)
    # =========================================================================
    elif method == 'empirical':
        if verbose:
            print(f"\n   Using empirical quantile-based calibration...")
        
        # Sigma states bazate pe quantile empirice ale |r|
        abs_r = np.abs(r)
        
        # Quantile pentru stări: 10%, 30%, 50%, 70%, 90%
        quantiles = [0.10, 0.30, 0.50, 0.70, 0.90]
        sigma_quantiles = np.quantile(abs_r, quantiles)
        
        sigma_low = sigma_quantiles[0]   # 10th percentile
        sigma_high = sigma_quantiles[-1]  # 90th percentile
        
        # p_stay calibrat pentru persistență empirică
        # Estimăm din autocorrelația |r|
        autocorr = pd.Series(abs_r).autocorr(lag=1)
        p_stay = max(0.90, min(0.99, 0.5 + 0.5 * autocorr))
        
        if verbose:
            print(f"     Abs return quantiles: {sigma_quantiles}")
            print(f"     Autocorrelation |r|: {autocorr:.3f}")
    
    # =========================================================================
    # METODA 4: Hybrid (MLE + ajustare VaR ITERATIVĂ)
    # =========================================================================
    elif method == 'hybrid':
        if verbose:
            print(f"\n   Running hybrid calibration (MLE + iterative VaR adjustment)...")
        
        # Pas 1: MLE pentru parametri inițiali
        bounds = [
            (std_r * 0.1, std_r * 0.8),
            (std_r * 1.5, std_r * 5.0),
            (0.90, 0.995)
        ]
        
        x0 = [std_r * 0.35, std_r * 3.0, 0.97]
        
        try:
            result = minimize(
                msm_log_likelihood,
                x0=x0,
                args=(r, num_states),
                method='L-BFGS-B',
                bounds=bounds
            )
            sigma_low, sigma_high, p_stay = result.x
            
            if verbose:
                print(f"     MLE step: σ_low={sigma_low:.3f}, σ_high={sigma_high:.3f}, p_stay={p_stay:.3f}")
        except Exception as e:
            logger.warning("MLE optimization failed, using empirical fallback: %s", e)
            sigma_low, sigma_high, p_stay = std_r * 0.35, std_r * 3.0, 0.97
        
        # Pas 2: Ajustare ITERATIVĂ pentru VaR breach rate
        # Folosim bisection pentru a găsi factorul de scalare optim
        if verbose:
            print(f"     Fine-tuning for target breach rate {target_var_breach*100:.1f}%...")
        
        best_scale = 1.0
        best_breach_error = float('inf')
        
        # Căutare binară pentru factorul de scalare
        scale_low, scale_high = 0.5, 1.5
        
        for iteration in range(15):  # Max 15 iterații
            scale_mid = (scale_low + scale_high) / 2
            
            # Testăm cu acest factor de scalare
            test_sigma_low = sigma_low * scale_mid
            test_sigma_high = sigma_high * scale_mid
            
            sigma_forecast, _, _, _, _ = msm_vol_forecast(
                returns, num_states, test_sigma_low, test_sigma_high, p_stay
            )
            z_alpha = norm.ppf(target_var_breach)
            var_5 = z_alpha * sigma_forecast
            current_breach = (returns < var_5).mean()
            
            breach_error = abs(current_breach - target_var_breach)
            
            if breach_error < best_breach_error:
                best_breach_error = breach_error
                best_scale = scale_mid
            
            # Dacă breach e prea mic, reducem sigma (scale down)
            # Dacă breach e prea mare, creștem sigma (scale up)
            if current_breach < target_var_breach:
                scale_high = scale_mid  # Reducem sigma
            else:
                scale_low = scale_mid   # Creștem sigma
            
            # Stop dacă suntem suficient de aproape
            if breach_error < 0.002:  # 0.2% tolerance
                break
        
        # Aplicăm cel mai bun factor de scalare
        sigma_low *= best_scale
        sigma_high *= best_scale
        
        # Verificare finală
        sigma_forecast, _, _, _, _ = msm_vol_forecast(returns, num_states, sigma_low, sigma_high, p_stay)
        var_5 = z_alpha * sigma_forecast
        final_breach_check = (returns < var_5).mean()
        
        if verbose:
            print(f"     Best scale factor: {best_scale:.4f}")
            print(f"     Final breach rate: {final_breach_check*100:.2f}%")
    
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    # Normalize p_stay to list for all code paths
    if isinstance(p_stay, (int, float, np.floating)):
        p_stay = [float(p_stay)] * K
    elif isinstance(p_stay, np.ndarray):
        p_stay = p_stay.tolist()
    p_stay = [float(p) for p in p_stay]

    # =========================================================================
    # FINAL VALIDATION
    # =========================================================================
    sigma_forecast_final, sigma_filtered_final, filter_probs_final, sigmas_final, P_final = msm_vol_forecast(
        returns, num_states, sigma_low, sigma_high, p_stay
    )

    z_alpha = norm.ppf(target_var_breach)
    var_5 = z_alpha * sigma_forecast_final
    final_breach = (returns < var_5).mean()
    final_corr = np.corrcoef(np.abs(r), sigma_forecast_final.values)[0, 1]

    n = len(r)
    num_params = 2 + len(p_stay)
    ll_params = [sigma_low, sigma_high] + p_stay
    ll = -msm_log_likelihood(ll_params, r, num_states)
    aic = 2 * num_params - 2 * ll
    bic = num_params * np.log(n) - 2 * ll

    result = {
        'sigma_low': sigma_low,
        'sigma_high': sigma_high,
        'p_stay': p_stay,
        'num_states': num_states,
        'sigma_states': sigmas_final,
        'method': method,
        'metrics': {
            'var_breach_rate': final_breach,
            'vol_correlation': final_corr,
            'log_likelihood': ll,
            'aic': aic,
            'bic': bic
        }
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"   CALIBRATION RESULTS")
        print(f"{'='*60}")
        print(f"   σ_low:    {sigma_low:.4f}%")
        print(f"   σ_high:   {sigma_high:.4f}%")
        print(f"   p_stay:   {[f'{p:.4f}' for p in p_stay]}")
        print(f"   States:   {num_states}")
        print(f"\n   Sigma states: {np.round(sigmas_final, 3)}")
        print(f"\n   --- Quality Metrics ---")
        print(f"   VaR breach rate: {final_breach*100:.2f}% (target: {target_var_breach*100:.1f}%)")
        print(f"   Corr(|r|, σ):    {final_corr:.3f}")
        print(f"   Log-likelihood:  {ll:.2f}")
        print(f"   AIC: {aic:.2f}")
        print(f"   BIC: {bic:.2f}")
        print(f"{'='*60}\n")

    return result


# -------------------------------------------------------
# 4. Probabilități de tail bazate pe MSM
# -------------------------------------------------------

def msm_tail_probs(
    rets_series: pd.Series,
    msm_filter_probs,
    sigma_states,
    alpha: float = 0.05,
    horizons=(1, 3, 5),
    use_student_t: bool = False,
    nu: float = 5.0
):
    """
    Calculează probabilitatea ca randamentul să cadă sub L1 (empirical tail)
    în următoarele H zile, condiționat pe regimul curent MSM.

    Parametri:
    ----------
    rets_series      : pd.Series cu randamente zilnice (în %).
    msm_filter_probs : array-like (T, K) – probabilități filtrate pe stări.
                       Poate fi np.ndarray sau pd.DataFrame.
    sigma_states     : array-like (K,) – vol. zilnică în % pentru fiecare stare.
    alpha            : tail empiric (default 0.05 = 5%).
    horizons         : tuple de orizonturi (zile) pentru care vrem P(tail).
    use_student_t    : dacă True, folosește Student-t în loc de Normal.
    nu               : df pentru Student-t (dacă use_student_t=True). Trebuie > 2.

    Return:
    -------
    dict cu:
      "L1": pragul empiric
      "p1": P(tail într-o singură zi t+1)
      "horizon_probs": {H: P(at least one tail in H days)}
    """
    # 1) Prag empiric L1 (5% quantile)
    L1 = float(rets_series.quantile(alpha))

    # 2) Ultimul vector de probabilități MSM (condiționat pe info curentă)
    P_filtered = np.asarray(msm_filter_probs)
    P_last = P_filtered[-1]   # shape (K,)

    sigma_states = np.asarray(sigma_states, dtype=float)

    # 3) Probabilitate tail condiționată pe fiecare stare
    # r | state k ~ distribuție cu media 0 și deviația standard sigma_k
    
    if use_student_t:
        # Student-t scalată: dacă T ~ t_nu (standard), atunci
        # X = sigma * T * sqrt((nu-2)/nu) are Var(X) = sigma²
        # Echivalent: X ~ t_nu cu scale = sigma * sqrt((nu-2)/nu)
        # 
        # Pentru a calcula P(X < L1), folosim:
        # P(X < L1) = P(T < L1 / scale) = student_t.cdf(L1/scale, df=nu)
        
        if nu <= 2:
            raise ValueError("Student-t df (nu) must be > 2 for finite variance")
        
        # Factorul de scalare pentru a avea varianța = sigma²
        variance_adjustment = np.sqrt((nu - 2) / nu)
        scale = sigma_states * variance_adjustment
        
        z = (L1 - 0.0) / scale
        p_k = student_t.cdf(z, df=nu)
    else:
        # Normal: r | state k ~ N(0, sigma_k²)
        z = (L1 - 0.0) / sigma_states
        p_k = norm.cdf(z)

    # 4) Probabilitatea 1-day tail cond. pe regimul curent
    p1 = float(np.dot(P_last, p_k))

    # 5) Probabilități pe 1, 3, 5 zile:
    # P(at least one tail in H zile) ≈ 1 - (1 - p1)^H
    horizon_probs = {}
    for H in horizons:
        horizon_probs[H] = 1.0 - (1.0 - p1)**H

    return {
        "L1": L1,
        "p1": p1,
        "horizon_probs": horizon_probs,
        "distribution": f"Student-t(df={nu})" if use_student_t else "Normal"
    }



def msm_var_forecast_next_day(
    filter_probs, sigma_states, P_matrix,
    alpha=0.05, mu=0.0,
    use_student_t: bool = False, nu: float = 5.0,
    use_evt: bool = False, evt_params: dict | None = None,
):
    """
    VaR(alpha) forecast for the next day using the transition matrix.

    Distribution hierarchy (highest priority first):
      1. use_evt=True and alpha < 0.01 → EVT-GPD tail estimate
      2. use_student_t=True → Student-t quantile
      3. default → Normal quantile

    evt_params must contain: xi, beta, threshold, n_total, n_exceedances
    """
    if use_student_t:
        if nu <= 2:
            raise ValueError("Student-t df (nu) must be > 2 for finite variance")

    pi_t = np.asarray(filter_probs.iloc[-1] if hasattr(filter_probs, 'iloc') else filter_probs[-1])
    pi_t1_given_t = pi_t @ P_matrix
    sigma_t1_forecast = float(np.dot(pi_t1_given_t, sigma_states))

    # EVT for extreme tail (alpha < 0.01)
    if use_evt and evt_params is not None and alpha < 0.01:
        from extreme_value_theory import evt_var as _evt_var
        var_loss = _evt_var(
            xi=evt_params["xi"],
            beta=evt_params["beta"],
            threshold=evt_params["threshold"],
            n_total=evt_params["n_total"],
            n_exceedances=evt_params["n_exceedances"],
            alpha=alpha,
        )
        var_t1 = mu - var_loss  # convert positive loss to negative return
        z_alpha = var_t1 / sigma_t1_forecast if sigma_t1_forecast > 1e-12 else float("nan")
        return var_t1, sigma_t1_forecast, z_alpha, pi_t1_given_t

    if use_student_t:
        z_alpha = float(student_t.ppf(alpha, df=nu))
        var_t1 = mu + z_alpha * sigma_t1_forecast
    else:
        z_alpha = float(norm.ppf(alpha))
        var_t1 = mu + z_alpha * sigma_t1_forecast

    return var_t1, sigma_t1_forecast, z_alpha, pi_t1_given_t







# -------------------------------------------------------
# 5. MAIN
# -------------------------------------------------------

if __name__ == "__main__":
    # ----------------------------
    # 5.1. Download date și randamente
    # ----------------------------
    ticker = "Financial_asset"  # Schimbă simbolul după nevoie
    start = "Y-M-D"
    
    # ========================================
    # SETEAZĂ DATA PENTRU CARE VREI FORECAST
    # ========================================
    # Pentru forecast pe ziua X, ai nevoie de lumânarea închisă din ziua X-1
    # Exemplu: pentru forecast pe 12 decembrie, ai nevoie de date până în 11 decembrie
    FORECAST_DATE = "Y-M-D"  # Schimbă această dată după nevoie
    
    from datetime import datetime, timedelta
    forecast_dt = datetime.strptime(FORECAST_DATE, "%Y-%m-%d")
    
    # Descarcă date cu marjă pentru a ne asigura că avem tot ce trebuie
    end = (forecast_dt + timedelta(days=2)).strftime("%Y-%m-%d")

    data = yf.download(ticker, start=start, end=end, auto_adjust=False, group_by='column')
    close_raw = _extract_close(data, ticker)
    
    # IMPORTANT: Păstrăm doar lumânările ÎNCHISE (strict înainte de FORECAST_DATE)
    # Excludem lumânarea curentă care e încă în curs
    close = close_raw[close_raw.index < FORECAST_DATE]
    
    last_candle_date = close.index[-1]
    expected_last = forecast_dt - timedelta(days=1)
    
    print(f"\nDownloaded {ticker}: raw={len(close_raw)}, filtered={len(close)} prices")
    print(f"Last CLOSED candle: {last_candle_date.strftime('%Y-%m-%d')}")
    print(f"FORECAST TARGET: {FORECAST_DATE}")
    
    if last_candle_date.date() < expected_last.date():
        print(f"\n⚠️  WARNING: Lipsește lumânarea din {expected_last.strftime('%Y-%m-%d')}")
        print(f"    Yahoo Finance are un delay. Încearcă mai târziu sau folosește Binance API.")
        actual_forecast = (last_candle_date + timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"    Forecast-ul va fi pentru: {actual_forecast}")

    # log-returns zilnice în %
    rets = 100 * np.diff(np.log(close.values))
    dates = close.index[1:]
    rets_series = pd.Series(rets, index=dates, name="r")

    print(f"Returns: {len(rets_series)} observations")
    print(f"Mean(|r|) = {np.abs(rets_series).mean():.3f} %")

    # ----------------------------
    # 5.2. CALIBRARE AVANSATĂ MSM
    # ----------------------------
    # Alege metoda de calibrare: 'mle', 'grid', 'empirical', 'hybrid'
    CALIBRATION_METHOD = 'hybrid'
    
    calibration_result = calibrate_msm_advanced(
        rets_series,
        num_states=5,
        method=CALIBRATION_METHOD,
        target_var_breach=0.05,
        verbose=True
    )
    
    # Extrage parametrii calibrați
    sigma_low = calibration_result['sigma_low']
    sigma_high = calibration_result['sigma_high']
    p_stay = calibration_result['p_stay']
    
    # ----------------------------
    # 5.3. Forecast volatilitate cu MSM (CORECT: out-of-sample)
    # ----------------------------
    sigma_forecast, sigma_filtered, msm_filter_probs, sigma_states, P_matrix = msm_vol_forecast(
        rets_series,
        num_states=5,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
        p_stay=p_stay
    )

    print("\n=== Vol Forecast Summary ===")
    print(f"States: 5, sigma_low={sigma_low:.3f}%, sigma_high={sigma_high:.3f}%, p_stay={p_stay:.3f}")
    print(f"Mean sigma_forecast (out-of-sample): {sigma_forecast.mean():.3f}%")
    print(f"Mean sigma_filtered (in-sample):     {sigma_filtered.mean():.3f}%")
    corr_forecast = np.corrcoef(np.abs(rets_series.values), sigma_forecast.values)[0, 1]
    corr_filtered = np.corrcoef(np.abs(rets_series.values), sigma_filtered.values)[0, 1]
    print(f"Corr(|r_t|, σ_forecast_t) = {corr_forecast:.3f}  (out-of-sample)")
    print(f"Corr(|r_t|, σ_filtered_t) = {corr_filtered:.3f}  (in-sample)")


    # VaR forecast CORECT pentru ziua următoare
    VaR_t1, sigma_t1, z, pi_t1 = msm_var_forecast_next_day(
        msm_filter_probs, sigma_states, P_matrix
    )

    # Data ultimei observații și data forecast-ului
    last_data_date = rets_series.index[-1]
    
    # Crypto tradează 24/7 - nu sărim weekendul
    is_crypto = ticker.endswith("-USD") and ticker not in ["^SPX", "^GSPC", "SPY"]
    
    forecast_date = last_data_date + pd.Timedelta(days=1)
    
    if not is_crypto:
        # Doar pentru acțiuni: ajustare pentru weekend
        if forecast_date.weekday() == 5:  # Sâmbătă
            forecast_date += pd.Timedelta(days=2)
        elif forecast_date.weekday() == 6:  # Duminică
            forecast_date += pd.Timedelta(days=1)

    print("\n=== MSM VaR(5%) Forecast for Tomorrow ===")
    print(f"Last data date: {last_data_date.strftime('%Y-%m-%d')} ({last_data_date.strftime('%A')})")
    print(f"Forecast date:  {forecast_date.strftime('%Y-%m-%d')} ({forecast_date.strftime('%A')})")
    print(f"Sigma forecast (t+1|t): {sigma_t1:.3f}%")
    print(f"z-score (5%): {z:.3f}")
    print(f"VaR(5%) forecast for {forecast_date.strftime('%Y-%m-%d')}: {VaR_t1:.3f}%")
    print(f"State probabilities for t+1: {np.round(pi_t1, 3)}")


    

    # ----------------------------
    # 5.4. Tail probabilities (1,3,5 zile)
    # ----------------------------
    tail_info = msm_tail_probs(
        rets_series=rets_series,
        msm_filter_probs=msm_filter_probs.values,  # (T, K)
        sigma_states=sigma_states,
        alpha=0.05,          # P_emp = 5%
        horizons=(1, 3, 5),  # 1, 3, 5 zile
        use_student_t=True  # poți pune True dacă vrei cozi mai groase
    )

    L1 = tail_info["L1"]
    p1 = tail_info["p1"]
    probs = tail_info["horizon_probs"]
    dist_used = tail_info.get("distribution", "Normal")

    print(f"\n===  Tail Probabilities (conditional on current regime) ===")
    print(f"Distribution used: {dist_used}")
    print(f"Empirical L1 (5% quantile)     : {L1:.3f} %")
    print(f"P(tail in 1 day)     ~ {p1*100:5.2f}%  (r_t+1 <= L1)")
    for H, pH in probs.items():
        print(f"P(tail in {H:1d} days) ~ {pH*100:5.2f}%  (at least one r <= L1)")

    # ----------------------------
    # 5.5. Construim VaR(5%) pe baza MSM - CORECT OUT-OF-SAMPLE
    # ----------------------------
    alpha = 0.05
    z_alpha = norm.ppf(alpha)   # ≈ -1.645

    # VaR_t = μ_t + z_alpha * sigma_{t|t-1} (FORECAST, nu filtered!)
    # Acum comparăm r_t cu VaR calculat din σ_{t|t-1} (făcut ÎNAINTE de a vedea r_t)
    var_5 = z_alpha * sigma_forecast  # CORECT: out-of-sample

    df_msm_var = pd.DataFrame({
        "r_realized": rets_series,
        "sigma_forecast": sigma_forecast,  # σ_{t|t-1}
        "sigma_filtered": sigma_filtered,  # σ_t (pentru comparație)
        "VaR_5": var_5
    }).dropna()

    df_msm_var["breach"] = (df_msm_var["r_realized"] < df_msm_var["VaR_5"]).astype(int)
    breach_rate = df_msm_var["breach"].mean()

    print("\n=== MSM–VaR(5%) Backtest (OUT-OF-SAMPLE CORECT) ===")
    print(f"Număr de zile testate: {len(df_msm_var)}")
    print(f"Număr de breach-uri:  {df_msm_var['breach'].sum()}")
    print(f"Rată empirică breach: {breach_rate*100:.2f}% (teoretic: 5%)")

    # ----------------------------
    # 5.6. Kupiec & Christoffersen pe MSM–VaR (OUT-OF-SAMPLE)
    # ----------------------------
    breaches = df_msm_var["breach"]

    LR_uc, p_uc, x_uc, n_uc = kupiec_test(breaches, alpha=alpha)
    LR_ind, p_ind, (n00, n01, n10, n11) = christoffersen_independence_test(breaches)

    if np.isfinite(LR_uc) and np.isfinite(LR_ind):
        LR_cc = LR_uc + LR_ind
        p_cc  = chi2.sf(LR_cc, df=2)
    else:
        LR_cc, p_cc = np.nan, np.nan

    print("\n--- Kupiec / Christoffersen Backtests (MSM–VaR 5% OUT-OF-SAMPLE) ---")
    print(f"Kupiec UC: LR={LR_uc:.3f} | p-value={p_uc:.4f} | hit ratio={100*x_uc/n_uc:.2f}% ({x_uc}/{n_uc})")
    print(f"Christoffersen IND: LR={LR_ind:.3f} | p-value={p_ind:.4f} | transitions n00={n00}, n01={n01}, n10={n10}, n11={n11}")
    print(f"Conditional Coverage (UC+IND): LR={LR_cc:.3f} | p-value={p_cc:.4f}")
    
    # Interpretare
    print("\n--- Interpretare ---")
    if p_uc >= 0.05:
        print(f"✅ Kupiec UC: PASS (p={p_uc:.4f} >= 0.05) - Breach rate e consistent cu 5%")
    else:
        print(f"❌ Kupiec UC: FAIL (p={p_uc:.4f} < 0.05) - Breach rate diferă semnificativ de 5%")
    
    if p_ind >= 0.05:
        print(f"✅ Christoffersen IND: PASS (p={p_ind:.4f} >= 0.05) - Breaches sunt independente")
    else:
        print(f"❌ Christoffersen IND: FAIL (p={p_ind:.4f} < 0.05) - Breaches vin în clustere")
    
    if p_cc >= 0.05:
        print(f"✅ Conditional Coverage: PASS (p={p_cc:.4f} >= 0.05) - Model bine calibrat")
    else:
        print(f"❌ Conditional Coverage: FAIL (p={p_cc:.4f} < 0.05) - Model necesită ajustări")

    # ----------------------------
    # 5.7. Plot returns + VaR MSM (ca în articol)
    # ----------------------------
    plt.figure(figsize=(14, 6))

    # returns (bare negre)
    plt.vlines(
        df_msm_var.index,
        ymin=0,
        ymax=df_msm_var["r_realized"],
        color="black",
        linewidth=0.5,
        alpha=0.7,
        label="Log-returns"
    )

    # VaR(5%) MSM (linie roșie)
    plt.plot(
        df_msm_var.index,
        df_msm_var["VaR_5"],
        color="red",
        linewidth=1.0,
        label="VaR(5%) Out-of-Sample"
    )

    plt.title(f"{ticker} — Log Returns vs VaR(5%) [Out-of-Sample]", fontsize=13)
    plt.ylabel("Return (%)")
    plt.grid(alpha=0.2)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

    # extra info: quantila empirică 5%
    L1_emp = np.quantile(rets_series, 0.05)
    print(f"\nEmpirical 5% quantile of returns: {L1_emp:.3f}%")
    print(end="\n")
