import numpy as np
import pandas as pd
import pytest

from cortex import msm


@pytest.fixture(scope="session")
def sample_returns() -> pd.Series:
    """300-point synthetic return series (seeded for reproducibility)."""
    rng = np.random.RandomState(42)
    return pd.Series(rng.randn(300) * 1.5, name="synthetic")


@pytest.fixture(scope="session")
def calibrated_model(sample_returns):
    """Pre-calibrated MSM model dict (empirical method — fast)."""
    cal = msm.calibrate_msm_advanced(
        sample_returns, num_states=5, method="empirical", verbose=False
    )
    sf, sfi, fp, ss, P = msm.msm_vol_forecast(
        sample_returns,
        num_states=cal["num_states"],
        sigma_low=cal["sigma_low"],
        sigma_high=cal["sigma_high"],
        p_stay=cal["p_stay"],
    )
    return {
        "calibration": cal,
        "returns": sample_returns,
        "sigma_forecast": sf,
        "sigma_filtered": sfi,
        "filter_probs": fp,
        "sigma_states": ss,
        "P_matrix": P,
    }


@pytest.fixture(scope="session")
def multivariate_returns() -> pd.DataFrame:
    """3-asset synthetic returns for portfolio tests."""
    rng = np.random.RandomState(123)
    n = 200
    common = rng.randn(n) * 0.5
    df = pd.DataFrame({
        "A": common + rng.randn(n) * 1.0,
        "B": common + rng.randn(n) * 1.2,
        "C": common + rng.randn(n) * 0.8,
    })
    return df

