"""Tests for FastAPI endpoints using TestClient."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestHealthAndDocs:
    def test_openapi_schema(self):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        assert "paths" in r.json()


class TestCalibrateAndVar:
    """Integration: calibrate → VaR → regime → backtest pipeline."""

    @pytest.fixture(autouse=True, scope="class")
    def _calibrate(self):
        """Calibrate once for all tests in this class."""
        r = client.post("/api/v1/calibrate", json={
            "token": "AAPL",
            "start_date": "2024-01-01",
            "end_date": "2025-01-01",
            "num_states": 5,
            "method": "empirical",
            "data_source": "yfinance",
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert "p_stay" in data
        assert isinstance(data["p_stay"], list)

    def test_var_normal(self):
        r = client.get("/api/v1/var/95", params={"token": "AAPL"})
        assert r.status_code == 200
        data = r.json()
        assert data["var_value"] < 0
        assert data["distribution"] == "normal"

    def test_var_student_t(self):
        r = client.get("/api/v1/var/95", params={
            "token": "AAPL", "use_student_t": True, "nu": 5.0,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["distribution"] == "student_t"

    def test_student_t_wider_than_normal(self):
        r_n = client.get("/api/v1/var/95", params={"token": "AAPL"})
        r_t = client.get("/api/v1/var/95", params={
            "token": "AAPL", "use_student_t": True, "nu": 5.0,
        })
        assert r_t.json()["var_value"] < r_n.json()["var_value"]

    def test_regime(self):
        r = client.get("/api/v1/regime/current", params={"token": "AAPL"})
        assert r.status_code == 200
        assert "regime_state" in r.json()

    def test_volatility(self):
        r = client.get("/api/v1/volatility/forecast", params={"token": "AAPL"})
        assert r.status_code == 200
        assert "sigma_forecast" in r.json()

    def test_backtest(self):
        r = client.get("/api/v1/backtest/summary", params={"token": "AAPL"})
        assert r.status_code == 200
        assert "breach_rate" in r.json()

    def test_regime_durations(self):
        r = client.get("/api/v1/regime/durations", params={"token": "AAPL"})
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data["p_stay"], list)
        assert len(data["durations"]) == 5

    def test_regime_history(self):
        r = client.get("/api/v1/regime/history", params={"token": "AAPL"})
        assert r.status_code == 200
        assert "periods" in r.json()

    def test_regime_statistics(self):
        r = client.get("/api/v1/regime/statistics", params={"token": "AAPL"})
        assert r.status_code == 200
        assert "statistics" in r.json()


class TestMissingToken:
    def test_var_404(self):
        r = client.get("/api/v1/var/95", params={"token": "NONEXISTENT"})
        assert r.status_code == 404

    def test_regime_404(self):
        r = client.get("/api/v1/regime/current", params={"token": "NONEXISTENT"})
        assert r.status_code == 404



class TestGuardianUnit:
    """Unit tests for guardian.py scoring functions."""

    def _make_evt_data(self, xi: float = -0.1, beta: float = 1.5) -> dict:
        return {
            "xi": xi, "beta": beta, "threshold": 2.0,
            "n_total": 300, "n_exceedances": 30,
        }

    def _make_svj_data(self, lambda_: float = 10.0, jump_share: float = 40.0) -> dict:
        import numpy as np
        import pandas as pd
        rng = np.random.RandomState(42)
        returns = pd.Series(rng.randn(200) * 2.0)
        cal = {
            "kappa": 5.0, "theta": 0.04, "sigma": 0.3, "rho": -0.7,
            "lambda_": lambda_, "mu_j": -0.02, "sigma_j": 0.05,
        }
        return {"returns": returns, "calibration": cal}

    def _make_hawkes_data(self) -> dict:
        import numpy as np
        return {
            "mu": 0.5, "alpha": 0.1, "beta": 1.0,
            "branching_ratio": 0.1, "half_life": 0.693,
            "stationary": True, "n_events": 15,
            "event_times": np.array([1.0, 2.5, 3.0, 5.0, 7.0, 8.0, 9.0, 10.0]),
            "event_returns": np.array([-3.0, -2.5, -4.0, -2.0, -3.5, -2.8, -3.2, -2.1]),
            "threshold": 2.0,
        }

    def _make_model_data(self, num_states: int = 5, crisis: bool = False) -> dict:
        import numpy as np
        import pandas as pd
        if crisis:
            probs = np.array([0.01, 0.02, 0.03, 0.10, 0.84])
        else:
            probs = np.array([0.60, 0.25, 0.10, 0.04, 0.01])
        filter_probs = pd.DataFrame([probs], columns=[f"state_{i+1}" for i in range(num_states)])
        return {
            "calibration": {"num_states": num_states},
            "filter_probs": filter_probs,
        }

    def test_score_evt_low_risk(self):
        from cortex.guardian import _score_evt
        result = _score_evt(self._make_evt_data(xi=-0.2, beta=1.0))
        assert 0 <= result["score"] <= 100
        assert result["component"] == "evt"
        assert "var_995" in result["details"]

    def test_score_svj_low_jump(self):
        from cortex.guardian import _score_svj
        result = _score_svj(self._make_svj_data(lambda_=5.0))
        assert 0 <= result["score"] <= 100
        assert result["component"] == "svj"
        assert "jump_share_pct" in result["details"]

    def test_score_hawkes_low_risk(self):
        from cortex.guardian import _score_hawkes
        result = _score_hawkes(self._make_hawkes_data())
        assert 0 <= result["score"] <= 100
        assert result["component"] == "hawkes"
        assert "risk_level" in result["details"]

    def test_score_regime_calm(self):
        from cortex.guardian import _score_regime
        result = _score_regime(self._make_model_data(crisis=False))
        assert result["score"] < 30
        assert result["details"]["current_regime"] == 1

    def test_score_regime_crisis(self):
        from cortex.guardian import _score_regime
        result = _score_regime(self._make_model_data(crisis=True))
        assert result["score"] > 70
        assert result["details"]["current_regime"] == 5

    def test_assess_trade_approved(self):
        from cortex.guardian import _cache, assess_trade
        _cache.clear()
        result = assess_trade(
            token="TEST", trade_size_usd=10000.0, direction="long",
            model_data=self._make_model_data(crisis=False),
            evt_data=self._make_evt_data(),
            svj_data=self._make_svj_data(),
            hawkes_data=self._make_hawkes_data(),
        )
        assert isinstance(result["approved"], bool)
        assert 0 <= result["risk_score"] <= 100
        # 4 of 7 components present (no news, no alams, no agent_confidence) → confidence ≈ 0.65
        assert result["confidence"] == pytest.approx(0.65, abs=0.01)
        assert len(result["component_scores"]) == 4
        assert result["recommended_size"] > 0

    def test_assess_trade_no_models(self):
        from cortex.guardian import _cache, assess_trade
        _cache.clear()
        result = assess_trade(
            token="EMPTY", trade_size_usd=5000.0, direction="short",
            model_data=None, evt_data=None, svj_data=None, hawkes_data=None,
        )
        assert result["risk_score"] == 50.0
        assert result["confidence"] == 0.0

    def test_assess_trade_crisis_veto(self):
        from cortex.guardian import _cache, assess_trade
        _cache.clear()
        result = assess_trade(
            token="CRISIS", trade_size_usd=10000.0, direction="long",
            model_data=self._make_model_data(crisis=True),
            evt_data=None, svj_data=None, hawkes_data=None,
        )
        assert result["regime_state"] == 5
        assert result["recommended_size"] < 10000.0

    def test_cache_hit(self):
        from cortex.guardian import _cache, assess_trade
        _cache.clear()
        r1 = assess_trade(
            token="CACHE", trade_size_usd=1000.0, direction="long",
            model_data=self._make_model_data(), evt_data=None,
            svj_data=None, hawkes_data=None,
        )
        assert r1["from_cache"] is False
        r2 = assess_trade(
            token="CACHE", trade_size_usd=1000.0, direction="long",
            model_data=self._make_model_data(), evt_data=None,
            svj_data=None, hawkes_data=None,
        )
        assert r2["from_cache"] is True

    def test_position_sizing_scales_down(self):
        from cortex.guardian import _recommend_size
        full = _recommend_size(10000.0, 0.0, 1, 5)
        half = _recommend_size(10000.0, 50.0, 1, 5)
        crisis = _recommend_size(10000.0, 50.0, 5, 5)
        assert full == 10000.0
        assert half == 5000.0
        assert crisis < half


class TestGuardianEndpoint:
    """Integration test: reuses AAPL calibration from TestCalibrateAndVar."""

    def test_guardian_assess_with_msm_only(self):
        # Ensure AAPL is calibrated first
        client.post("/api/v1/calibrate", json={
            "token": "AAPL",
            "start_date": "2024-01-01",
            "end_date": "2025-01-01",
            "num_states": 5,
            "method": "empirical",
            "data_source": "yfinance",
        })
        r = client.post("/api/v1/guardian/assess", json={
            "token": "AAPL",
            "trade_size_usd": 10000.0,
            "direction": "long",
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert "approved" in data
        assert "risk_score" in data
        assert "recommended_size" in data
        assert "component_scores" in data
        assert data["confidence"] > 0
        assert len(data["component_scores"]) >= 1

    def test_guardian_404_no_models(self):
        r = client.post("/api/v1/guardian/assess", json={
            "token": "NONEXISTENT_TOKEN",
            "trade_size_usd": 5000.0,
            "direction": "short",
        })
        assert r.status_code == 404

    def test_guardian_invalid_direction(self):
        r = client.post("/api/v1/guardian/assess", json={
            "token": "AAPL",
            "trade_size_usd": 5000.0,
            "direction": "sideways",
        })
        assert r.status_code == 422