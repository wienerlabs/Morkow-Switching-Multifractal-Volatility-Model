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
            "token": "TEST",
            "ticker": "AAPL",
            "period": "1y",
            "num_states": 5,
            "method": "empirical",
            "data_source": "yfinance",
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert "p_stay" in data
        # p_stay is now a list
        assert isinstance(data["p_stay"], list)

    def test_var_normal(self):
        r = client.get("/api/v1/var/95", params={"token": "TEST"})
        assert r.status_code == 200
        data = r.json()
        assert data["var_pct"] < 0
        assert data["distribution"] == "normal"

    def test_var_student_t(self):
        r = client.get("/api/v1/var/95", params={
            "token": "TEST", "use_student_t": True, "nu": 5.0,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["distribution"] == "student_t"

    def test_student_t_wider_than_normal(self):
        r_n = client.get("/api/v1/var/95", params={"token": "TEST"})
        r_t = client.get("/api/v1/var/95", params={
            "token": "TEST", "use_student_t": True, "nu": 5.0,
        })
        assert r_t.json()["var_pct"] < r_n.json()["var_pct"]

    def test_regime(self):
        r = client.get("/api/v1/regime", params={"token": "TEST"})
        assert r.status_code == 200
        assert "current_state" in r.json()

    def test_volatility(self):
        r = client.get("/api/v1/volatility", params={"token": "TEST"})
        assert r.status_code == 200
        assert "current_sigma" in r.json()

    def test_backtest(self):
        r = client.get("/api/v1/backtest", params={"token": "TEST"})
        assert r.status_code == 200
        assert "breach_rate" in r.json()

    def test_regime_durations(self):
        r = client.get("/api/v1/regime/durations", params={"token": "TEST"})
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data["p_stay"], list)
        assert len(data["durations"]) == 5

    def test_regime_history(self):
        r = client.get("/api/v1/regime/history", params={"token": "TEST"})
        assert r.status_code == 200
        assert "periods" in r.json()

    def test_regime_statistics(self):
        r = client.get("/api/v1/regime/statistics", params={"token": "TEST"})
        assert r.status_code == 200
        assert "stats" in r.json()


class TestMissingToken:
    def test_var_404(self):
        r = client.get("/api/v1/var/95", params={"token": "NONEXISTENT"})
        assert r.status_code == 404

    def test_regime_404(self):
        r = client.get("/api/v1/regime", params={"token": "NONEXISTENT"})
        assert r.status_code == 404

