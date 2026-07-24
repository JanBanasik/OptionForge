"""Tests for FastAPI endpoints."""

from fastapi.testclient import TestClient

from optionforge.main import app

client = TestClient(app)


class TestPriceEndpoint:
    """Test /api/price endpoint."""

    def test_price_european_call(self):
        resp = client.post("/api/price", json={
            "spot": 100.0,
            "strike": 100.0,
            "maturity": 1.0,
            "risk_free_rate": 0.05,
            "dividend_yield": 0.02,
            "volatility": 0.20,
            "n_paths": 10_000,
            "n_steps": 100,
            "option_type": "call",
            "payoff_type": "european",
            "seed": 42,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["price"] > 0
        assert data["standard_error"] > 0
        assert data["confidence_interval_lower"] < data["price"] < data["confidence_interval_upper"]
        assert data["black_scholes_price"] is not None
        assert data["absolute_error"] is not None

    def test_price_european_put(self):
        resp = client.post("/api/price", json={
            "spot": 100.0,
            "strike": 100.0,
            "maturity": 1.0,
            "risk_free_rate": 0.05,
            "dividend_yield": 0.02,
            "volatility": 0.20,
            "n_paths": 10_000,
            "n_steps": 100,
            "option_type": "put",
            "payoff_type": "european",
            "seed": 42,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["price"] > 0

    def test_price_asian_call(self):
        resp = client.post("/api/price", json={
            "spot": 100.0,
            "strike": 100.0,
            "maturity": 1.0,
            "risk_free_rate": 0.05,
            "dividend_yield": 0.02,
            "volatility": 0.20,
            "n_paths": 10_000,
            "n_steps": 50,
            "option_type": "call",
            "payoff_type": "asian",
            "seed": 42,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["price"] > 0
        # Asian should have no BS price
        assert data["black_scholes_price"] is None

    def test_price_with_antithetic(self):
        resp = client.post("/api/price", json={
            "spot": 100.0,
            "strike": 100.0,
            "maturity": 1.0,
            "risk_free_rate": 0.05,
            "dividend_yield": 0.02,
            "volatility": 0.20,
            "n_paths": 10_000,
            "n_steps": 100,
            "option_type": "call",
            "payoff_type": "european",
            "variance_reduction": "antithetic",
            "seed": 42,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["price"] > 0

    def test_greeks_in_response(self):
        resp = client.post("/api/price", json={
            "spot": 100.0,
            "strike": 100.0,
            "maturity": 1.0,
            "risk_free_rate": 0.05,
            "dividend_yield": 0.02,
            "volatility": 0.20,
            "n_paths": 10_000,
            "n_steps": 100,
            "option_type": "call",
            "payoff_type": "european",
            "seed": 42,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["greeks"] is not None
        assert "delta" in data["greeks"]
        assert data["bs_greeks"] is not None

    def test_validation_errors(self):
        """Test input validation."""
        # Negative spot
        resp = client.post("/api/price", json={
            "spot": -100.0, "strike": 100.0, "maturity": 1.0,
            "risk_free_rate": 0.05, "dividend_yield": 0.02,
            "volatility": 0.20, "n_paths": 1000, "n_steps": 100,
            "option_type": "call", "payoff_type": "european",
        })
        assert resp.status_code == 422

        # Invalid option_type
        resp = client.post("/api/price", json={
            "spot": 100.0, "strike": 100.0, "maturity": 1.0,
            "risk_free_rate": 0.05, "dividend_yield": 0.02,
            "volatility": 0.20, "n_paths": 1000, "n_steps": 100,
            "option_type": "digital", "payoff_type": "european",
        })
        assert resp.status_code == 422


class TestVisualizationEndpoint:
    """Test /api/visualization endpoint."""

    def test_visualization_data(self):
        resp = client.post("/api/visualization", json={
            "spot": 100.0,
            "strike": 100.0,
            "maturity": 1.0,
            "risk_free_rate": 0.05,
            "dividend_yield": 0.02,
            "volatility": 0.20,
            "n_paths": 5_000,
            "n_steps": 100,
            "option_type": "call",
            "payoff_type": "european",
            "seed": 42,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["sampled_paths"]) <= 100  # Capped
        assert len(data["time_grid"]) > 0
        assert len(data["terminal_bin_edges"]) > 0
        assert len(data["terminal_bin_counts"]) > 0
        assert len(data["payoff_bin_edges"]) > 0
        assert len(data["payoff_bin_counts"]) > 0
        assert len(data["convergence"]) > 0
        assert data["greeks"] is not None


class TestDeterministicAPI:
    """Same seed should give same results across calls."""

    def test_same_seed_same_price(self):
        params = {
            "spot": 100.0, "strike": 100.0, "maturity": 1.0,
            "risk_free_rate": 0.05, "dividend_yield": 0.02,
            "volatility": 0.20, "n_paths": 5_000, "n_steps": 100,
            "option_type": "call", "payoff_type": "european",
            "seed": 999,
        }
        resp1 = client.post("/api/price", json=params)
        resp2 = client.post("/api/price", json=params)
        assert resp1.json()["price"] == resp2.json()["price"]
        assert resp1.json()["standard_error"] == resp2.json()["standard_error"]
