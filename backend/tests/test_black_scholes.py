"""Tests for Black-Scholes analytical pricing."""

import numpy as np

from optionforge.models.types import OptionType
from optionforge.pricing.black_scholes import (
    _norm_cdf,
    black_scholes_greeks,
    black_scholes_price,
)


class TestBlackScholesReference:
    """Verify Black-Scholes against known reference values."""

    # Standard parameters
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.20

    def test_call_reference(self):
        """European call: S=100, K=100, T=1, r=5%, q=2%, σ=20%."""
        price = black_scholes_price(self.S, self.K, self.T, self.r, self.q, self.sigma, OptionType.CALL)
        # Expected ~9.23 (from known BS calculation)
        assert 9.0 < price < 9.5
        assert round(price, 2) == 9.23

    def test_put_reference(self):
        """European put at the same parameters."""
        price = black_scholes_price(self.S, self.K, self.T, self.r, self.q, self.sigma, OptionType.PUT)
        assert 6.0 < price < 6.7
        assert round(price, 2) == 6.33

    def test_deep_itm_call(self):
        """Deep ITM call: S >> K."""
        price = black_scholes_price(200.0, 100.0, 0.5, 0.05, 0.0, 0.3, OptionType.CALL)
        # Should be close to S - K*exp(-rT) ≈ 200 - 97.53 = 102.47
        assert 100.0 < price < 105.0

    def test_deep_otm_call(self):
        """Deep OTM call should approach zero."""
        price = black_scholes_price(50.0, 100.0, 0.5, 0.05, 0.0, 0.2, OptionType.CALL)
        assert price < 1.0

    def test_zero_volatility(self):
        """Zero volatility: deterministic forward."""
        price = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.0, OptionType.CALL)
        # Forward = 100*exp(0.05) = 105.13, intrinsic = 5.13, discounted = 5.13*exp(-0.05) = 4.88
        expected = max(100.0 - 100.0 * np.exp(-0.05), 0.0)
        assert abs(price - expected) < 1e-6

    def test_zero_maturity(self):
        """At expiry: intrinsic value only."""
        price = black_scholes_price(120.0, 100.0, 0.0, 0.05, 0.0, 0.3, OptionType.CALL)
        assert price == 20.0

        price = black_scholes_price(80.0, 100.0, 0.0, 0.05, 0.0, 0.3, OptionType.CALL)
        assert price == 0.0


class TestPutCallParity:
    """Verify put-call parity: C - P = S*exp(-qT) - K*exp(-rT)."""

    def test_parity_atm(self):
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.20
        call = black_scholes_price(S, K, T, r, q, sigma, OptionType.CALL)
        put = black_scholes_price(S, K, T, r, q, sigma, OptionType.PUT)
        lhs = call - put
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs(lhs - rhs) < 1e-10

    def test_parity_various_params(self):
        """Test parity holds across parameter grid."""
        rng = np.random.default_rng(123)
        for _ in range(100):
            S = rng.uniform(50, 150)
            K = rng.uniform(80, 120)
            T = rng.uniform(0.1, 3.0)
            r = rng.uniform(0.0, 0.15)
            q = rng.uniform(0.0, 0.08)
            sigma = rng.uniform(0.1, 0.8)
            call = black_scholes_price(S, K, T, r, q, sigma, OptionType.CALL)
            put = black_scholes_price(S, K, T, r, q, sigma, OptionType.PUT)
            lhs = call - put
            rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
            assert abs(lhs - rhs) < 1e-10

    def test_parity_with_dividends(self):
        """Parity with non-zero dividend yield."""
        S, K, T, r, q, sigma = 100.0, 105.0, 0.5, 0.03, 0.04, 0.25
        call = black_scholes_price(S, K, T, r, q, sigma, OptionType.CALL)
        put = black_scholes_price(S, K, T, r, q, sigma, OptionType.PUT)
        lhs = call - put
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs(lhs - rhs) < 1e-10


class TestBlackScholesGreeks:
    """Verify analytical Greeks have expected signs and magnitudes."""

    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.20

    def test_call_greeks_signs(self):
        g = black_scholes_greeks(self.S, self.K, self.T, self.r, self.q, self.sigma)
        # Call delta > 0, gamma > 0, vega > 0, rho > 0
        assert g.delta > 0
        assert g.gamma > 0
        assert g.vega > 0
        assert g.rho > 0
        # Theta can be negative for calls (time decay)

    def test_delta_range(self):
        """Delta should be between 0 and 1 for calls."""
        g = black_scholes_greeks(self.S, self.K, self.T, self.r, self.q, self.sigma)
        assert 0.4 < g.delta < 0.8  # ATM call approx 0.5-0.6

    def test_greeks_finite(self):
        """All Greeks should be finite."""
        g = black_scholes_greeks(self.S, self.K, self.T, self.r, self.q, self.sigma)
        for name in ["delta", "gamma", "vega", "theta", "rho"]:
            val = getattr(g, name)
            assert np.isfinite(val), f"{name} is not finite: {val}"


class TestNormCdf:
    """Verify normal CDF helper."""

    def test_symmetry(self):
        assert abs(_norm_cdf(0.0) - 0.5) < 1e-10
        assert abs(_norm_cdf(1.0) + _norm_cdf(-1.0) - 1.0) < 1e-10

    def test_extreme_values(self):
        assert _norm_cdf(-10.0) < 1e-10
        assert _norm_cdf(10.0) > 1.0 - 1e-10
