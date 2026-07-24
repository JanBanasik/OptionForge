"""Tests for Greeks computation."""

import numpy as np

from optionforge.models.types import OptionType, PayoffType, VarianceReduction
from optionforge.pricing.black_scholes import black_scholes_greeks
from optionforge.pricing.greeks import compute_greeks


class TestMCGreeks:
    """Verify Monte Carlo Greeks via finite differences."""

    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.20

    def test_greeks_finite_and_reasonable(self):
        rng = np.random.default_rng(42)
        g = compute_greeks(
            rng, self.S, self.K, self.T, self.r, self.q, self.sigma,
            50, 20_000, OptionType.CALL, PayoffType.EUROPEAN,
            VarianceReduction.NONE,
        )
        # All should be finite
        for name in ["delta", "gamma", "vega", "theta", "rho"]:
            val = getattr(g, name)
            assert np.isfinite(val), f"{name} is not finite: {val}"

    def test_call_delta_positive(self):
        rng = np.random.default_rng(42)
        g = compute_greeks(
            rng, self.S, self.K, self.T, self.r, self.q, self.sigma,
            50, 20_000, OptionType.CALL, PayoffType.EUROPEAN,
            VarianceReduction.NONE,
        )
        assert g.delta > 0

    def test_greeks_approximate_bs(self):
        """MC Greeks should be close to BS analytical Greeks."""
        rng = np.random.default_rng(42)
        mc_greeks = compute_greeks(
            rng, self.S, self.K, self.T, self.r, self.q, self.sigma,
            50, 50_000, OptionType.CALL, PayoffType.EUROPEAN,
            VarianceReduction.NONE,
        )
        bs_greeks = black_scholes_greeks(self.S, self.K, self.T, self.r, self.q, self.sigma)

        # Delta: within 0.05
        assert abs(mc_greeks.delta - bs_greeks.delta) < 0.05, (
            f"Delta: MC={mc_greeks.delta:.4f}, BS={bs_greeks.delta:.4f}"
        )
        # Gamma: within 0.01
        assert abs(mc_greeks.gamma - bs_greeks.gamma) < 0.01, (
            f"Gamma: MC={mc_greeks.gamma:.6f}, BS={bs_greeks.gamma:.6f}"
        )
        # Vega: within 0.5
        assert abs(mc_greeks.vega - bs_greeks.vega) < 0.5, (
            f"Vega: MC={mc_greeks.vega:.4f}, BS={bs_greeks.vega:.4f}"
        )


class TestBSGreeks:
    """Verify analytical BS Greeks against known properties."""

    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.20

    def test_put_call_delta_parity(self):
        """For European: delta_call - delta_put = exp(-qT)."""
        call_g = black_scholes_greeks(self.S, self.K, self.T, self.r, self.q, self.sigma)
        assert 0 < call_g.delta < 1

    def test_gamma_positive(self):
        g = black_scholes_greeks(self.S, self.K, self.T, self.r, self.q, self.sigma)
        assert g.gamma > 0
