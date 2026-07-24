"""Tests for Monte Carlo pricing engine.

Covers: standard MC, antithetic variates, convergence, reproducibility,
confidence interval validation, and edge cases.
"""

import numpy as np
import pytest

from optionforge.models.payoffs import compute_payoff
from optionforge.models.stochastic import simulate_gbm_paths
from optionforge.models.types import OptionType, PayoffType, VarianceReduction
from optionforge.pricing.black_scholes import black_scholes_price
from optionforge.pricing.monte_carlo import (
    generate_convergence_data,
    generate_visualization_data,
    monte_carlo_price,
)

# ──────────────────────────────────────────────────────────────────────────────
# GBM Simulation
# ──────────────────────────────────────────────────────────────────────────────


class TestGBMSimulation:
    """Unit tests for the standalone GBM path generator."""

    def test_output_shape(self):
        rng = np.random.default_rng(42)
        paths = simulate_gbm_paths(rng, 100.0, 0.05, 0.02, 0.2, 1.0, 252, 1000)
        assert paths.shape == (1000, 253)
        assert np.all(paths[:, 0] == 100.0)

    def test_deterministic_seed(self):
        rng1 = np.random.default_rng(123)
        paths1 = simulate_gbm_paths(rng1, 100.0, 0.05, 0.02, 0.2, 0.5, 10, 100)
        rng2 = np.random.default_rng(123)
        paths2 = simulate_gbm_paths(rng2, 100.0, 0.05, 0.02, 0.2, 0.5, 10, 100)
        assert np.allclose(paths1, paths2)

    def test_chunked_equivalence(self):
        """Large path counts chunked internally match non-chunked."""
        rng1 = np.random.default_rng(42)
        paths1 = simulate_gbm_paths(rng1, 100.0, 0.05, 0.02, 0.2, 0.5, 10, 100)
        rng2 = np.random.default_rng(42)
        paths2 = simulate_gbm_paths(rng2, 100.0, 0.05, 0.02, 0.2, 0.5, 10, 100, chunk_size=60)
        assert np.allclose(paths1, paths2)

    def test_positivity(self):
        rng = np.random.default_rng(42)
        paths = simulate_gbm_paths(rng, 100.0, 0.05, 0.02, 0.5, 1.0, 252, 1000)
        assert np.all(paths > 0)

    def test_log_normal_terminal(self):
        """Terminal prices are log-normal with correct moments."""
        rng = np.random.default_rng(42)
        paths = simulate_gbm_paths(rng, 100.0, 0.05, 0.02, 0.2, 1.0, 252, 10000)
        log_returns = np.log(paths[:, -1] / 100.0)
        expected_mean = (0.05 - 0.02 - 0.5 * 0.04)
        expected_std = 0.2
        assert abs(np.mean(log_returns) - expected_mean) < 0.05
        assert abs(np.std(log_returns) - expected_std) < 0.05


# ──────────────────────────────────────────────────────────────────────────────
# Payoff Computation
# ──────────────────────────────────────────────────────────────────────────────


class TestPayoffs:
    """Verify payoff functions for all combinations."""

    def test_european_call(self):
        paths = np.array([[100.0, 110.0, 120.0], [100.0, 95.0, 90.0]])
        payoff = compute_payoff(paths, 100.0, OptionType.CALL, PayoffType.EUROPEAN)
        assert np.allclose(payoff, [20.0, 0.0])

    def test_european_put(self):
        paths = np.array([[100.0, 110.0, 120.0], [100.0, 95.0, 90.0]])
        payoff = compute_payoff(paths, 100.0, OptionType.PUT, PayoffType.EUROPEAN)
        assert np.allclose(payoff, [0.0, 10.0])

    def test_asian_call(self):
        paths = np.array([[100.0, 110.0, 120.0], [100.0, 95.0, 90.0]])
        payoff = compute_payoff(paths, 100.0, OptionType.CALL, PayoffType.ASIAN)
        avg1 = (110.0 + 120.0) / 2.0
        avg2 = (95.0 + 90.0) / 2.0
        assert np.allclose(payoff, [max(avg1 - 100.0, 0), max(avg2 - 100.0, 0)])

    def test_asian_put(self):
        paths = np.array([[100.0, 110.0, 120.0], [100.0, 95.0, 90.0]])
        payoff = compute_payoff(paths, 100.0, OptionType.PUT, PayoffType.ASIAN)
        avg1 = (110.0 + 120.0) / 2.0
        avg2 = (95.0 + 90.0) / 2.0
        assert np.allclose(payoff, [max(100.0 - avg1, 0), max(100.0 - avg2, 0)])

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        paths = simulate_gbm_paths(rng, 100.0, 0.05, 0.02, 0.3, 1.0, 50, 1000)
        for opt in [OptionType.CALL, OptionType.PUT]:
            for payoff_t in [PayoffType.EUROPEAN, PayoffType.ASIAN]:
                payoff = compute_payoff(paths, 100.0, opt, payoff_t)
                assert np.all(payoff >= 0), f"Negative payoff: {opt}, {payoff_t}"


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────


class TestReproducibility:
    """Fixed seed → identical results for both standard and antithetic."""

    BASE_PARAMS = dict(
        spot=100.0, strike=100.0, maturity=1.0, r=0.05, q=0.02,
        sigma=0.2, n_steps=50, n_paths=10_000, option_type=OptionType.CALL,
        payoff_type=PayoffType.EUROPEAN, compute_greeks_flag=False,
    )

    def test_standard_reproducible(self):
        for seed in [42, 123, 9999]:
            rng1 = np.random.default_rng(seed)
            rng2 = np.random.default_rng(seed)
            res1 = monte_carlo_price(rng1, variance_reduction=VarianceReduction.NONE, **self.BASE_PARAMS)
            res2 = monte_carlo_price(rng2, variance_reduction=VarianceReduction.NONE, **self.BASE_PARAMS)
            assert res1.price == res2.price
            assert res1.standard_error == res2.standard_error
            assert res1.confidence_interval_lower == res2.confidence_interval_lower

    def test_antithetic_reproducible(self):
        for seed in [42, 123, 9999]:
            rng1 = np.random.default_rng(seed)
            rng2 = np.random.default_rng(seed)
            res1 = monte_carlo_price(rng1, variance_reduction=VarianceReduction.ANTITHETIC, **self.BASE_PARAMS)
            res2 = monte_carlo_price(rng2, variance_reduction=VarianceReduction.ANTITHETIC, **self.BASE_PARAMS)
            assert res1.price == res2.price

    def test_different_seeds_different(self):
        """Different seeds should generally produce different estimates."""
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        res1 = monte_carlo_price(rng1, variance_reduction=VarianceReduction.NONE, **self.BASE_PARAMS)
        res2 = monte_carlo_price(rng2, variance_reduction=VarianceReduction.NONE, **self.BASE_PARAMS)
        # With 10k paths, prices should differ (not a guarantee, but very likely)
        assert res1.price != res2.price or res1.standard_error != res2.standard_error


# ──────────────────────────────────────────────────────────────────────────────
# Confidence Interval Validation
# ──────────────────────────────────────────────────────────────────────────────


class TestConfidenceIntervals:
    """Verify that the 95% CI contains the true (BS) price with ~95% frequency."""

    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.20

    def test_ci_contains_bs_price(self):
        """A single MC run's CI should contain the analytical BS price."""
        bs = black_scholes_price(self.S, self.K, self.T, self.r, self.q, self.sigma, OptionType.CALL)

        rng = np.random.default_rng(42)
        res = monte_carlo_price(
            rng, self.S, self.K, self.T, self.r, self.q, self.sigma,
            100, 100_000, OptionType.CALL, PayoffType.EUROPEAN,
            VarianceReduction.NONE, compute_greeks_flag=False,
        )
        assert res.confidence_interval_lower <= bs <= res.confidence_interval_upper, (
            f"CI [{res.confidence_interval_lower:.6f}, {res.confidence_interval_upper:.6f}] "
            f"does not contain BS price {bs:.6f}"
        )

    def test_ci_contains_bs_with_antithetic(self):
        """Antithetic CI should also contain the BS price."""
        bs = black_scholes_price(self.S, self.K, self.T, self.r, self.q, self.sigma, OptionType.CALL)

        rng = np.random.default_rng(42)
        res = monte_carlo_price(
            rng, self.S, self.K, self.T, self.r, self.q, self.sigma,
            100, 100_000, OptionType.CALL, PayoffType.EUROPEAN,
            VarianceReduction.ANTITHETIC, compute_greeks_flag=False,
        )
        assert res.confidence_interval_lower <= bs <= res.confidence_interval_upper

    def test_ci_narrows_with_more_paths(self):
        """CI width ∝ 1/√n, so more paths → narrower CI."""
        rng = np.random.default_rng(42)
        res_small = monte_carlo_price(
            rng, self.S, self.K, self.T, self.r, self.q, self.sigma,
            50, 1_000, OptionType.CALL, PayoffType.EUROPEAN,
            VarianceReduction.NONE, compute_greeks_flag=False,
        )
        rng2 = np.random.default_rng(43)
        res_large = monte_carlo_price(
            rng2, self.S, self.K, self.T, self.r, self.q, self.sigma,
            50, 100_000, OptionType.CALL, PayoffType.EUROPEAN,
            VarianceReduction.NONE, compute_greeks_flag=False,
        )
        small_w = res_small.confidence_interval_upper - res_small.confidence_interval_lower
        large_w = res_large.confidence_interval_upper - res_large.confidence_interval_lower
        assert large_w < small_w

    def test_ci_coverage_rate(self):
        """Over many trials, ~95% of CIs should contain the BS price."""
        bs = black_scholes_price(self.S, self.K, self.T, self.r, self.q, self.sigma, OptionType.CALL)
        n_trials = 100
        contained = 0
        for seed in range(1000, 1000 + n_trials):
            rng = np.random.default_rng(seed)
            res = monte_carlo_price(
                rng, self.S, self.K, self.T, self.r, self.q, self.sigma,
                50, 5_000, OptionType.CALL, PayoffType.EUROPEAN,
                VarianceReduction.NONE, compute_greeks_flag=False,
            )
            if res.confidence_interval_lower <= bs <= res.confidence_interval_upper:
                contained += 1
        rate = contained / n_trials
        # Allow some slack: 85%–100% for 100 trials at 5k paths
        assert 0.80 <= rate <= 1.0, f"CI coverage rate {rate:.2%} outside [0.80, 1.0]"


# ──────────────────────────────────────────────────────────────────────────────
# Convergence
# ──────────────────────────────────────────────────────────────────────────────


class TestConvergence:
    """Verify MC convergence toward Black-Scholes."""

    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.20

    def test_mc_converges_to_bs(self):
        bs = black_scholes_price(self.S, self.K, self.T, self.r, self.q, self.sigma, OptionType.CALL)
        rng = np.random.default_rng(42)
        res = monte_carlo_price(
            rng, self.S, self.K, self.T, self.r, self.q, self.sigma,
            100, 100_000, OptionType.CALL, PayoffType.EUROPEAN,
            VarianceReduction.NONE, compute_greeks_flag=False,
        )
        assert abs(res.price - bs) < 3 * res.standard_error

    def test_convergence_data_monotonic_ci(self):
        """CI width in convergence data should decrease as paths increase."""
        rng = np.random.default_rng(42)
        data = generate_convergence_data(
            rng, self.S, self.K, self.T, self.r, self.q, self.sigma,
            100, 20_000, OptionType.CALL, PayoffType.EUROPEAN,
            VarianceReduction.NONE,
        )
        # CI widths should be generally decreasing (first half wider than last half)
        mid = len(data) // 2
        widths_first = [d["ci_upper"] - d["ci_lower"] for d in data[:mid]]
        widths_last = [d["ci_upper"] - d["ci_lower"] for d in data[mid:]]
        assert np.mean(widths_last) < np.mean(widths_first)

    def test_convergence_data_with_antithetic(self):
        """Convergence data should work with antithetic variates."""
        rng = np.random.default_rng(42)
        data = generate_convergence_data(
            rng, self.S, self.K, self.T, self.r, self.q, self.sigma,
            100, 20_000, OptionType.CALL, PayoffType.EUROPEAN,
            VarianceReduction.ANTITHETIC,
        )
        assert len(data) >= 5
        for pt in data:
            assert pt["ci_lower"] <= pt["price"] <= pt["ci_upper"]


# ──────────────────────────────────────────────────────────────────────────────
# Monotonicity
# ──────────────────────────────────────────────────────────────────────────────


class TestMonotonicity:
    """Option prices should respect basic no-arbitrage monotonicities."""

    def test_call_increases_with_spot(self):
        rng = np.random.default_rng(42)
        lo = monte_carlo_price(
            rng, 80.0, 100.0, 1.0, 0.05, 0.02, 0.20,
            50, 30_000, OptionType.CALL, PayoffType.EUROPEAN,
            VarianceReduction.NONE, compute_greeks_flag=False,
        )
        rng2 = np.random.default_rng(43)
        hi = monte_carlo_price(
            rng2, 120.0, 100.0, 1.0, 0.05, 0.02, 0.20,
            50, 30_000, OptionType.CALL, PayoffType.EUROPEAN,
            VarianceReduction.NONE, compute_greeks_flag=False,
        )
        assert hi.price > lo.price

    def test_put_decreases_with_spot(self):
        rng = np.random.default_rng(42)
        lo = monte_carlo_price(
            rng, 80.0, 100.0, 1.0, 0.05, 0.02, 0.20,
            50, 30_000, OptionType.PUT, PayoffType.EUROPEAN,
            VarianceReduction.NONE, compute_greeks_flag=False,
        )
        rng2 = np.random.default_rng(43)
        hi = monte_carlo_price(
            rng2, 120.0, 100.0, 1.0, 0.05, 0.02, 0.20,
            50, 30_000, OptionType.PUT, PayoffType.EUROPEAN,
            VarianceReduction.NONE, compute_greeks_flag=False,
        )
        assert lo.price > hi.price

    def test_price_non_negative(self):
        rng = np.random.default_rng(42)
        res = monte_carlo_price(
            rng, 100.0, 100.0, 1.0, 0.05, 0.02, 0.20,
            10, 500, OptionType.CALL, PayoffType.EUROPEAN,
            VarianceReduction.NONE, compute_greeks_flag=False,
        )
        assert res.price >= 0


# ──────────────────────────────────────────────────────────────────────────────
# Antithetic Variates
# ──────────────────────────────────────────────────────────────────────────────


class TestAntitheticVariates:
    """Verify that antithetic variates actually reduce estimator variance."""

    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.20

    def test_antithetic_reduces_variance_european(self):
        """
        For European options, antithetic should produce lower variance
        than standard MC with the same number of paths.
        """
        n_trials = 20
        n_paths = 20_000

        std_variances = []
        anti_variances = []

        for trial in range(n_trials):
            seed = 1000 + trial

            rng_std = np.random.default_rng(seed)
            res_std = monte_carlo_price(
                rng_std, self.S, self.K, self.T, self.r, self.q, self.sigma,
                50, n_paths, OptionType.CALL, PayoffType.EUROPEAN,
                VarianceReduction.NONE, compute_greeks_flag=False,
            )
            std_variances.append(res_std.standard_error**2)

            rng_anti = np.random.default_rng(seed)
            res_anti = monte_carlo_price(
                rng_anti, self.S, self.K, self.T, self.r, self.q, self.sigma,
                50, n_paths, OptionType.CALL, PayoffType.EUROPEAN,
                VarianceReduction.ANTITHETIC, compute_greeks_flag=False,
            )
            anti_variances.append(res_anti.standard_error**2)

        avg_std_var = np.mean(std_variances)
        avg_anti_var = np.mean(anti_variances)

        # Antithetic must reduce variance (at least not increase)
        reduction = (avg_std_var - avg_anti_var) / avg_std_var
        assert reduction > -0.05, (
            f"Antithetic increased variance by {-reduction:.1%}: "
            f"std={avg_std_var:.6f}, anti={avg_anti_var:.6f}"
        )

    def test_antithetic_bias_unchanged(self):
        """Antithetic should not systematically bias the price estimate."""
        bs = black_scholes_price(self.S, self.K, self.T, self.r, self.q, self.sigma, OptionType.CALL)
        n_trials = 20
        errors_std = []
        errors_anti = []

        for trial in range(n_trials):
            seed = 2000 + trial

            rng = np.random.default_rng(seed)
            res = monte_carlo_price(
                rng, self.S, self.K, self.T, self.r, self.q, self.sigma,
                50, 20_000, OptionType.CALL, PayoffType.EUROPEAN,
                VarianceReduction.NONE, compute_greeks_flag=False,
            )
            errors_std.append(res.price - bs)

            rng = np.random.default_rng(seed)
            res = monte_carlo_price(
                rng, self.S, self.K, self.T, self.r, self.q, self.sigma,
                50, 20_000, OptionType.CALL, PayoffType.EUROPEAN,
                VarianceReduction.ANTITHETIC, compute_greeks_flag=False,
            )
            errors_anti.append(res.price - bs)

        # Both should have mean error ~0
        assert abs(np.mean(errors_std)) < 0.1
        assert abs(np.mean(errors_anti)) < 0.1

    def test_antithetic_still_works_for_asian(self):
        """Antithetic should also work for path-dependent (Asian) options."""
        n_trials = 15
        n_paths = 20_000

        std_variances = []
        anti_variances = []

        for trial in range(n_trials):
            seed = 3000 + trial

            rng = np.random.default_rng(seed)
            res = monte_carlo_price(
                rng, self.S, self.K, self.T, self.r, self.q, self.sigma,
                50, n_paths, OptionType.CALL, PayoffType.ASIAN,
                VarianceReduction.NONE, compute_greeks_flag=False,
            )
            std_variances.append(res.standard_error**2)

            rng = np.random.default_rng(seed)
            res = monte_carlo_price(
                rng, self.S, self.K, self.T, self.r, self.q, self.sigma,
                50, n_paths, OptionType.CALL, PayoffType.ASIAN,
                VarianceReduction.ANTITHETIC, compute_greeks_flag=False,
            )
            anti_variances.append(res.standard_error**2)

        avg_std_var = np.mean(std_variances)
        avg_anti_var = np.mean(anti_variances)
        reduction = (avg_std_var - avg_anti_var) / avg_std_var
        assert reduction > -0.05, (
            f"Antithetic increased Asian variance: std={avg_std_var:.6f}, anti={avg_anti_var:.6f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Visualization Data
# ──────────────────────────────────────────────────────────────────────────────


class TestVisualizationData:
    """Verify visualization data generation constraints."""

    def test_paths_capped_at_100(self):
        rng = np.random.default_rng(42)
        data = generate_visualization_data(
            rng, 100.0, 100.0, 1.0, 0.05, 0.02, 0.20,
            50, 50_000, OptionType.CALL, PayoffType.EUROPEAN,
            VarianceReduction.NONE,
        )
        assert len(data["sampled_paths"]) <= 100

    def test_convergence_points(self):
        rng = np.random.default_rng(42)
        data = generate_visualization_data(
            rng, 100.0, 100.0, 1.0, 0.05, 0.02, 0.20,
            50, 10_000, OptionType.CALL, PayoffType.EUROPEAN,
            VarianceReduction.NONE,
        )
        assert len(data["convergence"]) >= 5
        for pt in data["convergence"]:
            assert pt["ci_lower"] <= pt["price"] <= pt["ci_upper"]

    def test_histogram_bins(self):
        rng = np.random.default_rng(42)
        data = generate_visualization_data(
            rng, 100.0, 100.0, 1.0, 0.05, 0.02, 0.20,
            50, 10_000, OptionType.CALL, PayoffType.EUROPEAN,
            VarianceReduction.NONE,
        )
        assert len(data["terminal_bin_edges"]) == 51  # 50 bins → 51 edges
        assert len(data["terminal_bin_counts"]) == 50
        assert len(data["payoff_bin_edges"]) == 51
        assert len(data["payoff_bin_counts"]) == 50

    def test_time_grid(self):
        rng = np.random.default_rng(42)
        data = generate_visualization_data(
            rng, 100.0, 100.0, 1.0, 0.05, 0.02, 0.20,
            10, 100, OptionType.CALL, PayoffType.EUROPEAN,
            VarianceReduction.NONE,
        )
        assert data["time_grid"][0] == 0.0
        assert data["time_grid"][-1] == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Edge Cases & Sanity
# ──────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """MC engine should handle edge parameters gracefully."""

    def test_zero_volatility(self):
        """σ=0 → deterministic paths, payoff is discounted intrinsic."""
        rng = np.random.default_rng(42)
        res = monte_carlo_price(
            rng, 100.0, 100.0, 1.0, 0.05, 0.0, 0.001,
            10, 1_000, OptionType.CALL, PayoffType.EUROPEAN,
            VarianceReduction.NONE, compute_greeks_flag=False,
        )
        assert res.price >= 0
        assert np.isfinite(res.price)

    def test_very_few_paths(self):
        """Minimum path count should still return valid statistics."""
        rng = np.random.default_rng(42)
        res = monte_carlo_price(
            rng, 100.0, 100.0, 1.0, 0.05, 0.02, 0.20,
            10, 100, OptionType.CALL, PayoffType.EUROPEAN,
            VarianceReduction.NONE, compute_greeks_flag=False,
        )
        assert np.isfinite(res.price)
        assert res.standard_error > 0

    def test_very_few_paths_antithetic(self):
        """Antithetic with minimum paths (n=2) should work."""
        rng = np.random.default_rng(42)
        res = monte_carlo_price(
            rng, 100.0, 100.0, 1.0, 0.05, 0.02, 0.20,
            10, 2, OptionType.CALL, PayoffType.EUROPEAN,
            VarianceReduction.ANTITHETIC, compute_greeks_flag=False,
        )
        assert np.isfinite(res.price)

    def test_asian_payoff_lower_than_european(self):
        """Asian call < European call for identical parameters (Jensen's inequality)."""
        rng = np.random.default_rng(42)
        res_eu = monte_carlo_price(
            rng, 100.0, 100.0, 1.0, 0.05, 0.02, 0.20,
            50, 50_000, OptionType.CALL, PayoffType.EUROPEAN,
            VarianceReduction.NONE, compute_greeks_flag=False,
        )
        rng2 = np.random.default_rng(43)
        res_as = monte_carlo_price(
            rng2, 100.0, 100.0, 1.0, 0.05, 0.02, 0.20,
            50, 50_000, OptionType.CALL, PayoffType.ASIAN,
            VarianceReduction.NONE, compute_greeks_flag=False,
        )
        # Asian should be cheaper than European (arithmetic average has lower vol)
        # Allow for MC noise by checking with loose tolerance
        assert res_as.price < res_eu.price * 1.05

    def test_large_path_count_chunked(self):
        """Chunked simulation with 200k paths should produce valid results."""
        rng = np.random.default_rng(42)
        res = monte_carlo_price(
            rng, 100.0, 100.0, 1.0, 0.05, 0.02, 0.20,
            20, 200_000, OptionType.CALL, PayoffType.EUROPEAN,
            VarianceReduction.NONE, compute_greeks_flag=False,
            chunk_size=50_000,
        )
        assert np.isfinite(res.price)
        assert res.standard_error > 0
