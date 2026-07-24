"""Monte Carlo pricing engine with standard and antithetic sampling."""

import numpy as np
from numpy.random import Generator

from optionforge.models.payoffs import compute_payoff
from optionforge.models.types import (
    OptionType,
    PayoffType,
    PricingResult,
    VarianceReduction,
)
from optionforge.pricing.black_scholes import black_scholes_greeks, black_scholes_price


def _simulate_standard(
    rng: Generator,
    spot: float,
    strike: float,
    maturity: float,
    r: float,
    q: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    option_type: OptionType,
    payoff_type: PayoffType,
    discount: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Standard Monte Carlo: generate paths and compute discounted payoffs."""
    dt = maturity / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = spot

    if n_paths <= chunk_size:
        z = rng.normal(0.0, 1.0, size=(n_paths, n_steps))
        increments = np.exp(drift + vol * z)
        paths[:, 1:] = spot * np.cumprod(increments, axis=1)
    else:
        n_chunks = (n_paths + chunk_size - 1) // chunk_size
        for c in range(n_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, n_paths)
            chunk_n = end - start
            z = rng.normal(0.0, 1.0, size=(chunk_n, n_steps))
            increments = np.exp(drift + vol * z)
            paths[start:end, 1:] = spot * np.cumprod(increments, axis=1)

    undiscounted = compute_payoff(paths, strike, option_type, payoff_type)
    discounted = discount * undiscounted
    return paths, discounted


def _simulate_antithetic_chunked(
    rng: Generator,
    spot: float,
    strike: float,
    maturity: float,
    r: float,
    q: float,
    sigma: float,
    n_steps: int,
    n_pairs: int,
    option_type: OptionType,
    payoff_type: PayoffType,
    discount: float,
    chunk_size: int,
) -> np.ndarray:
    """
    Antithetic variates simulation.

    For each of n_pairs standard normal draws Z, generate both the regular path
    using +Z and the antithetic path using −Z.  The estimator for each pair is
    the average of the two discounted payoffs.  This produces n_pairs independent
    estimates with lower variance than standard Monte Carlo.

    Returns averaged discounted payoffs of shape (n_pairs,).
    """
    dt = maturity / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    averaged = np.empty(n_pairs, dtype=np.float64)

    if n_pairs <= chunk_size:
        z = rng.normal(0.0, 1.0, size=(n_pairs, n_steps))

        # Regular paths (+Z)
        reg_inc = np.exp(drift + vol * z)
        reg_paths = np.empty((n_pairs, n_steps + 1), dtype=np.float64)
        reg_paths[:, 0] = spot
        reg_paths[:, 1:] = spot * np.cumprod(reg_inc, axis=1)

        # Antithetic paths (−Z)
        anti_inc = np.exp(drift + vol * (-z))
        anti_paths = np.empty((n_pairs, n_steps + 1), dtype=np.float64)
        anti_paths[:, 0] = spot
        anti_paths[:, 1:] = spot * np.cumprod(anti_inc, axis=1)

        reg_disc = discount * compute_payoff(reg_paths, strike, option_type, payoff_type)
        anti_disc = discount * compute_payoff(anti_paths, strike, option_type, payoff_type)
        averaged = 0.5 * (reg_disc + anti_disc)
    else:
        n_chunks = (n_pairs + chunk_size - 1) // chunk_size
        for c in range(n_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, n_pairs)
            chunk_n = end - start

            z = rng.normal(0.0, 1.0, size=(chunk_n, n_steps))

            reg_inc = np.exp(drift + vol * z)
            reg_paths = np.empty((chunk_n, n_steps + 1), dtype=np.float64)
            reg_paths[:, 0] = spot
            reg_paths[:, 1:] = spot * np.cumprod(reg_inc, axis=1)

            anti_inc = np.exp(drift + vol * (-z))
            anti_paths = np.empty((chunk_n, n_steps + 1), dtype=np.float64)
            anti_paths[:, 0] = spot
            anti_paths[:, 1:] = spot * np.cumprod(anti_inc, axis=1)

            reg_disc = discount * compute_payoff(reg_paths, strike, option_type, payoff_type)
            anti_disc = discount * compute_payoff(anti_paths, strike, option_type, payoff_type)
            averaged[start:end] = 0.5 * (reg_disc + anti_disc)

    return averaged


def _compute_statistics(discounted: np.ndarray) -> dict:
    """Compute price, std, SE, CI from discounted payoffs."""
    n = len(discounted)
    price = float(np.mean(discounted))
    payoff_std = float(np.std(discounted, ddof=1))
    standard_error = payoff_std / np.sqrt(n)
    ci_half = 1.96 * standard_error
    return {
        "price": price,
        "payoff_std": payoff_std,
        "standard_error": standard_error,
        "confidence_interval_lower": price - ci_half,
        "confidence_interval_upper": price + ci_half,
    }


def monte_carlo_price(
    rng: Generator,
    spot: float,
    strike: float,
    maturity: float,
    r: float,
    q: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    option_type: OptionType,
    payoff_type: PayoffType,
    variance_reduction: VarianceReduction,
    compute_greeks_flag: bool = True,
    chunk_size: int = 50_000,
) -> PricingResult:
    """
    Price an option via Monte Carlo simulation.

    Supports standard sampling and antithetic variates for variance reduction.

    Args:
        rng: Seeded NumPy random generator for reproducibility.
        spot: Current asset price (S₀).
        strike: Option strike price (K).
        maturity: Time to maturity in years (T).
        r: Risk-free interest rate (continuous).
        q: Dividend yield (continuous).
        sigma: Annualized volatility (σ).
        n_steps: Number of time steps per path.
        n_paths: Number of Monte Carlo paths.
        option_type: CALL or PUT.
        payoff_type: EUROPEAN or ASIAN.
        variance_reduction: NONE or ANTITHETIC.
        compute_greeks_flag: Whether to compute finite-difference Greeks.
        chunk_size: Max paths per chunk for memory control.

    Returns:
        PricingResult with price, statistics, BS benchmark, and Greeks.
    """
    discount = np.exp(-r * maturity)

    # --- Simulate ---
    if variance_reduction == VarianceReduction.ANTITHETIC:
        # Generate n_paths/2 pairs of (Z, −Z), each averaged → n_paths/2 estimates
        n_pairs = max(n_paths // 2, 1)
        discounted = _simulate_antithetic_chunked(
            rng=rng,
            spot=spot,
            strike=strike,
            maturity=maturity,
            r=r,
            q=q,
            sigma=sigma,
            n_steps=n_steps,
            n_pairs=n_pairs,
            option_type=option_type,
            payoff_type=payoff_type,
            discount=discount,
            chunk_size=chunk_size,
        )
        payoff_mean_disc = float(np.mean(discounted))
    else:
        _, discounted = _simulate_standard(
            rng=rng,
            spot=spot,
            strike=strike,
            maturity=maturity,
            r=r,
            q=q,
            sigma=sigma,
            n_steps=n_steps,
            n_paths=n_paths,
            option_type=option_type,
            payoff_type=payoff_type,
            discount=discount,
            chunk_size=chunk_size,
        )
        payoff_mean_disc = float(np.mean(discounted))

    stats = _compute_statistics(discounted)

    # --- Black-Scholes benchmark (European only) ---
    bs_price = None
    abs_err = None
    rel_err = None
    bs_greeks = None

    if payoff_type == PayoffType.EUROPEAN:
        bs_price = black_scholes_price(spot, strike, maturity, r, q, sigma, option_type)
        abs_err = stats["price"] - bs_price
        rel_err = abs_err / bs_price if bs_price > 1e-12 else None
        bs_greeks = black_scholes_greeks(spot, strike, maturity, r, q, sigma)

    # --- Greeks ---
    mc_greeks = None
    if compute_greeks_flag:
        from optionforge.pricing.greeks import compute_greeks

        mc_greeks = compute_greeks(
            rng=rng,
            spot=spot,
            strike=strike,
            maturity=maturity,
            r=r,
            q=q,
            sigma=sigma,
            n_steps=n_steps,
            n_paths=n_paths,
            option_type=option_type,
            payoff_type=payoff_type,
            variance_reduction=variance_reduction,
        )

    return PricingResult(
        price=stats["price"],
        payoff_mean=payoff_mean_disc / discount,
        payoff_std=stats["payoff_std"],
        standard_error=stats["standard_error"],
        confidence_interval_lower=stats["confidence_interval_lower"],
        confidence_interval_upper=stats["confidence_interval_upper"],
        black_scholes_price=bs_price,
        absolute_error=abs_err,
        relative_error=rel_err,
        greeks=mc_greeks,
        bs_greeks=bs_greeks,
    )


def generate_convergence_data(
    rng: Generator,
    spot: float,
    strike: float,
    maturity: float,
    r: float,
    q: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    option_type: OptionType,
    payoff_type: PayoffType,
    variance_reduction: VarianceReduction,
    num_points: int = 40,
) -> list[dict]:
    """
    Generate convergence series showing how the price estimate and CI
    evolve as the number of paths increases.

    Uses a single large simulation, then cumulatively averages at
    logarithmically-spaced intervals to avoid re-simulating.
    """
    discount = np.exp(-r * maturity)

    # Simulate all paths once
    if variance_reduction == VarianceReduction.ANTITHETIC:
        n_pairs = max(n_paths // 2, 1)
        discounted = _simulate_antithetic_chunked(
            rng=rng,
            spot=spot,
            strike=strike,
            maturity=maturity,
            r=r,
            q=q,
            sigma=sigma,
            n_steps=n_steps,
            n_pairs=n_pairs,
            option_type=option_type,
            payoff_type=payoff_type,
            discount=discount,
            chunk_size=50_000,
        )
    else:
        _, discounted = _simulate_standard(
            rng=rng,
            spot=spot,
            strike=strike,
            maturity=maturity,
            r=r,
            q=q,
            sigma=sigma,
            n_steps=n_steps,
            n_paths=n_paths,
            option_type=option_type,
            payoff_type=payoff_type,
            discount=discount,
            chunk_size=50_000,
        )

    # For antithetic each element is a pair-average (≡ 2 paths).
    multiplier = 2 if variance_reduction == VarianceReduction.ANTITHETIC else 1
    n_estimates = len(discounted)

    path_counts = np.logspace(
        np.log10(max(100, n_estimates // num_points)),
        np.log10(n_estimates),
        num_points,
        dtype=int,
    )
    path_counts = np.unique(np.clip(path_counts, 100, n_estimates))

    convergence = []
    for pc in path_counts:
        cumulative = discounted[:pc]
        mean = float(np.mean(cumulative))
        se = float(np.std(cumulative, ddof=1)) / np.sqrt(pc)
        ci_half = 1.96 * se
        convergence.append({
            "n_paths": int(pc * multiplier),
            "price": round(mean, 8),
            "ci_lower": round(mean - ci_half, 8),
            "ci_upper": round(mean + ci_half, 8),
        })

    return convergence


def generate_visualization_data(
    rng: Generator,
    spot: float,
    strike: float,
    maturity: float,
    r: float,
    q: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    option_type: OptionType,
    payoff_type: PayoffType,
    variance_reduction: VarianceReduction,
) -> dict:
    """
    Generate visualization data: sampled paths, histograms, convergence,
    and Greeks. Paths are capped at 100 to keep payload sizes small.
    """
    dt = maturity / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    discount = np.exp(-r * maturity)

    # Sample paths for display (at most 100), using a fresh sub-generator
    # so display paths don't consume the main rng state
    display_rng = np.random.default_rng(rng.integers(0, 2**31))
    display_n = min(n_paths, 100)
    z_display = display_rng.normal(0.0, 1.0, size=(display_n, n_steps))
    inc_display = np.exp(drift + vol * z_display)
    paths_display = np.empty((display_n, n_steps + 1), dtype=np.float64)
    paths_display[:, 0] = spot
    paths_display[:, 1:] = spot * np.cumprod(inc_display, axis=1)

    # Simulate payoff data for histogram.
    # Always use standard (non-averaged) paths so the histogram shows the
    # true payoff distribution, regardless of the variance reduction method.
    _, undiscounted_payoffs = _simulate_standard(
        rng=rng,
        spot=spot,
        strike=strike,
        maturity=maturity,
        r=r,
        q=q,
        sigma=sigma,
        n_steps=n_steps,
        n_paths=n_paths,
        option_type=option_type,
        payoff_type=payoff_type,
        discount=1.0,
        chunk_size=50_000,
    )
    discounted_payoffs = discount * undiscounted_payoffs

    # Terminal price histogram — we need terminal prices from paths
    # Re-simulate just terminals for histogram (lighter weight)
    term_rng = np.random.default_rng(rng.integers(0, 2**31))
    n_hist = min(n_paths, 200_000)
    z_term = term_rng.normal(0.0, 1.0, size=n_hist)
    drift_t = (r - q - 0.5 * sigma**2) * maturity
    vol_t = sigma * np.sqrt(maturity)
    terminal_prices = spot * np.exp(drift_t + vol_t * z_term)

    terminal_hist, terminal_edges = np.histogram(terminal_prices, bins=50)
    payoff_hist, payoff_edges = np.histogram(discounted_payoffs, bins=50)

    # Convergence
    convergence = generate_convergence_data(
        rng=np.random.default_rng(rng.integers(0, 2**31)),
        spot=spot, strike=strike, maturity=maturity,
        r=r, q=q, sigma=sigma, n_steps=n_steps, n_paths=n_paths,
        option_type=option_type, payoff_type=payoff_type,
        variance_reduction=variance_reduction,
    )

    # Greeks
    from optionforge.pricing.greeks import compute_greeks

    mc_greeks = compute_greeks(
        rng=np.random.default_rng(rng.integers(0, 2**31)),
        spot=spot, strike=strike, maturity=maturity,
        r=r, q=q, sigma=sigma, n_steps=n_steps, n_paths=n_paths,
        option_type=option_type, payoff_type=payoff_type,
        variance_reduction=variance_reduction,
    )

    bs_greeks = None
    if payoff_type == PayoffType.EUROPEAN:
        bs_greeks = black_scholes_greeks(spot, strike, maturity, r, q, sigma)

    return {
        "sampled_paths": paths_display.tolist(),
        "time_grid": [round(i * dt, 8) for i in range(n_steps + 1)],
        "terminal_bin_edges": terminal_edges.tolist(),
        "terminal_bin_counts": terminal_hist.tolist(),
        "payoff_bin_edges": payoff_edges.tolist(),
        "payoff_bin_counts": payoff_hist.tolist(),
        "convergence": convergence,
        "greeks": {
            "delta": mc_greeks.delta,
            "gamma": mc_greeks.gamma,
            "vega": mc_greeks.vega,
            "theta": mc_greeks.theta,
            "rho": mc_greeks.rho,
        } if mc_greeks else None,
        "bs_greeks": {
            "delta": bs_greeks.delta,
            "gamma": bs_greeks.gamma,
            "vega": bs_greeks.vega,
            "theta": bs_greeks.theta,
            "rho": bs_greeks.rho,
        } if bs_greeks else None,
    }
