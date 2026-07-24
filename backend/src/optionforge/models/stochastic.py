"""Geometric Brownian Motion simulation with chunked execution."""

import numpy as np
from numpy.random import Generator


def simulate_gbm_paths(
    rng: Generator,
    spot: float,
    r: float,
    q: float,
    sigma: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    chunk_size: int = 50_000,
) -> np.ndarray:
    """
    Simulate GBM price paths using risk-neutral dynamics:

        S(t+dt) = S(t) * exp((r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    Returns an array of shape (n_paths, n_steps + 1) where column 0 is spot.
    Uses chunked simulation to limit peak memory for large path counts.
    """
    dt = maturity / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    # Pre-allocate: first column = spot, rest simulated
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = spot

    if n_paths <= chunk_size:
        # Single chunk
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

    return paths


def simulate_heston_paths(
    rng: Generator,
    spot: float,
    r: float,
    q: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
    chunk_size: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate Heston stochastic volatility paths.

    Dynamics (risk-neutral):
        dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW^S_t
        dv_t = κ(θ - v_t) dt + ξ sqrt(v_t) dW^v_t
        Corr(dW^S_t, dW^v_t) = ρ

    Uses Euler-Maruyama discretization with full truncation
    (Lord, Koekkoek & van Dijk 2010) to keep variance non-negative.

    Returns:
        paths: (n_paths, n_steps+1) asset price paths
        variance: (n_paths, n_steps+1) variance paths
    """
    dt = maturity / n_steps
    sqrt_dt = np.sqrt(dt)

    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    variance = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = spot
    variance[:, 0] = v0

    if n_paths <= chunk_size:
        # Generate correlated normals
        z_s = rng.normal(0.0, 1.0, size=(n_paths, n_steps))
        z_v = rng.normal(0.0, 1.0, size=(n_paths, n_steps))
        # Correlate: Z_S = ρ Z_v + sqrt(1-ρ²) Z_s
        z_corr = rho * z_v + np.sqrt(max(1.0 - rho**2, 0.0)) * z_s

        v = v0
        s = spot
        for t in range(n_steps):
            # Full truncation: use v⁺ = max(v, 0)
            v_plus = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_plus)

            v = v + kappa * (theta - v_plus) * dt + xi * sqrt_v * sqrt_dt * z_v[:, t]
            v = np.maximum(v, 0.0)  # reflect at zero

            s = s * np.exp(
                (r - q - 0.5 * v_plus) * dt + sqrt_v * sqrt_dt * z_corr[:, t]
            )

            variance[:, t + 1] = v
            paths[:, t + 1] = s
    else:
        n_chunks = (n_paths + chunk_size - 1) // chunk_size
        for c in range(n_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, n_paths)
            chunk_n = end - start

            z_s = rng.normal(0.0, 1.0, size=(chunk_n, n_steps))
            z_v = rng.normal(0.0, 1.0, size=(chunk_n, n_steps))
            z_corr = rho * z_v + np.sqrt(max(1.0 - rho**2, 0.0)) * z_s

            v = np.full(chunk_n, v0, dtype=np.float64)
            s = np.full(chunk_n, spot, dtype=np.float64)

            for t in range(n_steps):
                v_plus = np.maximum(v, 0.0)
                sqrt_v = np.sqrt(v_plus)

                v = v + kappa * (theta - v_plus) * dt + xi * sqrt_v * sqrt_dt * z_v[:, t]
                v = np.maximum(v, 0.0)

                s = s * np.exp(
                    (r - q - 0.5 * v_plus) * dt + sqrt_v * sqrt_dt * z_corr[:, t]
                )

                variance[start:end, t + 1] = v
                paths[start:end, t + 1] = s

    return paths, variance
