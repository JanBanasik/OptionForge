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
