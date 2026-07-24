"""Variance reduction via antithetic variates.

We provide a helper that, given paths simulated with normal draws Z,
returns the corresponding antithetic paths using -Z in place of Z.

The key idea: for every path S_t = f(Z), we add S'_t = f(-Z).
The estimator (f(Z) + f(-Z))/2 has lower variance when f is monotonic
(which option payoffs are).
"""

import numpy as np


def generate_antithetic_pair(
    rng_state: np.random.Generator,
    spot: float,
    r: float,
    q: float,
    sigma: float,
    maturity: float,
    n_steps: int,
    n_pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate pairs of regular and antithetic GBM paths.

    Returns (regular_paths, antithetic_paths), each of shape (n_pairs, n_steps+1).
    """
    dt = maturity / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    z = rng_state.normal(0.0, 1.0, size=(n_pairs, n_steps))

    reg_inc = np.exp(drift + vol * z)
    anti_inc = np.exp(drift + vol * (-z))

    reg_paths = np.empty((n_pairs, n_steps + 1), dtype=np.float64)
    anti_paths = np.empty((n_pairs, n_steps + 1), dtype=np.float64)

    reg_paths[:, 0] = spot
    anti_paths[:, 0] = spot
    reg_paths[:, 1:] = spot * np.cumprod(reg_inc, axis=1)
    anti_paths[:, 1:] = spot * np.cumprod(anti_inc, axis=1)

    return reg_paths, anti_paths
