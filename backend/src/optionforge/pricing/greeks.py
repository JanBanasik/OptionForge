"""Greeks computation via central finite differences with common random numbers.

All bumps reuse the same random seed so that path-level noise cancels
in the finite-difference estimates, producing lower-variance Greeks.
"""

import numpy as np
from numpy.random import Generator

from optionforge.models.types import Greeks, OptionType, PayoffType, VarianceReduction
from optionforge.pricing.monte_carlo import monte_carlo_price


def compute_greeks(
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
) -> Greeks:
    """
    Compute Greeks via central finite differences using the full MC engine.

    All bumps share a single base seed so that the same random numbers
    drive every simulation.  This common-random-numbers technique cancels
    path noise in the finite-difference numerators, tightening the
    Greek estimates.

    Delta, Gamma  — spot ± ε·S
    Vega          — σ ± ε·σ   (reported per +1 pp change)
    Theta         — T ± 1 day (reported per day)
    Rho           — r ± ε·r   (reported per +1 pp change)
    """
    eps = 1e-4
    base_seed = rng.integers(0, 2**31)

    def _price(s: float, sig: float, t: float, rate: float) -> float:
        local_rng = np.random.default_rng(base_seed)
        result = monte_carlo_price(
            rng=local_rng,
            spot=s,
            strike=strike,
            maturity=t,
            r=rate,
            q=q,
            sigma=sig,
            n_steps=n_steps,
            n_paths=n_paths,
            option_type=option_type,
            payoff_type=payoff_type,
            variance_reduction=variance_reduction,
            compute_greeks_flag=False,  # prevent recursion
        )
        return result.price

    # --- Delta & Gamma (spot bumps) ---
    spot_up = spot * (1.0 + eps)
    spot_down = spot * (1.0 - eps)

    p_up = _price(spot_up, sigma, maturity, r)
    p_down = _price(spot_down, sigma, maturity, r)
    p_center = _price(spot, sigma, maturity, r)

    delta = (p_up - p_down) / (2.0 * eps * spot)
    gamma = (p_up - 2.0 * p_center + p_down) / (eps * spot) ** 2

    # --- Vega (sigma bump, reported per +1 pp) ---
    sigma_up = sigma * (1.0 + eps)
    sigma_down = max(sigma * (1.0 - eps), 1e-6)

    p_sig_up = _price(spot, sigma_up, maturity, r)
    p_sig_down = _price(spot, sigma_down, maturity, r)

    vega = (p_sig_up - p_sig_down) / (2.0 * eps * sigma) * 0.01

    # --- Theta (time decay per day) ---
    day = 1.0 / 365.0
    t_up = max(maturity + day, 1e-10)
    t_down = max(maturity - day, 1e-10)

    p_t_up = _price(spot, sigma, t_up, r)
    p_t_down = _price(spot, sigma, t_down, r)

    theta = -(p_t_up - p_t_down) / (2.0 * day)

    # --- Rho (rate bump, reported per +1 pp) ---
    r_up = r * (1.0 + eps)
    r_down = max(r * (1.0 - eps), 0.0)

    p_r_up = _price(spot, sigma, maturity, r_up)
    p_r_down = _price(spot, sigma, maturity, r_down)

    rho = (p_r_up - p_r_down) / (2.0 * eps * r) * 0.01 if r > 1e-10 else 0.0

    return Greeks(
        delta=round(delta, 8),
        gamma=round(gamma, 8),
        vega=round(vega, 8),
        theta=round(theta, 8),
        rho=round(rho, 8),
    )
