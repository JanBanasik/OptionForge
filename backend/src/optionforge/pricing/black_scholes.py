"""Analytical Black-Scholes pricing with continuous dividend yield.

Uses SciPy's erf for robust CDF computation of the standard normal distribution.
"""

import numpy as np
from scipy.special import erf as _erf

from optionforge.models.types import Greeks, OptionType


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using complementary error function."""
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)


def black_scholes_price(
    spot: float,
    strike: float,
    maturity: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType,
) -> float:
    """Price a European option using the Black-Scholes-Merton formula."""
    if maturity <= 0.0:
        # At expiry, return intrinsic value
        if option_type == OptionType.CALL:
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    if sigma <= 0.0:
        # Zero volatility: deterministic forward value
        forward = spot * np.exp((r - q) * maturity)
        if option_type == OptionType.CALL:
            return max(np.exp(-r * maturity) * (forward - strike), 0.0)
        return max(np.exp(-r * maturity) * (strike - forward), 0.0)

    d1 = (np.log(spot / strike) + (r - q + 0.5 * sigma**2) * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)

    discount = np.exp(-r * maturity)
    forward = spot * np.exp(-q * maturity)

    if option_type == OptionType.CALL:
        return forward * _norm_cdf(d1) - strike * discount * _norm_cdf(d2)
    else:
        return strike * discount * _norm_cdf(-d2) - forward * _norm_cdf(-d1)


def black_scholes_greeks(
    spot: float,
    strike: float,
    maturity: float,
    r: float,
    q: float,
    sigma: float,
) -> Greeks:
    """Compute analytical Greeks for European options via Black-Scholes."""
    if maturity <= 0.0 or sigma <= 0.0:
        return Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    sqrt_t = np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (r - q + 0.5 * sigma**2) * maturity) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    discount = np.exp(-r * maturity)
    forward_discount = np.exp(-q * maturity)
    pdf_d1 = _norm_pdf(d1)

    # Delta
    delta = forward_discount * _norm_cdf(d1)

    # Gamma (same for call and put)
    gamma_numerator = forward_discount * pdf_d1
    gamma_denominator = spot * sigma * sqrt_t
    gamma = gamma_numerator / gamma_denominator if gamma_denominator > 1e-16 else 0.0

    # Vega (per 1% = 0.01 change in sigma)
    vega = spot * forward_discount * pdf_d1 * sqrt_t * 0.01

    # Theta (per day = 1/365 year)
    term1 = -(spot * forward_discount * pdf_d1 * sigma) / (2.0 * sqrt_t)
    term2 = -r * strike * discount * _norm_cdf(d2)
    term3 = q * spot * forward_discount * _norm_cdf(d1)
    theta = (term1 + term2 + term3) / 365.0

    # Rho (per 1% = 0.01 change in r)
    rho = strike * maturity * discount * _norm_cdf(d2) * 0.01

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
