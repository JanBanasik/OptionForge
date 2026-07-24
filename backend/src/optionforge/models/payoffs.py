"""Option payoff functions for European and Asian options."""

import numpy as np

from optionforge.models.types import OptionType, PayoffType


def compute_payoff(
    paths: np.ndarray,
    strike: float,
    option_type: OptionType,
    payoff_type: PayoffType,
) -> np.ndarray:
    """
    Compute undiscounted payoff for each path.

    Args:
        paths: (n_paths, n_steps+1) array of asset prices.
        strike: Strike price.
        option_type: CALL or PUT.
        payoff_type: EUROPEAN or ASIAN.

    Returns:
        (n_paths,) array of undiscounted payoffs.
    """
    if payoff_type == PayoffType.EUROPEAN:
        terminal_prices = paths[:, -1]
    else:
        terminal_prices = np.mean(paths[:, 1:], axis=1)  # arithmetic average

    if option_type == OptionType.CALL:
        payoff = np.maximum(terminal_prices - strike, 0.0)
    else:
        payoff = np.maximum(strike - terminal_prices, 0.0)

    return payoff
