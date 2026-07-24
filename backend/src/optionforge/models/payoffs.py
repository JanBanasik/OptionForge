"""Option payoff functions for European, Asian, and Barrier options."""

import numpy as np

from optionforge.models.types import BarrierType, OptionType, PayoffType


def compute_payoff(
    paths: np.ndarray,
    strike: float,
    option_type: OptionType,
    payoff_type: PayoffType,
    barrier_type: BarrierType | None = None,
    barrier_level: float = 0.0,
) -> np.ndarray:
    """
    Compute undiscounted payoff for each path.

    Args:
        paths: (n_paths, n_steps+1) array of asset prices.
        strike: Strike price.
        option_type: CALL or PUT.
        payoff_type: EUROPEAN, ASIAN, or BARRIER.
        barrier_type: Type of barrier (required if payoff_type == BARRIER).
        barrier_level: Barrier price level.

    Returns:
        (n_paths,) array of undiscounted payoffs.
    """
    if payoff_type == PayoffType.EUROPEAN:
        terminal_prices = paths[:, -1]
    elif payoff_type == PayoffType.ASIAN:
        terminal_prices = np.mean(paths[:, 1:], axis=1)
    else:
        # Barrier: use terminal prices, but apply knock-in/out logic
        terminal_prices = paths[:, -1]

    # Base payoff (European-style on terminal/average)
    if option_type == OptionType.CALL:
        base_payoff = np.maximum(terminal_prices - strike, 0.0)
    else:
        base_payoff = np.maximum(strike - terminal_prices, 0.0)

    # Barrier knock-in/out logic
    if payoff_type == PayoffType.BARRIER and barrier_type is not None:
        # Monitor the path for barrier crossing (exclude S₀ at index 0)
        path_prices = paths[:, 1:]
        path_max = np.max(path_prices, axis=1)
        path_min = np.min(path_prices, axis=1)

        if barrier_type == BarrierType.UP_AND_OUT:
            knocked = path_max >= barrier_level
            base_payoff[knocked] = 0.0
        elif barrier_type == BarrierType.DOWN_AND_OUT:
            knocked = path_min <= barrier_level
            base_payoff[knocked] = 0.0
        elif barrier_type == BarrierType.UP_AND_IN:
            knocked = path_max < barrier_level
            base_payoff[knocked] = 0.0
        elif barrier_type == BarrierType.DOWN_AND_IN:
            knocked = path_min > barrier_level
            base_payoff[knocked] = 0.0

    return base_payoff
