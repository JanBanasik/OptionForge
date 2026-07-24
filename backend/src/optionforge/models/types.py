"""Domain types for OptionForge pricing engine."""

from dataclasses import dataclass
from enum import Enum


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


class PayoffType(Enum):
    EUROPEAN = "european"
    ASIAN = "asian"


class VarianceReduction(Enum):
    NONE = "none"
    ANTITHETIC = "antithetic"
    CONTROL_VARIATE = "control_variate"


@dataclass
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


@dataclass
class PricingResult:
    """Container for Monte Carlo pricing output."""

    price: float
    payoff_mean: float
    payoff_std: float
    standard_error: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    black_scholes_price: float | None = None
    absolute_error: float | None = None
    relative_error: float | None = None
    greeks: Greeks | None = None
    bs_greeks: Greeks | None = None


@dataclass
class ConvergencePoint:
    n_paths: int
    price: float
    ci_lower: float
    ci_upper: float


@dataclass
class VisualizationData:
    """Lightweight visualization data with capped paths and pre-computed histograms."""

    # Sampled price paths (at most 100 for bandwidth)
    sampled_paths: list  # list of lists of floats; time_steps x sampled_paths_count
    time_grid: list[float]

    # Terminal price histogram data
    terminal_bin_edges: list[float]
    terminal_bin_counts: list[int]

    # Payoff histogram data
    payoff_bin_edges: list[float]
    payoff_bin_counts: list[int]

    # Convergence series
    convergence: list[dict]  # each dict: {n_paths, price, ci_lower, ci_upper}

    # Greeks if computed
    greeks: dict | None = None
    bs_greeks: dict | None = None
