"""Pydantic request and response models for the OptionForge API."""


from pydantic import BaseModel, Field, model_validator


class PricingRequest(BaseModel):
    """Input parameters for option pricing."""

    spot: float = Field(default=100.0, gt=0, description="Current asset price")
    strike: float = Field(default=100.0, gt=0, description="Option strike price")
    maturity: float = Field(default=1.0, gt=0, description="Time to maturity in years")
    risk_free_rate: float = Field(default=0.05, ge=0, description="Risk-free interest rate")
    dividend_yield: float = Field(default=0.02, ge=0, description="Continuous dividend yield")
    volatility: float = Field(default=0.2, gt=0, le=5.0, description="Annualized volatility")
    n_paths: int = Field(default=50_000, ge=100, le=500_000, description="Number of Monte Carlo paths")
    n_steps: int = Field(default=252, ge=1, le=2_000, description="Time steps per path")
    option_type: str = Field(default="call", pattern="^(call|put)$")
    payoff_type: str = Field(default="european", pattern="^(european|asian|barrier)$")
    variance_reduction: str = Field(default="none", pattern="^(none|antithetic|control_variate)$")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    barrier_type: str | None = Field(default=None, pattern="^(up_and_out|down_and_out|up_and_in|down_and_in)$")
    barrier_level: float | None = Field(default=None, gt=0)
    model_type: str = Field(default="gbm", pattern="^(gbm|heston)$")
    heston_kappa: float | None = Field(default=None, gt=0)
    heston_theta: float | None = Field(default=None, gt=0)
    heston_xi: float | None = Field(default=None, gt=0)
    heston_rho: float | None = Field(default=None, ge=-1, le=1)
    heston_v0: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_paths_steps(self) -> "PricingRequest":
        if self.n_steps > self.n_paths * 10:
            raise ValueError("n_steps must not exceed 10x n_paths")
        return self


class GreeksResponse(BaseModel):
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class ConvergencePointResponse(BaseModel):
    n_paths: int
    price: float
    ci_lower: float
    ci_upper: float


class PricingResponse(BaseModel):
    price: float
    payoff_mean: float
    payoff_std: float
    standard_error: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    black_scholes_price: float | None = None
    absolute_error: float | None = None
    relative_error: float | None = None
    greeks: GreeksResponse | None = None
    bs_greeks: GreeksResponse | None = None


class VisualizationResponse(BaseModel):
    sampled_paths: list[list[float]]
    time_grid: list[float]
    terminal_bin_edges: list[float]
    terminal_bin_counts: list[int]
    payoff_bin_edges: list[float]
    payoff_bin_counts: list[int]
    convergence: list[ConvergencePointResponse]
    greeks: GreeksResponse | None = None
    bs_greeks: GreeksResponse | None = None


class IVRequest(BaseModel):
    """Request for implied volatility calculation."""

    market_price: float = Field(gt=0, description="Observed market price of the option")
    spot: float = Field(gt=0, description="Current asset price")
    strike: float = Field(gt=0, description="Option strike price")
    maturity: float = Field(gt=0, description="Time to maturity in years")
    risk_free_rate: float = Field(ge=0, description="Risk-free interest rate")
    dividend_yield: float = Field(ge=0, description="Continuous dividend yield")
    option_type: str = Field(pattern="^(call|put)$")


class IVResponse(BaseModel):
    implied_volatility: float
    iterations: int | None = None
    price_error: float | None = None


class VolSurfaceRequest(BaseModel):
    """Request for volatility surface data."""

    spot: float = Field(default=100.0, gt=0)
    risk_free_rate: float = Field(default=0.05, ge=0)
    dividend_yield: float = Field(default=0.02, ge=0)
    atm_vol: float = Field(default=0.2, gt=0, le=5.0)
    skew: float = Field(default=-0.05, description="Vol skew per unit moneyness (K/S - 1)")
    smile: float = Field(default=0.15, description="Vol convexity (smile curvature)")
    term: float = Field(default=0.02, description="Term structure slope per year")
    n_strikes: int = Field(default=20, ge=5, le=50)
    n_maturities: int = Field(default=10, ge=3, le=30)


class VolSurfaceResponse(BaseModel):
    strikes: list[float]
    maturities: list[float]
    iv_grid: list[list[float]]  # [n_maturities][n_strikes]
    spot: float
