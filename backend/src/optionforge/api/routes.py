"""FastAPI route handlers for OptionForge pricing endpoints."""

import numpy as np
from fastapi import APIRouter, HTTPException

from optionforge.api.schemas import (
    GreeksResponse,
    IVRequest,
    IVResponse,
    PricingRequest,
    PricingResponse,
    VisualizationResponse,
    VolSurfaceRequest,
    VolSurfaceResponse,
)
from optionforge.models.types import BarrierType, OptionType, PayoffType, VarianceReduction
from optionforge.pricing.black_scholes import black_scholes_price, implied_volatility
from optionforge.pricing.monte_carlo import generate_visualization_data, monte_carlo_price

router = APIRouter(prefix="/api")


def _to_enum(value: str, enum_cls: type) -> object:
    try:
        return enum_cls(value)
    except ValueError:
        valid = [e.value for e in enum_cls]
        raise HTTPException(
            status_code=422,
            detail=f"Invalid {enum_cls.__name__}: '{value}'. Must be one of {valid}",
        )


@router.post("/price", response_model=PricingResponse)
def price_option(req: PricingRequest) -> PricingResponse:
    """Run Monte Carlo pricing and return results with analytics."""
    option_type = _to_enum(req.option_type, OptionType)
    payoff_type = _to_enum(req.payoff_type, PayoffType)
    var_reduction = _to_enum(req.variance_reduction, VarianceReduction)
    barrier_type = _to_enum(req.barrier_type, BarrierType) if req.barrier_type else None
    barrier_level = req.barrier_level or 0.0

    seed = req.seed if req.seed is not None else np.random.default_rng().integers(0, 2**31)
    rng = np.random.default_rng(seed)

    result = monte_carlo_price(
        rng=rng,
        spot=req.spot,
        strike=req.strike,
        maturity=req.maturity,
        r=req.risk_free_rate,
        q=req.dividend_yield,
        sigma=req.volatility,
        n_steps=req.n_steps,
        n_paths=req.n_paths,
        option_type=option_type,
        payoff_type=payoff_type,
        variance_reduction=var_reduction,
        barrier_type=barrier_type,
        barrier_level=barrier_level,
    )

    return PricingResponse(
        price=round(result.price, 8),
        payoff_mean=round(result.payoff_mean, 8),
        payoff_std=round(result.payoff_std, 8),
        standard_error=round(result.standard_error, 8),
        confidence_interval_lower=round(result.confidence_interval_lower, 8),
        confidence_interval_upper=round(result.confidence_interval_upper, 8),
        black_scholes_price=round(result.black_scholes_price, 8) if result.black_scholes_price is not None else None,
        absolute_error=round(result.absolute_error, 8) if result.absolute_error is not None else None,
        relative_error=round(result.relative_error, 8) if result.relative_error is not None else None,
        greeks=GreeksResponse(
            delta=result.greeks.delta,
            gamma=result.greeks.gamma,
            vega=result.greeks.vega,
            theta=result.greeks.theta,
            rho=result.greeks.rho,
        ) if result.greeks else None,
        bs_greeks=GreeksResponse(
            delta=result.bs_greeks.delta,
            gamma=result.bs_greeks.gamma,
            vega=result.bs_greeks.vega,
            theta=result.bs_greeks.theta,
            rho=result.bs_greeks.rho,
        ) if result.bs_greeks else None,
    )


@router.post("/visualization", response_model=VisualizationResponse)
def visualization_data(req: PricingRequest) -> VisualizationResponse:
    """Generate chart data: paths, histograms, convergence, Greeks."""
    option_type = _to_enum(req.option_type, OptionType)
    payoff_type = _to_enum(req.payoff_type, PayoffType)
    var_reduction = _to_enum(req.variance_reduction, VarianceReduction)
    barrier_type = _to_enum(req.barrier_type, BarrierType) if req.barrier_type else None
    barrier_level = req.barrier_level or 0.0

    seed = req.seed if req.seed is not None else np.random.default_rng().integers(0, 2**31)
    rng = np.random.default_rng(seed)

    data = generate_visualization_data(
        rng=rng,
        spot=req.spot,
        strike=req.strike,
        maturity=req.maturity,
        r=req.risk_free_rate,
        q=req.dividend_yield,
        sigma=req.volatility,
        n_steps=req.n_steps,
        n_paths=req.n_paths,
        option_type=option_type,
        payoff_type=payoff_type,
        variance_reduction=var_reduction,
        barrier_type=barrier_type,
        barrier_level=barrier_level,
    )

    return VisualizationResponse(
        sampled_paths=data["sampled_paths"],
        time_grid=data["time_grid"],
        terminal_bin_edges=data["terminal_bin_edges"],
        terminal_bin_counts=data["terminal_bin_counts"],
        payoff_bin_edges=data["payoff_bin_edges"],
        payoff_bin_counts=data["payoff_bin_counts"],
        convergence=data["convergence"],
        greeks=GreeksResponse(**data["greeks"]) if data["greeks"] else None,
        bs_greeks=GreeksResponse(**data["bs_greeks"]) if data["bs_greeks"] else None,
    )


@router.post("/iv", response_model=IVResponse)
def compute_iv(req: IVRequest) -> IVResponse:
    """Compute implied volatility from market price via Newton-Raphson."""
    option_type = _to_enum(req.option_type, OptionType)
    result = implied_volatility(
        market_price=req.market_price,
        spot=req.spot,
        strike=req.strike,
        maturity=req.maturity,
        r=req.risk_free_rate,
        q=req.dividend_yield,
        option_type=option_type,
    )
    return IVResponse(**result)


@router.post("/vol-surface", response_model=VolSurfaceResponse)
def vol_surface(req: VolSurfaceRequest) -> VolSurfaceResponse:
    """Generate volatility surface from a parameterized smile."""
    import numpy as np

    strikes = np.linspace(
        req.spot * 0.7, req.spot * 1.3, req.n_strikes
    ).tolist()
    maturities = np.linspace(0.1, 3.0, req.n_maturities).tolist()

    iv_grid: list[list[float]] = []

    for maturity in maturities:
        row: list[float] = []
        for strike in strikes:
            moneyness = strike / req.spot - 1.0
            true_sigma = (
                req.atm_vol
                + req.skew * moneyness
                + req.smile * moneyness**2
                + req.term * (maturity - 1.0)
            )
            true_sigma = max(true_sigma, 0.01)

            price = black_scholes_price(
                req.spot, strike, maturity, req.risk_free_rate,
                req.dividend_yield, true_sigma, OptionType.CALL,
            )
            result = implied_volatility(
                price, req.spot, strike, maturity,
                req.risk_free_rate, req.dividend_yield, OptionType.CALL,
            )
            row.append(round(result["implied_volatility"], 8))
        iv_grid.append(row)

    return VolSurfaceResponse(
        strikes=strikes,
        maturities=maturities,
        iv_grid=iv_grid,
        spot=req.spot,
    )
