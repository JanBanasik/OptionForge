"""FastAPI route handlers for OptionForge pricing endpoints."""

import numpy as np
from fastapi import APIRouter, HTTPException

from optionforge.api.schemas import (
    GreeksResponse,
    PricingRequest,
    PricingResponse,
    VisualizationResponse,
)
from optionforge.models.types import OptionType, PayoffType, VarianceReduction
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
