"""GPU-accelerated Monte Carlo simulation using PyTorch CUDA."""

import time

import numpy as np
import torch

from optionforge.models.types import OptionType, PayoffType, VarianceReduction


def _gbm_gpu(
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
    discount: float,
    seed: int,
) -> tuple[float, float, float]:
    """Run a full MC pricing on GPU and return (price, std_error, elapsed_ms)."""
    device = torch.device("cuda")
    rng = torch.Generator(device=device).manual_seed(seed)

    dt = maturity / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    t0 = time.perf_counter()

    # Generate all normals on GPU and exponentiate
    z = torch.randn(n_paths, n_steps, generator=rng, device=device, dtype=torch.float64)
    increments = torch.exp(drift + vol * z)
    # Cumulative product along time axis
    paths = spot * torch.cumprod(increments, dim=1)

    # Payoff
    if payoff_type == PayoffType.EUROPEAN:
        terminal = paths[:, -1]
    else:
        terminal = torch.mean(paths, dim=1)

    if option_type == OptionType.CALL:
        payoffs = torch.clamp(terminal - strike, min=0.0)
    else:
        payoffs = torch.clamp(strike - terminal, min=0.0)

    discounted = discount * payoffs

    # Statistics on GPU
    price = float(torch.mean(discounted).cpu())
    std_val = float(torch.std(discounted, unbiased=True).cpu())
    n = len(discounted)
    standard_error = std_val / np.sqrt(n) if n > 1 else 0.0

    # Sync to get accurate GPU timing
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return price, standard_error, elapsed_ms


def run_benchmark(
    spot: float = 100.0,
    strike: float = 100.0,
    maturity: float = 1.0,
    r: float = 0.05,
    q: float = 0.02,
    sigma: float = 0.2,
    n_steps: int = 252,
    n_paths: int = 500_000,
    option_type: str = "call",
    payoff_type: str = "european",
    seed: int = 42,
) -> dict:
    """
    Run identical MC pricing on CPU (NumPy) and GPU (PyTorch),
    returning timing, price, and SE for comparison.
    """
    from optionforge.pricing.monte_carlo import monte_carlo_price

    opt = OptionType(option_type)
    payoff = PayoffType(payoff_type)
    discount = np.exp(-r * maturity)

    # ── CPU (NumPy) ──
    t0 = time.perf_counter()
    cpu_rng = np.random.default_rng(seed)
    cpu_result = monte_carlo_price(
        cpu_rng, spot, strike, maturity, r, q, sigma,
        n_steps, n_paths, opt, payoff, VarianceReduction.NONE,
        compute_greeks_flag=False,
    )
    cpu_ms = (time.perf_counter() - t0) * 1000.0

    # ── GPU (PyTorch CUDA) ──
    if not torch.cuda.is_available():
        return {
            "cpu_ms": round(cpu_ms, 1),
            "cpu_price": round(cpu_result.price, 6),
            "cpu_se": round(cpu_result.standard_error, 6),
            "gpu_ms": None,
            "gpu_price": None,
            "gpu_se": None,
            "speedup": None,
            "n_paths": n_paths,
        }

    # Warmup GPU
    _gbm_gpu(spot, strike, maturity, r, q, sigma, 10, 1000, opt, payoff, discount, seed)
    torch.cuda.synchronize()

    gpu_price, gpu_se, gpu_ms = _gbm_gpu(
        spot, strike, maturity, r, q, sigma,
        n_steps, n_paths, opt, payoff, discount, seed,
    )

    return {
        "cpu_ms": round(cpu_ms, 1),
        "cpu_price": round(cpu_result.price, 6),
        "cpu_se": round(cpu_result.standard_error, 6),
        "gpu_ms": round(gpu_ms, 1),
        "gpu_price": round(gpu_price, 6),
        "gpu_se": round(gpu_se, 6),
        "speedup": round(cpu_ms / gpu_ms, 1) if gpu_ms > 0 else None,
        "n_paths": n_paths,
    }
