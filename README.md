# OptionForge

Interactive Monte Carlo option pricing laboratory — built with FastAPI + React + Plotly.

![Python](https://img.shields.io/badge/python-3.12+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688)
![React](https://img.shields.io/badge/React-19-61dafb)
![CUDA](https://img.shields.io/badge/CUDA-12.6-76b900)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

### Pricing Engine
- **Black-Scholes-Merton** analytical pricing with continuous dividend yield
- **Monte Carlo** — vectorized GBM simulation with chunked memory management
- **Heston stochastic volatility** — Euler-Maruyama with full truncation
- European, Asian (arithmetic), and Barrier options
- Analytical + finite-difference Greeks (Δ, Γ, ν, Θ, ρ)
- 95% confidence intervals and standard errors

### Variance Reduction
| Method | Variance reduction |
|--------|--------------------|
| Antithetic variates | ~44% |
| Control variate (European Sₜ as control) | ~83% |
| Control variate (European BS for Asian) | ~71% |

### GPU Acceleration
- PyTorch CUDA simulation on NVIDIA RTX 4060
- **6–21× speedup** vs NumPy CPU
- Interactive benchmark dashboard

### Analytics
- Implied volatility solver (Newton-Raphson, ~3 iterations to 10⁻⁸)
- Parameterized volatility surface with 3D visualization
- Convergence plots showing estimate + CI vs path count

### Dashboard
- Dark-themed professional UI with animated metrics
- Tabbed layout: Pricing | Vol Surface | GPU Benchmark
- 4 interactive Plotly charts per simulation
- Staggered fade-in animations, skeleton loading states
- Ctrl+Enter keyboard shortcut

## Architecture

```
backend/src/optionforge/
├── models/
│   ├── types.py          # Enums, dataclasses (OptionType, HestonParams, ...)
│   ├── stochastic.py     # GBM + Heston path simulation (chunked, vectorized)
│   └── payoffs.py        # European, Asian, Barrier payoff logic
├── pricing/
│   ├── black_scholes.py  # Analytical BS price, Greeks, implied volatility
│   ├── monte_carlo.py    # MC engine: standard, antithetic, control variate
│   ├── variance.py       # Antithetic pair generation
│   ├── greeks.py         # Central finite-difference MC Greeks
│   └── gpu_simulation.py # PyTorch CUDA-accelerated simulation + benchmark
├── api/
│   ├── schemas.py        # Pydantic request/response models
│   └── routes.py         # FastAPI endpoints
└── main.py               # App entry point

frontend/src/
├── components/
│   ├── ConfigPanel.tsx   # Left sidebar with all parameters + presets
│   ├── Charts.tsx        # 4 Plotly charts (paths, distributions, convergence)
│   ├── GreeksDisplay.tsx # Animated Greek cards with symbols
│   ├── VolSurface.tsx    # 3D volatility surface chart
│   └── Benchmark.tsx     # CPU vs GPU timing comparison
├── hooks/
│   └── useAnimatedValue.ts  # Smooth counter animation hook
├── api/client.ts         # Typed API client
└── types/index.ts        # Centralized TypeScript types
```

## Quick Start

### Prerequisites
- Python 3.12+, Node.js 22+, Docker
- (GPU benchmark) NVIDIA GPU with CUDA 12.6+

### Local Development

```bash
# Backend
cd backend
uv sync --extra dev
uv run uvicorn optionforge.main:app --reload --port 8000

# Frontend (proxies /api to :8000)
cd frontend
npm ci
npm run dev
```

Open `http://localhost:5173`

### Docker

```bash
docker compose up -d --build
```

Open `http://localhost:3000`

For GPU access in Docker:
```bash
sudo dnf install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker compose up -d --build
```

## API Reference

### `POST /api/price` — Monte Carlo pricing

```json
{
  "spot": 100, "strike": 100, "maturity": 1,
  "risk_free_rate": 0.05, "dividend_yield": 0.02,
  "volatility": 0.2, "n_paths": 50000, "n_steps": 252,
  "option_type": "call", "payoff_type": "european",
  "variance_reduction": "antithetic",
  "model_type": "gbm"
}
```

Returns: price, SE, 95% CI, BS benchmark, Greeks, errors.

### `POST /api/visualization` — Chart data

Returns: sampled paths (≤100), histograms, convergence data, Greeks.

### `POST /api/iv` — Implied volatility

```json
{
  "market_price": 9.23, "spot": 100, "strike": 100,
  "maturity": 1, "risk_free_rate": 0.05,
  "dividend_yield": 0.02, "option_type": "call"
}
```

### `POST /api/vol-surface` — Volatility surface grid

```json
{
  "spot": 100, "atm_vol": 0.2, "skew": -0.05,
  "smile": 0.15, "term": 0.02
}
```

### `POST /api/benchmark` — CPU vs GPU timing

Same params as `/api/price`, returns timing + speedup for both backends.

### Supported parameters

| Parameter | Values |
|-----------|--------|
| `option_type` | `call`, `put` |
| `payoff_type` | `european`, `asian`, `barrier` |
| `variance_reduction` | `none`, `antithetic`, `control_variate` |
| `model_type` | `gbm`, `heston` |
| `barrier_type` | `up_and_out`, `down_and_out`, `up_and_in`, `down_and_in` |

## Testing

```bash
cd backend
uv run pytest -v          # 62 tests
uv run ruff check src/ tests/   # Lint
```

Test coverage:
- Black-Scholes reference values, put-call parity (100 random combos)
- GBM path properties, chunking equivalence, deterministic seeds
- MC convergence to BS, CI coverage rate (~95%)
- Antithetic variance reduction, control variate correlation
- Barrier in-out parity, Heston pricing
- API validation errors, endpoint integration

## Project Structure

```
.
├── backend/
│   ├── pyproject.toml
│   ├── src/optionforge/
│   └── tests/
├── frontend/
│   ├── src/
│   └── server.cjs          # Express proxy for production
├── docker/
│   ├── Dockerfile.backend
│   └── Dockerfile.frontend
├── docker-compose.yml
├── Makefile
└── .github/workflows/ci.yml
```

## License

MIT
