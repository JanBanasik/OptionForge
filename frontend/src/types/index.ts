export interface GreeksData {
  delta: number;
  gamma: number;
  vega: number;
  theta: number;
  rho: number;
}

export interface PricingRequest {
  spot: number;
  strike: number;
  maturity: number;
  risk_free_rate: number;
  dividend_yield: number;
  volatility: number;
  n_paths: number;
  n_steps: number;
  option_type: "call" | "put";
  payoff_type: "european" | "asian";
  variance_reduction: "none" | "antithetic" | "control_variate";
  seed?: number;
}

export interface PricingResponse {
  price: number;
  payoff_mean: number;
  payoff_std: number;
  standard_error: number;
  confidence_interval_lower: number;
  confidence_interval_upper: number;
  black_scholes_price: number | null;
  absolute_error: number | null;
  relative_error: number | null;
  greeks: GreeksData | null;
  bs_greeks: GreeksData | null;
}

export interface ConvergencePoint {
  n_paths: number;
  price: number;
  ci_lower: number;
  ci_upper: number;
}

export interface VisualizationResponse {
  sampled_paths: number[][];
  time_grid: number[];
  terminal_bin_edges: number[];
  terminal_bin_counts: number[];
  payoff_bin_edges: number[];
  payoff_bin_counts: number[];
  convergence: ConvergencePoint[];
  greeks: GreeksData | null;
  bs_greeks: GreeksData | null;
}
