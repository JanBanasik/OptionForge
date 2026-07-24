import { type PricingRequest } from "../types";

interface Props {
  params: PricingRequest;
  onChange: (p: PricingRequest) => void;
  onRun: () => void;
  loading: boolean;
}

// Quick-start presets
const PRESETS: Record<string, Partial<PricingRequest>> = {
  "ATM Call": {
    spot: 100, strike: 100, maturity: 1, risk_free_rate: 0.05,
    dividend_yield: 0.02, volatility: 0.2, option_type: "call",
    payoff_type: "european", variance_reduction: "none",
  },
  "ITM Put": {
    spot: 90, strike: 100, maturity: 0.5, risk_free_rate: 0.04,
    dividend_yield: 0.01, volatility: 0.3, option_type: "put",
    payoff_type: "european", variance_reduction: "antithetic",
  },
  "Asian Call": {
    spot: 100, strike: 105, maturity: 1, risk_free_rate: 0.05,
    dividend_yield: 0.02, volatility: 0.25, option_type: "call",
    payoff_type: "asian", variance_reduction: "antithetic",
  },
};

export default function ConfigPanel({ params, onChange, onRun, loading }: Props) {
  const set = (patch: Partial<PricingRequest>) => onChange({ ...params, ...patch });

  const field = (label: string, key: keyof PricingRequest, step = "any", min?: number, max?: number) => (
    <label className="block">
      <span className="mb-1 block text-[11px] font-medium uppercase tracking-widest text-zinc-500">
        {label}
      </span>
      <input
        type="number"
        step={step}
        min={min}
        max={max}
        value={params[key] ?? ""}
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          if (!isNaN(v)) set({ [key]: v });
        }}
        className="w-full rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2 text-sm text-zinc-200 placeholder-zinc-600 outline-none transition-all duration-200 focus:border-cyan-600 focus:bg-zinc-900 focus:ring-1 focus:ring-cyan-600/30"
      />
    </label>
  );

  const select = (label: string, key: keyof PricingRequest, opts: string[]) => (
    <label className="block">
      <span className="mb-1 block text-[11px] font-medium uppercase tracking-widest text-zinc-500">
        {label}
      </span>
      <select
        value={params[key] as string}
        onChange={(e) => set({ [key]: e.target.value })}
        className="w-full cursor-pointer rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2 text-sm text-zinc-200 outline-none transition-all duration-200 focus:border-cyan-600 focus:ring-1 focus:ring-cyan-600/30"
      >
        {opts.map((o) => (
          <option key={o} value={o} className="bg-zinc-900">
            {o}
          </option>
        ))}
      </select>
    </label>
  );

  return (
    <aside className="flex w-72 shrink-0 flex-col border-r border-zinc-800/60 bg-zinc-950">
      {/* Header */}
      <div className="border-b border-zinc-800/60 px-5 py-4">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-cyan-500 to-cyan-700 text-sm font-bold text-white">
            Δ
          </div>
          <div>
            <h1 className="text-sm font-semibold tracking-tight text-zinc-100">
              OptionForge
            </h1>
            <p className="text-[10px] text-zinc-600">Monte Carlo Lab</p>
          </div>
        </div>
      </div>

      {/* Presets */}
      <div className="border-b border-zinc-800/60 px-5 py-3">
        <div className="flex flex-wrap gap-1.5">
          {Object.entries(PRESETS).map(([name, preset]) => (
            <button
              key={name}
              onClick={() => onChange({ ...params, ...preset })}
              className="rounded-md border border-zinc-800 bg-zinc-900/40 px-2.5 py-1 text-[11px] font-medium text-zinc-400 transition-all hover:border-zinc-700 hover:text-zinc-200"
            >
              {name}
            </button>
          ))}
        </div>
      </div>

      {/* Scrollable options */}
      <div className="flex-1 space-y-4 overflow-y-auto px-5 py-4">
        <div className="space-y-3">
          <h3 className="text-[10px] font-semibold uppercase tracking-[0.2em] text-zinc-600">
            Market
          </h3>
          {field("Spot Price S₀", "spot", "0.1", 0.01)}
          {field("Strike Price K", "strike", "0.1", 0.01)}
          {field("Maturity T (years)", "maturity", "0.01", 0.01)}
          {field("Risk-Free Rate r", "risk_free_rate", "0.001", 0)}
          {field("Dividend Yield q", "dividend_yield", "0.001", 0)}
          {field("Volatility σ", "volatility", "0.01", 0.01, 5)}
        </div>

        <div className="space-y-3">
          <h3 className="text-[10px] font-semibold uppercase tracking-[0.2em] text-zinc-600">
            Contract
          </h3>
          {select("Option Type", "option_type", ["call", "put"])}
          {select("Payoff Style", "payoff_type", ["european", "asian", "barrier"])}
        </div>

        {params.payoff_type === "barrier" && (
          <div className="space-y-3">
            <h3 className="text-[10px] font-semibold uppercase tracking-[0.2em] text-zinc-600">
              Barrier
            </h3>
            {select("Barrier Type", "barrier_type" as keyof PricingRequest, [
              "up_and_out",
              "down_and_out",
              "up_and_in",
              "down_and_in",
            ])}
            {field("Barrier Level", "barrier_level" as keyof PricingRequest, "0.1", 0.01)}
          </div>
        )}

        <div className="space-y-3">
          <h3 className="text-[10px] font-semibold uppercase tracking-[0.2em] text-zinc-600">
            Simulation
          </h3>
          {field("Paths", "n_paths", "1000", 100, 500000)}
          {field("Time Steps", "n_steps", "1", 1, 2000)}
          {select("Variance Reduction", "variance_reduction", ["none", "antithetic", "control_variate"])}
          <label className="block">
            <span className="mb-1 block text-[11px] font-medium uppercase tracking-widest text-zinc-500">
              Seed
            </span>
            <input
              type="number"
              value={params.seed ?? ""}
              onChange={(e) => {
                const v = e.target.value;
                set({ seed: v === "" ? undefined : parseInt(v) || 0 });
              }}
              placeholder="Random"
              className="w-full rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2 text-sm text-zinc-200 placeholder-zinc-600 outline-none transition-all duration-200 focus:border-cyan-600 focus:ring-1 focus:ring-cyan-600/30"
            />
          </label>
        </div>
      </div>

      {/* Run button */}
      <div className="border-t border-zinc-800/60 p-4">
        <button
          onClick={onRun}
          disabled={loading}
          className={`group relative w-full overflow-hidden rounded-xl py-3 text-sm font-semibold uppercase tracking-widest transition-all duration-300 ${
            loading
              ? "cursor-not-allowed border border-zinc-800 bg-zinc-900 text-zinc-600"
              : "cursor-pointer border border-cyan-700/50 bg-gradient-to-r from-cyan-600 to-cyan-700 text-white shadow-lg shadow-cyan-900/20 hover:from-cyan-500 hover:to-cyan-600 hover:shadow-cyan-900/40 pulse-ring"
          }`}
        >
          {loading ? (
            <span className="inline-flex items-center gap-2">
              <svg className="spin-slow h-4 w-4" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeDasharray="32" strokeLinecap="round" />
              </svg>
              Running...
            </span>
          ) : (
            <>
              <span className="relative z-10">Run Simulation</span>
              <span className="ml-2 text-xs text-cyan-300/70">⌃↵</span>
            </>
          )}
        </button>
      </div>
    </aside>
  );
}
