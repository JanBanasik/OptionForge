import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { PricingRequest } from "../types";

interface BenchmarkResponse {
  cpu_ms: number;
  cpu_price: number;
  cpu_se: number;
  gpu_ms: number | null;
  gpu_price: number | null;
  gpu_se: number | null;
  speedup: number | null;
  n_paths: number;
}

async function fetchBenchmark(req: PricingRequest): Promise<BenchmarkResponse> {
  const res = await fetch("/api/benchmark", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error("Benchmark failed");
  return res.json();
}

export default function Benchmark() {
  const [pathCounts] = useState([50_000, 200_000, 500_000]);
  const [active, setActive] = useState<number | null>(null);

  return (
    <div className="animate-fade-up space-y-4">
      <div className="flex flex-wrap items-center gap-3 rounded-xl border border-zinc-800/60 bg-zinc-900/40 p-4">
        <span className="text-xs font-semibold uppercase tracking-[0.15em] text-zinc-500">
          GPU Benchmark — RTX 4060
        </span>
        {pathCounts.map((n) => (
          <button
            key={n}
            onClick={() => setActive(n)}
            className={`rounded-lg px-4 py-2 text-xs font-semibold uppercase tracking-widest transition-all ${
              active === n
                ? "border border-cyan-600/50 bg-cyan-900/30 text-cyan-300"
                : "border border-zinc-800 bg-zinc-900/40 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200"
            }`}
          >
            {n >= 1000 ? `${(n / 1000).toFixed(0)}k` : n} paths
          </button>
        ))}
      </div>

      {active && (
        <BenchmarkResult nPaths={active} />
      )}
    </div>
  );
}

function BenchmarkResult({ nPaths }: { nPaths: number }) {
  const params: PricingRequest = {
    spot: 100, strike: 100, maturity: 1,
    risk_free_rate: 0.05, dividend_yield: 0.02, volatility: 0.2,
    n_paths: nPaths, n_steps: 252,
    option_type: "call", payoff_type: "european",
    variance_reduction: "none", seed: 42,
  };

  const query = useQuery<BenchmarkResponse>({
    queryKey: ["benchmark", nPaths],
    queryFn: () => fetchBenchmark(params),
    staleTime: Infinity,
  });

  const data = query.data;

  if (query.isFetching) {
    return (
      <div className="rounded-xl border border-zinc-800/60 bg-zinc-900/40 p-5">
        <div className="skeleton h-16 w-full rounded-lg" />
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="rounded-xl border border-zinc-800/60 bg-zinc-900/40 p-5 space-y-3">
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2 rounded-lg border border-zinc-800 bg-zinc-900/60 p-3">
          <div className="text-[10px] font-medium uppercase tracking-widest text-zinc-500">
            CPU (NumPy)
          </div>
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-2xl font-bold text-zinc-300">
              {data.cpu_ms.toFixed(0)}
            </span>
            <span className="text-xs text-zinc-500">ms</span>
          </div>
          <div className="font-mono text-xs text-zinc-500">
            price={data.cpu_price.toFixed(6)} SE={data.cpu_se.toFixed(6)}
          </div>
        </div>
        <div className="space-y-2 rounded-lg border border-cyan-800/40 bg-cyan-900/20 p-3">
          <div className="text-[10px] font-medium uppercase tracking-widest text-cyan-400">
            GPU (CUDA)
          </div>
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-2xl font-bold text-cyan-300">
              {data.gpu_ms?.toFixed(0) ?? "—"}
            </span>
            <span className="text-xs text-cyan-500">ms</span>
          </div>
          <div className="font-mono text-xs text-cyan-600">
            {data.gpu_price ? `price=${data.gpu_price.toFixed(6)}` : "CUDA not available"}
          </div>
        </div>
      </div>
      {data.speedup && (
        <div className="flex items-center gap-2 rounded-lg bg-green-900/20 border border-green-800/40 px-4 py-2">
          <span className="text-xs font-semibold uppercase tracking-widest text-green-400">
            {data.speedup}x speedup
          </span>
          <span className="text-xs text-green-600">
            with {data.n_paths.toLocaleString()} paths
          </span>
        </div>
      )}
    </div>
  );
}
