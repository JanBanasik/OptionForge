import { useCallback, useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import ConfigPanel from "./components/ConfigPanel";
import Charts from "./components/Charts";
import GreeksDisplay from "./components/GreeksDisplay";
import VolSurface from "./components/VolSurface";
import Benchmark from "./components/Benchmark";
import { fetchPrice, fetchVisualization } from "./api/client";
import { useAnimatedValue } from "./hooks/useAnimatedValue";
import type { PricingRequest, PricingResponse, VisualizationResponse } from "./types";

const defaults: PricingRequest = {
  spot: 100,
  strike: 100,
  maturity: 1,
  risk_free_rate: 0.05,
  dividend_yield: 0.02,
  volatility: 0.2,
  n_paths: 50000,
  n_steps: 252,
  option_type: "call",
  payoff_type: "european",
  variance_reduction: "none",
};

/* ── Animated metric card ── */
function MetricCard({
  label,
  value,
  suffix = "",
  precision = 4,
  highlight = false,
  delay = 0,
}: {
  label: string;
  value: number;
  suffix?: string;
  precision?: number;
  highlight?: boolean;
  delay?: number;
}) {
  const animated = useAnimatedValue(value, 500);
  const display = animated.toFixed(precision);

  return (
    <div
      className="animate-fade-up group relative overflow-hidden rounded-xl border px-4 py-3.5 transition-all duration-300 hover:border-zinc-600"
      style={{
        animationDelay: `${delay}ms`,
        background: highlight
          ? "linear-gradient(135deg, rgba(6,182,212,0.08) 0%, rgba(6,182,212,0.02) 100%)"
          : "linear-gradient(135deg, rgba(24,24,27,0.6) 0%, rgba(24,24,27,0.3) 100%)",
        borderColor: highlight ? "rgba(6,182,212,0.3)" : "rgba(39,39,42,0.6)",
      }}
    >
      <div className="mb-0.5 text-[11px] font-medium uppercase tracking-widest text-zinc-500">
        {label}
      </div>
      <div
        className={`font-mono text-lg font-semibold tracking-tight tabular-nums ${
          highlight ? "text-cyan-300" : "text-zinc-100"
        }`}
      >
        {display}
        {suffix && (
          <span className="ml-1 text-xs font-normal text-zinc-500">{suffix}</span>
        )}
      </div>
    </div>
  );
}

/* ── Loading skeleton card ── */
function SkeletonCard() {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 px-4 py-3.5">
      <div className="skeleton mb-2 h-3 w-16" />
      <div className="skeleton h-5 w-24" />
    </div>
  );
}

/* ── Empty state ── */
function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl border border-zinc-800 bg-zinc-900/50">
        <svg className="h-7 w-7 text-zinc-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6A2.25 2.25 0 016 3.75h2.25A2.25 2.25 0 0110.5 6v2.25a2.25 2.25 0 01-2.25 2.25H6a2.25 2.25 0 01-2.25-2.25V6zM3.75 15.75A2.25 2.25 0 016 13.5h2.25a2.25 2.25 0 012.25 2.25V18a2.25 2.25 0 01-2.25 2.25H6A2.25 2.25 0 013.75 18v-2.25zM13.5 6a2.25 2.25 0 012.25-2.25H18A2.25 2.25 0 0120.25 6v2.25A2.25 2.25 0 0118 10.5h-2.25a2.25 2.25 0 01-2.25-2.25V6zM13.5 15.75a2.25 2.25 0 012.25-2.25H18a2.25 2.25 0 012.25 2.25V18A2.25 2.25 0 0118 20.25h-2.25A2.25 2.25 0 0113.5 18v-2.25z" />
        </svg>
      </div>
      <p className="text-sm font-medium text-zinc-400">Configure parameters</p>
      <p className="mt-1 text-xs text-zinc-600">
        Adjust the settings on the left and click <span className="font-semibold text-cyan-500">Run Simulation</span>
      </p>
    </div>
  );
}

/* ── Loading state ── */
function LoadingState() {
  return (
    <div className="space-y-5 animate-fade-up">
      {/* Skeleton metric cards */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
        {Array.from({ length: 6 }).map((_, i) => (
          <SkeletonCard key={i} />
        ))}
      </div>
      {/* Skeleton chart containers */}
      <div className="space-y-4">
        {[350, 280, 320].map((h, i) => (
          <div key={i} className="rounded-xl border border-zinc-800 bg-zinc-950/60 p-5">
            <div className="skeleton mb-4 h-3 w-32" />
            <div className={`skeleton w-full rounded-lg`} style={{ height: h }} />
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [params, setParams] = useState<PricingRequest>(defaults);
  const [run, setRun] = useState(0);

  const priceQuery = useQuery<PricingResponse>({
    queryKey: ["price", params, run],
    queryFn: () => fetchPrice(params),
    enabled: run > 0,
    retry: false,
    staleTime: Infinity,
  });

  const vizQuery = useQuery<VisualizationResponse>({
    queryKey: ["viz", params, run],
    queryFn: () => fetchVisualization(params),
    enabled: run > 0,
    retry: false,
    staleTime: Infinity,
  });

  const handleRun = useCallback(() => {
    setRun((r) => r + 1);
  }, []);

  // Ctrl+Enter shortcut
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        handleRun();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [handleRun]);

  const error = priceQuery.error || vizQuery.error;
  const priceData = priceQuery.data;
  const vizData = vizQuery.data;
  const isLoading = priceQuery.isFetching || vizQuery.isFetching;
  const hasRun = run > 0;

  return (
    <div className="flex h-screen overflow-hidden bg-zinc-950">
      {/* ── Left panel ── */}
      <ConfigPanel
        params={params}
        onChange={setParams}
        onRun={handleRun}
        loading={isLoading}
      />

      {/* ── Main content ── */}
      <main className="flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto max-w-5xl space-y-5">
          {/* Error banner */}
          {error && (
            <div className="animate-fade-up rounded-xl border border-red-800/50 bg-red-950/30 px-4 py-3">
              <p className="text-sm text-red-400">{(error as Error).message}</p>
            </div>
          )}

          {/* Empty state */}
          {!hasRun && !isLoading && <EmptyState />}

          {/* Loading */}
          {isLoading && <LoadingState />}

          {/* Results */}
          {priceData && !isLoading && (
            <div className="space-y-5">
              {/* Metrics row */}
              <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
                <MetricCard
                  label="MC Price"
                  value={priceData.price}
                  highlight
                  delay={0}
                />
                <MetricCard
                  label="BS Price"
                  value={priceData.black_scholes_price ?? 0}
                  delay={60}
                />
                <MetricCard
                  label="Abs Error"
                  value={priceData.absolute_error ?? 0}
                  precision={6}
                  delay={120}
                />
                <MetricCard
                  label="Rel Error"
                  value={(priceData.relative_error ?? 0) * 100}
                  suffix="%"
                  precision={4}
                  delay={180}
                />
                <MetricCard
                  label="Std Error"
                  value={priceData.standard_error}
                  precision={6}
                  delay={240}
                />
                <MetricCard
                  label="95% CI"
                  value={priceData.confidence_interval_upper - priceData.confidence_interval_lower}
                  precision={4}
                  delay={300}
                />
              </div>

              {/* CI detail */}
              <div
                className="animate-fade-up rounded-xl border border-zinc-800 bg-zinc-900/40 px-4 py-3"
                style={{ animationDelay: "120ms" }}
              >
                <div className="flex items-center gap-3 text-sm">
                  <span className="text-zinc-500">95% confidence interval:</span>
                  <span className="font-mono tabular-nums text-zinc-300">
                    [{priceData.confidence_interval_lower.toFixed(6)},{" "}
                    {priceData.confidence_interval_upper.toFixed(6)}]
                  </span>
                  {priceData.black_scholes_price !== null && (
                    <>
                      <span className="text-zinc-600">|</span>
                      <span className="text-zinc-500">
                        BS:{" "}
                        <span className="font-mono text-purple-400">
                          {priceData.black_scholes_price.toFixed(6)}
                        </span>
                      </span>
                    </>
                  )}
                </div>
              </div>

              {/* Greeks */}
              <div
                className="animate-fade-up rounded-xl border border-zinc-800 bg-zinc-900/40 p-5"
                style={{ animationDelay: "180ms" }}
              >
                <h3 className="mb-3 text-xs font-semibold uppercase tracking-[0.2em] text-zinc-500">
                  Greeks
                </h3>
                <GreeksDisplay
                  greeks={priceData.greeks}
                  label="Monte Carlo"
                  color="cyan"
                  delay={0}
                />
                {priceData.bs_greeks && (
                  <GreeksDisplay
                    greeks={priceData.bs_greeks}
                    label="Black-Scholes"
                    color="purple"
                    delay={200}
                  />
                )}
              </div>

              {/* Charts */}
              {vizData && <Charts data={vizData} />}
            </div>
          )}
        </div>

        {/* Volatility Surface — always available */}
        <VolSurface />
        <Benchmark />
      </main>
    </div>
  );
}
