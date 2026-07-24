import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Plotly from "plotly.js-cartesian-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

const Plot = createPlotlyComponent(Plotly);

interface VolSurfaceResponse {
  strikes: number[];
  maturities: number[];
  iv_grid: number[][];
  spot: number;
}

interface VolSurfaceRequest {
  spot: number;
  risk_free_rate: number;
  dividend_yield: number;
  atm_vol: number;
  skew: number;
  smile: number;
  term: number;
  n_strikes: number;
  n_maturities: number;
}

const defaults: VolSurfaceRequest = {
  spot: 100,
  risk_free_rate: 0.05,
  dividend_yield: 0.02,
  atm_vol: 0.2,
  skew: -0.05,
  smile: 0.15,
  term: 0.02,
  n_strikes: 20,
  n_maturities: 10,
};

async function fetchVolSurface(req: VolSurfaceRequest): Promise<VolSurfaceResponse> {
  const res = await fetch("/api/vol-surface", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error("Failed to fetch vol surface");
  return res.json();
}

function field(label: string, value: number, onChange: (v: number) => void, step = "any", min?: number, max?: number) {
  return (
    <label className="block">
      <span className="mb-1 block text-[11px] font-medium uppercase tracking-widest text-zinc-500">{label}</span>
      <input
        type="number" step={step} min={min} max={max} value={value}
        onChange={(e) => { const v = parseFloat(e.target.value); if (!isNaN(v)) onChange(v); }}
        className="w-full rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2 text-sm text-zinc-200 outline-none transition-all duration-200 focus:border-cyan-600 focus:ring-1 focus:ring-cyan-600/30"
      />
    </label>
  );
}

export default function VolSurface() {
  const [params, setParams] = useState(defaults);
  const [run, setRun] = useState(0);

  const query = useQuery<VolSurfaceResponse>({
    queryKey: ["vol-surface", params, run],
    queryFn: () => fetchVolSurface(params),
    enabled: run > 0,
    staleTime: Infinity,
  });

  const data = query.data;

  return (
    <div className="animate-fade-up space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap items-end gap-3 rounded-xl border border-zinc-800/60 bg-zinc-900/40 p-4">
        {field("ATM Vol σ₀", params.atm_vol, (v) => setParams({ ...params, atm_vol: v }), "0.01", 0.01, 5)}
        {field("Skew", params.skew, (v) => setParams({ ...params, skew: v }), "0.01")}
        {field("Smile", params.smile, (v) => setParams({ ...params, smile: v }), "0.01", 0)}
        {field("Term Slope", params.term, (v) => setParams({ ...params, term: v }), "0.01")}
        <button
          onClick={() => setRun((r) => r + 1)}
          disabled={query.isFetching}
          className={`rounded-xl px-5 py-2 text-sm font-semibold uppercase tracking-widest transition-all ${
            query.isFetching
              ? "cursor-not-allowed border border-zinc-800 bg-zinc-900 text-zinc-600"
              : "cursor-pointer border border-cyan-700/50 bg-gradient-to-r from-cyan-600 to-cyan-700 text-white hover:from-cyan-500 hover:to-cyan-600"
          }`}
        >
          {query.isFetching ? "Loading..." : "Generate"}
        </button>
      </div>

      {/* Surface chart */}
      {(query.isFetching || data) && (
        <div className="rounded-xl border border-zinc-800/60 bg-zinc-900/40 p-5">
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-[0.15em] text-zinc-500">
            Volatility Surface σ(K, T)
          </h3>
          {query.isFetching ? (
            <div className="skeleton h-[420px] w-full rounded-lg" />
          ) : data ? (
            <Plot
              data={[
                {
                  z: data.iv_grid,
                  x: data.strikes,
                  y: data.maturities,
                  type: "surface" as const,
                  colorscale: [
                    [0, "#06b6d4"],
                    [0.5, "#0e7490"],
                    [1, "#a855f7"],
                  ] as [number, string][],
                  contours: {
                    z: { show: true, usecolormap: true, highlightcolor: "rgba(255,255,255,0.3)", project: { z: true } },
                  },
                  colorbar: {
                    title: { text: "σ", font: { color: "#a1a1aa" } },
                    tickfont: { color: "#a1a1aa", size: 10 },
                  },
                },
              ]}
              layout={{
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                font: { color: "#a1a1aa", size: 11 },
                height: 480,
                autosize: true,
                margin: { l: 20, r: 20, t: 10, b: 20 },
                scene: {
                  xaxis: { title: "Strike", gridcolor: "rgba(39,39,42,0.4)", color: "#a1a1aa" },
                  yaxis: { title: "Maturity (T)", gridcolor: "rgba(39,39,42,0.4)", color: "#a1a1aa" },
                  zaxis: { title: "Implied Vol", gridcolor: "rgba(39,39,42,0.4)", color: "#a1a1aa" },
                  bgcolor: "transparent",
                },
              }}
              useResizeHandler
              style={{ width: "100%" }}
              config={{ displayModeBar: false }}
            />
          ) : null}
        </div>
      )}
    </div>
  );
}
