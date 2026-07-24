import Plotly from "plotly.js-cartesian-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";
import type { VisualizationResponse } from "../types";

const Plot = createPlotlyComponent(Plotly);

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type PlotlyData = any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type PlotlyLayout = any;

const baseLayout: PlotlyLayout = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#71717a", size: 11, family: "Inter, system-ui, sans-serif" },
  xaxis: {
    gridcolor: "rgba(39,39,42,0.5)",
    zerolinecolor: "rgba(63,63,70,0.5)",
    zerolinewidth: 1,
  },
  yaxis: {
    gridcolor: "rgba(39,39,42,0.5)",
    zerolinecolor: "rgba(63,63,70,0.5)",
    zerolinewidth: 1,
  },
  margin: { l: 55, r: 25, t: 30, b: 45 },
  autosize: true,
  hovermode: "closest" as const,
};

interface Props {
  data: VisualizationResponse;
}

function ChartContainer({
  title,
  children,
  delay = 0,
}: {
  title: string;
  children: React.ReactNode;
  delay?: number;
}) {
  return (
    <div
      className="animate-fade-up rounded-xl border border-zinc-800/60 bg-zinc-900/40 p-5 transition-colors duration-300 hover:border-zinc-700/60"
      style={{ animationDelay: `${delay}ms` }}
    >
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-[0.15em] text-zinc-500">
        {title}
      </h3>
      {children}
    </div>
  );
}

export default function Charts({ data }: Props) {
  const timeGrid = data.time_grid;
  const sampled = data.sampled_paths;

  // Path chart — faint background paths, one highlighted
  const pathTraces: PlotlyData[] = sampled.slice(0, 100).map((path, i) => ({
    x: timeGrid,
    y: path,
    type: "scatter" as const,
    mode: "lines" as const,
    line: {
      width: i === 0 ? 1.2 : 0.3,
      color: i === 0 ? "#06b6d4" : "rgba(63,63,70,0.4)",
    },
    showlegend: false,
    hoverinfo: i === 0 ? ("text" as const) : ("skip" as const),
    text: i === 0 ? "Path 1" : undefined,
  }));

  // Terminal price histogram
  const edges = data.terminal_bin_edges;
  const terminalHist: PlotlyData[] = [
    {
      x: edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2),
      y: data.terminal_bin_counts,
      type: "bar" as const,
      marker: {
        color: "rgba(6,182,212,0.7)",
        line: { width: 1, color: "rgba(6,182,212,0.9)" },
      },
      showlegend: false,
    },
  ];

  // Payoff histogram
  const payoffEdges = data.payoff_bin_edges;
  const payoffHist: PlotlyData[] = [
    {
      x: payoffEdges.slice(0, -1).map((e, i) => (e + payoffEdges[i + 1]) / 2),
      y: data.payoff_bin_counts,
      type: "bar" as const,
      marker: {
        color: "rgba(168,85,247,0.6)",
        line: { width: 1, color: "rgba(168,85,247,0.8)" },
      },
      showlegend: false,
    },
  ];

  // Convergence
  const conv = data.convergence;
  const convTrace: PlotlyData[] = [
    {
      x: conv.map((c) => c.n_paths),
      y: conv.map((c) => c.price),
      type: "scatter" as const,
      mode: "lines" as const,
      line: { color: "#22c55e", width: 2.5, shape: "spline" as const },
      name: "MC Estimate",
    },
    {
      x: [...conv.map((c) => c.n_paths), ...conv.map((c) => c.n_paths).reverse()],
      y: [
        ...conv.map((c) => c.ci_upper),
        ...conv.map((c) => c.ci_lower).reverse(),
      ],
      type: "scatter" as const,
      fill: "toself",
      fillcolor: "rgba(34,197,94,0.1)",
      line: { width: 0 },
      name: "95% CI",
    },
  ];

  return (
    <div className="space-y-4">
      <ChartContainer title="Simulated Price Paths" delay={240}>
        <Plot
          data={pathTraces}
          layout={{
            ...baseLayout,
            height: 350,
            xaxis: { ...baseLayout.xaxis, title: "Time (years)" },
            yaxis: { ...baseLayout.yaxis, title: "Asset Price" },
          }}
          useResizeHandler
          style={{ width: "100%" }}
          config={{ displayModeBar: false }}
        />
      </ChartContainer>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <ChartContainer title="Terminal Price Distribution" delay={300}>
          <Plot
            data={terminalHist}
            layout={{
              ...baseLayout,
              height: 280,
              xaxis: { ...baseLayout.xaxis, title: "Sₜ" },
              yaxis: { ...baseLayout.yaxis, title: "Frequency" },
              bargap: 0.05,
            }}
            useResizeHandler
            style={{ width: "100%" }}
            config={{ displayModeBar: false }}
          />
        </ChartContainer>
        <ChartContainer title="Discounted Payoff Distribution" delay={360}>
          <Plot
            data={payoffHist}
            layout={{
              ...baseLayout,
              height: 280,
              xaxis: { ...baseLayout.xaxis, title: "Payoff" },
              yaxis: { ...baseLayout.yaxis, title: "Frequency" },
              bargap: 0.05,
            }}
            useResizeHandler
            style={{ width: "100%" }}
            config={{ displayModeBar: false }}
          />
        </ChartContainer>
      </div>

      <ChartContainer title="Convergence" delay={420}>
        <Plot
          data={convTrace}
          layout={{
            ...baseLayout,
            height: 340,
            xaxis: {
              ...baseLayout.xaxis,
              title: "Number of Paths",
              type: "log" as const,
            },
            yaxis: { ...baseLayout.yaxis, title: "Option Price" },
            showlegend: true,
            legend: {
              x: 1,
              y: 1,
              bgcolor: "rgba(24,24,27,0.9)",
              bordercolor: "rgba(39,39,42,0.6)",
              borderwidth: 1,
              font: { size: 10, color: "#a1a1aa" },
            },
          }}
          useResizeHandler
          style={{ width: "100%" }}
          config={{ displayModeBar: false }}
        />
      </ChartContainer>
    </div>
  );
}
