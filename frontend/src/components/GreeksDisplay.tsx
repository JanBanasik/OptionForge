import { useAnimatedValue } from "../hooks/useAnimatedValue";

interface Props {
  greeks: {
    delta: number;
    gamma: number;
    vega: number;
    theta: number;
    rho: number;
  } | null;
  label: string;
  color: "cyan" | "purple";
  delay?: number;
}

const GREEK_CONFIG = {
  delta: { name: "Delta", symbol: "Δ", precision: 4, desc: "Price sensitivity to spot" },
  gamma: { name: "Gamma", symbol: "Γ", precision: 6, desc: "Delta sensitivity to spot" },
  vega:  { name: "Vega",  symbol: "ν", precision: 4, desc: "Price sensitivity to σ (+1%)" },
  theta: { name: "Theta", symbol: "Θ", precision: 4, desc: "Time decay (per day)" },
  rho:   { name: "Rho",   symbol: "ρ", precision: 4, desc: "Price sensitivity to r (+1%)" },
} as const;

const COLORS = {
  cyan:   { border: "rgba(6,182,212,0.25)", bg: "rgba(6,182,212,0.06)", text: "text-cyan-300" },
  purple: { border: "rgba(168,85,247,0.25)", bg: "rgba(168,85,247,0.06)", text: "text-purple-300" },
};

export default function GreeksDisplay({ greeks, label, color, delay = 0 }: Props) {
  if (!greeks) return null;
  const c = COLORS[color];

  const entries = (Object.keys(GREEK_CONFIG) as (keyof typeof GREEK_CONFIG)[]).map((key) => ({
    ...GREEK_CONFIG[key],
    value: greeks[key],
  }));

  return (
    <div className="mt-3 animate-fade-up" style={{ animationDelay: `${delay}ms` }}>
      <div className="mb-2 flex items-center gap-2">
        <div
          className="h-1.5 w-1.5 rounded-full"
          style={{ backgroundColor: color === "cyan" ? "#06b6d4" : "#a855f7" }}
        />
        <span className="text-[11px] font-medium uppercase tracking-widest text-zinc-500">
          {label}
        </span>
      </div>
      <div className="grid grid-cols-5 gap-2">
        {entries.map(({ name, symbol, value, precision, desc }, i) => {
          const animated = useAnimatedValue(value, 400);
          return (
            <div
              key={name}
              className="group relative rounded-lg border px-3 py-2.5 text-center transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[0_0_10px_rgba(6,182,212,0.05)]"
              style={{
                borderColor: c.border,
                backgroundColor: c.bg,
                animationDelay: `${delay + i * 40}ms`,
              }}
              title={desc}
            >
              <div className="mb-0.5 text-[10px] font-medium uppercase tracking-wider text-zinc-500">
                {symbol}
              </div>
              <div className={`font-mono text-sm font-semibold tabular-nums ${c.text}`}>
                {animated.toFixed(precision)}
              </div>
              <div className="mt-0.5 text-[9px] text-zinc-600">{name}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
