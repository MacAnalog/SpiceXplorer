"use client";
import { PlotlyChart, COLORS } from "./PlotlyChart";

interface RunMetric {
  label: string;
  values: (number | null)[];
  color: string;
}

interface Props {
  metric: string;
  runs: RunMetric[];
  target?: number | null;
  goal?: string;
  height?: number;
}

function bestSoFar(values: (number | null)[], goal: string): (number | null)[] {
  let best: number | null = null;
  return values.map((v) => {
    if (v === null) return best;
    if (best === null) { best = v; return best; }
    if (goal === "minimize") best = Math.min(best, v);
    else best = Math.max(best, v);
    return best;
  });
}

export function MetricConvergenceChart({ metric, runs, target, goal = "exceed", height = 220 }: Props) {
  const traces: Plotly.Data[] = [];

  for (const run of runs) {
    const bsf = bestSoFar(run.values, goal);
    const x = run.values.map((_, i) => i);
    traces.push({
      x, y: bsf,
      type: "scatter", mode: "lines",
      name: run.label,
      line: { color: run.color, width: 2 },
    });
  }

  const shapes: Partial<Plotly.Shape>[] = [];
  if (target !== null && target !== undefined) {
    shapes.push({
      type: "line",
      x0: 0, x1: 1, xref: "paper",
      y0: target, y1: target, yref: "y",
      line: { color: COLORS.amber, width: 1.5, dash: "dash" },
    });
  }

  return (
    <PlotlyChart
      data={traces}
      height={height}
      layout={{
        title: { text: `${metric} — best so far`, font: { size: 12 } },
        xaxis: { title: { text: "Iteration" } },
        yaxis: { title: { text: metric } },
        shapes,
      }}
    />
  );
}
