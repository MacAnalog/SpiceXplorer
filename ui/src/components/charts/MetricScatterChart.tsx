"use client";
import { PlotlyChart, COLORS } from "./PlotlyChart";
import type { ScatterPoint } from "@/types/api";

interface RunScatter {
  label: string;
  points: ScatterPoint[];
}

interface Props {
  runs: RunScatter[];
  metricX: string;
  metricY: string;
  targetX?: number | null;
  targetY?: number | null;
  height?: number;
}

export function MetricScatterChart({ runs, metricX, metricY, targetX, targetY, height = 300 }: Props) {
  const traces: Plotly.Data[] = [];
  const runColors = [COLORS.indigo, COLORS.sky];

  runs.forEach((run, ri) => {
    const feasible = run.points.filter((p) => p.feasible);
    const infeasible = run.points.filter((p) => !p.feasible);
    const col = runColors[ri % runColors.length];

    if (infeasible.length) {
      traces.push({
        x: infeasible.map((p) => p.x),
        y: infeasible.map((p) => p.y),
        type: "scatter", mode: "markers",
        name: `${run.label} (infeasible)`,
        marker: { color: "#d4d4d8", size: 4, opacity: 0.5 },
        showlegend: ri === 0,
      });
    }
    if (feasible.length) {
      traces.push({
        x: feasible.map((p) => p.x),
        y: feasible.map((p) => p.y),
        type: "scatter", mode: "markers",
        name: `${run.label} (feasible)`,
        marker: { color: col, size: 6, opacity: 0.85 },
      });
    }
  });

  const shapes: Partial<Plotly.Shape>[] = [];
  if (targetX != null) {
    shapes.push({ type: "line", x0: targetX, x1: targetX, xref: "x", y0: 0, y1: 1, yref: "paper", line: { color: COLORS.amber, dash: "dash", width: 1 } });
  }
  if (targetY != null) {
    shapes.push({ type: "line", x0: 0, x1: 1, xref: "paper", y0: targetY, y1: targetY, yref: "y", line: { color: COLORS.amber, dash: "dash", width: 1 } });
  }

  return (
    <PlotlyChart
      data={traces}
      height={height}
      layout={{
        title: { text: `${metricX} vs ${metricY}`, font: { size: 12 } },
        xaxis: { title: { text: metricX } },
        yaxis: { title: { text: metricY } },
        shapes,
      }}
    />
  );
}
