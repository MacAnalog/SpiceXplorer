"use client";
import { PlotlyChart, COLORS } from "./PlotlyChart";

interface RunSeries {
  label: string;
  scores: (number | null)[];
  best_scores: (number | null)[];
  color: string;
}

interface Props {
  runs: RunSeries[];
  height?: number;
}

export function ScoreConvergenceChart({ runs, height = 260 }: Props) {
  const traces: Plotly.Data[] = [];

  for (const run of runs) {
    const x = run.scores.map((_, i) => i);
    traces.push({
      x,
      y: run.scores,
      type: "scatter",
      mode: "lines",
      name: `${run.label} (raw)`,
      line: { color: run.color, width: 1, dash: "dot" },
      opacity: 0.45,
    });
    traces.push({
      x,
      y: run.best_scores,
      type: "scatter",
      mode: "lines",
      name: `${run.label} (best)`,
      line: { color: run.color, width: 2 },
    });
  }

  return (
    <PlotlyChart
      data={traces}
      height={height}
      layout={{
        title: { text: "Score vs. Iteration", font: { size: 12 } },
        xaxis: { title: { text: "Iteration" } },
        yaxis: { title: { text: "Score" } },
        shapes: [
          {
            type: "line",
            x0: 0, x1: 1, xref: "paper",
            y0: 0, y1: 0, yref: "y",
            line: { color: COLORS.emerald, width: 1.5, dash: "dash" },
          },
        ],
      }}
    />
  );
}
