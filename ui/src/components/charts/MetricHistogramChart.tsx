"use client";
import { PlotlyChart, COLORS } from "./PlotlyChart";

interface RunHistogram {
  label: string;
  values: (number | null)[];
  color: string;
}

interface Props {
  runs: RunHistogram[];
  metric: string;
  target?: number | null;
  nbins?: number;
  height?: number;
}

export function MetricHistogramChart({ runs, metric, target, nbins = 40, height = 220 }: Props) {
  const traces: Plotly.Data[] = runs.map((run) => ({
    x: run.values.filter((v): v is number => v !== null),
    type: "histogram",
    name: run.label,
    nbinsx: nbins,
    opacity: 0.65,
    marker: { color: run.color },
  }));

  const shapes: Partial<Plotly.Shape>[] = [];
  if (target != null) {
    shapes.push({
      type: "line",
      x0: target, x1: target, xref: "x",
      y0: 0, y1: 1, yref: "paper",
      line: { color: COLORS.amber, width: 1.5, dash: "dash" },
    });
  }

  return (
    <PlotlyChart
      data={traces}
      height={height}
      layout={{
        title: { text: `${metric} — all designs`, font: { size: 12 } },
        xaxis: { title: { text: metric } },
        yaxis: { title: { text: "Count" } },
        barmode: "overlay",
        shapes,
      }}
    />
  );
}
