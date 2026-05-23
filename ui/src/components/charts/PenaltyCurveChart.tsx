"use client";
import { PlotlyChart, COLORS } from "./PlotlyChart";
import type { ScoreCurve } from "@/types/api";

interface Props {
  curve: ScoreCurve;
  currentValue?: number | null;
  height?: number;
}

export function PenaltyCurveChart({ curve, currentValue, height = 280 }: Props) {
  const shapes: Partial<Plotly.Shape>[] = [
    {
      type: "line",
      x0: curve.target, x1: curve.target, xref: "x",
      y0: 0, y1: 1, yref: "paper",
      line: { color: COLORS.emerald, width: 1.5, dash: "dot" },
    },
  ];
  if (currentValue !== null && currentValue !== undefined) {
    shapes.push({
      type: "line",
      x0: currentValue, x1: currentValue, xref: "x",
      y0: 0, y1: 1, yref: "paper",
      line: { color: COLORS.indigo, width: 2, dash: "dash" },
    });
  }

  return (
    <PlotlyChart
      data={[
        {
          x: curve.values, y: curve.linear,
          type: "scatter", mode: "lines",
          name: "Linear P̂",
          line: { color: COLORS.red, width: 2 },
        },
        {
          x: curve.values, y: curve.sigmoid,
          type: "scatter", mode: "lines",
          name: "Sigmoid P̂",
          line: { color: COLORS.indigo, width: 2.5 },
        },
      ]}
      height={height}
      layout={{
        title: { text: "Normalized Penalty P̂(m)", font: { size: 12 } },
        xaxis: { title: { text: "Metric value m" } },
        yaxis: { title: { text: "Penalty P̂" }, rangemode: "tozero" },
        shapes,
        annotations: [
          {
            x: curve.target, y: 0, xref: "x", yref: "y",
            text: "target", showarrow: false,
            font: { size: 10, color: COLORS.emerald },
            yshift: -14,
          },
        ],
      }}
    />
  );
}
