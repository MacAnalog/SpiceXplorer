"use client";
import dynamic from "next/dynamic";
import type { PlotParams } from "react-plotly.js";

// Dynamic import avoids SSR issues with Plotly (uses window)
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const LAYOUT_BASE: Partial<Plotly.Layout> = {
  paper_bgcolor: "white",
  plot_bgcolor: "#fafafa",
  font: { family: "ui-monospace, monospace", size: 11, color: "#3f3f46" },
  margin: { t: 24, r: 16, b: 40, l: 48 },
  showlegend: true,
  legend: { orientation: "h", y: -0.18, x: 0 },
  xaxis: { gridcolor: "#e4e4e7", linecolor: "#e4e4e7", zerolinecolor: "#d4d4d8" },
  yaxis: { gridcolor: "#e4e4e7", linecolor: "#e4e4e7", zerolinecolor: "#d4d4d8" },
};

export const COLORS = {
  indigo: "#6366f1",
  emerald: "#10b981",
  red: "#ef4444",
  amber: "#f59e0b",
  zinc: "#71717a",
  sky: "#0ea5e9",
};

interface Props extends Omit<PlotParams, "layout"> {
  layout?: Partial<Plotly.Layout>;
  height?: number;
}

export function PlotlyChart({ layout, height = 260, ...rest }: Props) {
  return (
    <Plot
      {...rest}
      layout={{ ...LAYOUT_BASE, height, ...layout }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%" }}
      useResizeHandler
    />
  );
}
