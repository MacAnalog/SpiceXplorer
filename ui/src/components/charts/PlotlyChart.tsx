"use client";
import dynamic from "next/dynamic";
import type { PlotParams } from "react-plotly.js";

// Dynamic import avoids SSR issues with Plotly (uses window)
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const AXIS: Partial<Plotly.LayoutAxis> = {
  gridcolor: "rgba(0,0,0,0.06)",
  linecolor: "#a1a1aa",
  tickfont: { color: "#71717a", size: 10 },
  title: { font: { color: "#71717a", size: 10 } },
  zeroline: false,
  automargin: true,
};

const LAYOUT_BASE: Partial<Plotly.Layout> = {
  font: {
    family: "JetBrains Mono, ui-monospace, monospace",
    size: 10,
    color: "#71717a",
  },
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  margin: { l: 44, r: 14, t: 14, b: 28 },
  showlegend: false,
  xaxis: AXIS,
  yaxis: AXIS,
  hoverlabel: {
    bgcolor: "#ffffff",
    bordercolor: "#e4e4e7",
    font: { family: "JetBrains Mono, ui-monospace, monospace", size: 11, color: "#0a0a0a" },
  },
};

export const COLORS = {
  // Observatory palette
  primary: "#4f46e5", // indigo-600
  primarySoft: "#eef2ff",
  secondary: "#0891b2", // cyan-600
  tertiary: "#ea580c", // orange-600 (current-value markers)
  ok: "#059669", // emerald-600 (targets / zero line)
  danger: "#dc2626", // metric target line
  muted: "#71717a",
  faint: "#a1a1aa",
  // Legacy aliases (kept so older code keeps compiling)
  indigo: "#4f46e5",
  sky: "#0891b2",
  emerald: "#059669",
  amber: "#dc2626",
  red: "#dc2626",
  zinc: "#71717a",
};

export const STROKE = {
  primary: 1.6,
  raw: 0.8,
  axis: 1,
};

interface Props extends Omit<PlotParams, "layout"> {
  layout?: Partial<Plotly.Layout>;
  height?: number;
}

function mergeAxis(
  base: Partial<Plotly.LayoutAxis>,
  override?: Partial<Plotly.LayoutAxis>,
): Partial<Plotly.LayoutAxis> {
  if (!override) return base;
  return {
    ...base,
    ...override,
    title: {
      ...(base.title || {}),
      ...(override.title || {}),
      font: { ...(base.title?.font || {}), ...(override.title?.font || {}) },
    },
    tickfont: { ...(base.tickfont || {}), ...(override.tickfont || {}) },
  };
}

export function PlotlyChart({ layout, height = 240, ...rest }: Props) {
  const merged: Partial<Plotly.Layout> = {
    ...LAYOUT_BASE,
    ...layout,
    height,
    xaxis: mergeAxis(AXIS, layout?.xaxis),
    yaxis: mergeAxis(AXIS, layout?.yaxis),
  };
  return (
    <Plot
      {...rest}
      layout={merged}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%" }}
      useResizeHandler
    />
  );
}
