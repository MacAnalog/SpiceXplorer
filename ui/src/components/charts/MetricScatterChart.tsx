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
  goalX?: string;
  goalY?: string;
  height?: number;
}

function feasibleRect(
  pts: ScatterPoint[],
  targetX: number | null | undefined,
  targetY: number | null | undefined,
  goalX: string,
  goalY: string,
): Partial<Plotly.Shape> | null {
  if (targetX == null || targetY == null || pts.length === 0) return null;
  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const ymin = Math.min(...ys);
  const ymax = Math.max(...ys);
  const x0 = goalX === "minimize" ? xmin : targetX;
  const x1 = goalX === "minimize" ? targetX : xmax;
  const y0 = goalY === "minimize" ? ymin : targetY;
  const y1 = goalY === "minimize" ? targetY : ymax;
  return {
    type: "rect",
    x0,
    x1,
    y0,
    y1,
    xref: "x",
    yref: "y",
    line: { width: 0 },
    fillcolor: COLORS.ok,
    opacity: 0.05,
    layer: "below",
  };
}

export function MetricScatterChart({
  runs,
  metricX,
  metricY,
  targetX,
  targetY,
  goalX = "exceed",
  goalY = "minimize",
  height = 240,
}: Props) {
  const traces: Plotly.Data[] = [];
  const feasibleColors = [COLORS.primary, COLORS.secondary];

  runs.forEach((run, ri) => {
    const feasible = run.points.filter((p) => p.feasible);
    const infeasible = run.points.filter((p) => !p.feasible);
    const col = feasibleColors[ri % feasibleColors.length];

    if (infeasible.length) {
      traces.push({
        x: infeasible.map((p) => p.x),
        y: infeasible.map((p) => p.y),
        type: "scatter",
        mode: "markers",
        name: `${run.label} (infeasible)`,
        marker: { color: COLORS.faint, size: 4, opacity: 0.5 },
      });
    }
    if (feasible.length) {
      traces.push({
        x: feasible.map((p) => p.x),
        y: feasible.map((p) => p.y),
        type: "scatter",
        mode: "markers",
        name: `${run.label}`,
        marker: { color: col, size: 5, opacity: 0.85 },
      });
    }
  });

  const allPoints = runs.flatMap((r) => r.points);
  const shapes: Partial<Plotly.Shape>[] = [];
  const region = feasibleRect(allPoints, targetX, targetY, goalX, goalY);
  if (region) shapes.push(region);

  if (targetX != null) {
    shapes.push({
      type: "line",
      x0: targetX,
      x1: targetX,
      xref: "x",
      y0: 0,
      y1: 1,
      yref: "paper",
      line: { color: COLORS.ok, dash: "dash", width: 1 },
    });
  }
  if (targetY != null) {
    shapes.push({
      type: "line",
      x0: 0,
      x1: 1,
      xref: "paper",
      y0: targetY,
      y1: targetY,
      yref: "y",
      line: { color: COLORS.ok, dash: "dash", width: 1 },
    });
  }

  return (
    <PlotlyChart
      data={traces}
      height={height}
      layout={{
        xaxis: { title: { text: metricX } },
        yaxis: { title: { text: metricY } },
        shapes,
      }}
    />
  );
}
