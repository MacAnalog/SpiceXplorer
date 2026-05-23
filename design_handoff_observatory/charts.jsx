// Themed SVG chart primitives for SpiceXplorer UI
// Each chart accepts a `theme` prop with colors and stroke widths,
// so the same chart looks Observatory / Bench / Manuscript appropriately.

// Theme shape:
//   {
//     bg, fg, muted, grid, axis, primary, secondary, tertiary, danger, ok,
//     gridDash: "none" | "2 3",
//     strokeWidth: number,
//     font: string,
//     monoFont: string,
//     labelSize: number,
//   }

const { useState, useEffect, useMemo, useRef } = React;

function niceTicks(min, max, count = 5) {
  const range = max - min;
  if (range === 0) return [min];
  const rough = range / count;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const norm = rough / mag;
  let step;
  if (norm < 1.5) step = 1 * mag;
  else if (norm < 3) step = 2 * mag;
  else if (norm < 7) step = 5 * mag;
  else step = 10 * mag;
  const ticks = [];
  const start = Math.ceil(min / step) * step;
  for (let v = start; v <= max + step * 0.001; v += step) ticks.push(v);
  return ticks;
}

function ChartFrame({ width, height, padding, theme, xDomain, yDomain, xLabel, yLabel, xTickFormat, yTickFormat, children, title, rightLabel }) {
  const { left = 44, right = 14, top = 14, bottom = 28 } = padding || {};
  const plotW = width - left - right;
  const plotH = height - top - bottom;
  const sx = v => left + ((v - xDomain[0]) / (xDomain[1] - xDomain[0])) * plotW;
  const sy = v => top + plotH - ((v - yDomain[0]) / (yDomain[1] - yDomain[0])) * plotH;
  const xTicks = niceTicks(xDomain[0], xDomain[1], 6);
  const yTicks = niceTicks(yDomain[0], yDomain[1], 5);
  const fmt = xTickFormat || (v => v.toString());
  const yfmt = yTickFormat || (v => v.toString());
  const labelSize = theme.labelSize || 10;

  return (
    <svg width={width} height={height} style={{ fontFamily: theme.monoFont, fontSize: labelSize, color: theme.muted, display: "block" }}>
      {title && (
        <text x={left} y={11} fill={theme.fg} style={{ fontFamily: theme.font, fontSize: labelSize + 1, fontWeight: 500 }}>{title}</text>
      )}
      {rightLabel && (
        <text x={width - right} y={11} textAnchor="end" fill={theme.muted} style={{ fontFamily: theme.monoFont, fontSize: labelSize - 1 }}>{rightLabel}</text>
      )}
      {/* Grid */}
      {yTicks.map((t, i) => (
        <line key={"yg" + i} x1={left} x2={left + plotW} y1={sy(t)} y2={sy(t)} stroke={theme.grid} strokeWidth={0.75} strokeDasharray={theme.gridDash} />
      ))}
      {xTicks.map((t, i) => (
        <line key={"xg" + i} x1={sx(t)} x2={sx(t)} y1={top} y2={top + plotH} stroke={theme.grid} strokeWidth={0.75} strokeDasharray={theme.gridDash} />
      ))}
      {/* Axes */}
      <line x1={left} x2={left + plotW} y1={top + plotH} y2={top + plotH} stroke={theme.axis} strokeWidth={1} />
      <line x1={left} x2={left} y1={top} y2={top + plotH} stroke={theme.axis} strokeWidth={1} />
      {/* Tick labels */}
      {yTicks.map((t, i) => (
        <text key={"yl" + i} x={left - 6} y={sy(t) + 3} textAnchor="end" fill={theme.muted}>{yfmt(t)}</text>
      ))}
      {xTicks.map((t, i) => (
        <text key={"xl" + i} x={sx(t)} y={top + plotH + 14} textAnchor="middle" fill={theme.muted}>{fmt(t)}</text>
      ))}
      {/* Plot content */}
      {children({ sx, sy, plotW, plotH, left, top })}
      {xLabel && (
        <text x={left + plotW / 2} y={height - 4} textAnchor="middle" fill={theme.muted} style={{ fontSize: labelSize - 1 }}>{xLabel}</text>
      )}
      {yLabel && (
        <text x={10} y={top + plotH / 2} textAnchor="middle" fill={theme.muted} transform={`rotate(-90, 10, ${top + plotH / 2})`} style={{ fontSize: labelSize - 1 }}>{yLabel}</text>
      )}
    </svg>
  );
}

function LinePath({ data, sx, sy, color, width = 1.5, dash, opacity = 1, glow = false }) {
  if (!data || data.length === 0) return null;
  const d = data.map((p, i) => (i === 0 ? "M" : "L") + sx(p.x).toFixed(2) + " " + sy(p.y).toFixed(2)).join(" ");
  return (
    <g>
      {glow && <path d={d} fill="none" stroke={color} strokeWidth={width * 4} opacity={0.18} strokeLinecap="round" strokeLinejoin="round" />}
      <path d={d} fill="none" stroke={color} strokeWidth={width} strokeDasharray={dash} opacity={opacity} strokeLinecap="round" strokeLinejoin="round" />
    </g>
  );
}

// === Score Convergence ===
function ScoreConvergenceChart({ width, height, theme, traj, showRaw = true, label, overlay, animate = false, iterMax }) {
  const [drawIter, setDrawIter] = useState(animate ? 0 : (iterMax ?? traj.raw.length));
  useEffect(() => {
    if (!animate) { setDrawIter(iterMax ?? traj.raw.length); return; }
    let raf, start;
    const total = iterMax ?? traj.raw.length;
    const step = ts => {
      if (!start) start = ts;
      const t = (ts - start) / 1500;
      const frac = Math.min(1, t);
      setDrawIter(Math.floor(frac * total));
      if (frac < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [animate, iterMax, traj]);

  const xDomain = [0, traj.raw.length - 1];
  const allRaw = traj.raw.slice(0, drawIter + 1);
  const allBest = traj.best.slice(0, drawIter + 1);
  const maxY = Math.max(...traj.raw) * 1.05;
  const yDomain = [0, maxY];
  const rawData = allRaw.map((v, i) => ({ x: i, y: v }));
  const bestData = allBest.map((v, i) => ({ x: i, y: v }));
  const overlayData = overlay ? overlay.best.slice(0, drawIter + 1).map((v, i) => ({ x: i, y: v })) : null;

  return (
    <ChartFrame width={width} height={height} theme={theme}
      xDomain={xDomain} yDomain={yDomain}
      title={label || "score · F(x) ↓"} rightLabel={`iter ${drawIter}/${traj.raw.length - 1}`}
      xTickFormat={v => Math.round(v).toString()}
      yTickFormat={v => v.toFixed(1)}>
      {({ sx, sy, plotW, top, plotH, left }) => (
        <>
          {/* Target line at zero */}
          <line x1={left} x2={left + plotW} y1={sy(0)} y2={sy(0)} stroke={theme.ok} strokeWidth={1} strokeDasharray="3 3" opacity={0.6} />
          {showRaw && <LinePath data={rawData} sx={sx} sy={sy} color={theme.muted} width={0.8} opacity={0.55} />}
          {overlayData && <LinePath data={overlayData} sx={sx} sy={sy} color={theme.secondary} width={theme.strokeWidth} />}
          <LinePath data={bestData} sx={sx} sy={sy} color={theme.primary} width={theme.strokeWidth} glow={theme.glow} />
        </>
      )}
    </ChartFrame>
  );
}

// === Metric Convergence ===
function MetricConvergenceChart({ width, height, theme, traj, target, unit, label, overlay, animate = false }) {
  const [drawIter, setDrawIter] = useState(animate ? 0 : traj.length);
  useEffect(() => {
    if (!animate) { setDrawIter(traj.length); return; }
    let raf, start;
    const step = ts => {
      if (!start) start = ts;
      const t = (ts - start) / 1500;
      const frac = Math.min(1, t);
      setDrawIter(Math.floor(frac * traj.length));
      if (frac < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [animate, traj]);

  const xDomain = [0, traj.length - 1];
  const allVals = [...traj, target];
  if (overlay) allVals.push(...overlay);
  const min = Math.min(...allVals) * 0.9;
  const max = Math.max(...allVals) * 1.1;
  const data = traj.slice(0, drawIter + 1).map((v, i) => ({ x: i, y: v }));
  const overlayData = overlay ? overlay.slice(0, drawIter + 1).map((v, i) => ({ x: i, y: v })) : null;

  return (
    <ChartFrame width={width} height={height} theme={theme}
      xDomain={xDomain} yDomain={[min, max]}
      title={label}
      xTickFormat={v => Math.round(v).toString()}
      yTickFormat={v => window.SPICE_FMT.formatEng(v, 2)}>
      {({ sx, sy, plotW, left }) => (
        <>
          <line x1={left} x2={left + plotW} y1={sy(target)} y2={sy(target)} stroke={theme.danger} strokeWidth={1} strokeDasharray="4 3" />
          <text x={left + plotW - 4} y={sy(target) - 3} textAnchor="end" fill={theme.danger} style={{ fontSize: 9 }}>target {window.SPICE_FMT.formatEng(target, 2)}{unit}</text>
          {overlayData && <LinePath data={overlayData} sx={sx} sy={sy} color={theme.secondary} width={theme.strokeWidth} />}
          <LinePath data={data} sx={sx} sy={sy} color={theme.primary} width={theme.strokeWidth} glow={theme.glow} />
        </>
      )}
    </ChartFrame>
  );
}

// === Penalty Curve (sigmoid vs linear) ===
function PenaltyCurveChart({ width, height, theme, spec, currentValue }) {
  const curves = window.SPICE_DATA.penaltyCurves(spec);
  const xDomain = [curves.xs[0], curves.xs[curves.xs.length - 1]];
  const yDomain = [0, 1.05];
  const sigData = curves.xs.map((x, i) => ({ x, y: curves.sig[i] }));
  const linData = curves.xs.map((x, i) => ({ x, y: curves.lin[i] }));

  // P̂ at current value
  const dir = spec.goal === ">" ? (spec.target - currentValue) : (currentValue - spec.target);
  const sigP = 1 / (1 + Math.exp(-dir / (spec.range * 0.35)));
  const linP = Math.max(0, Math.min(1, dir / spec.range));

  return (
    <ChartFrame width={width} height={height} theme={theme}
      xDomain={xDomain} yDomain={yDomain}
      title={`P̂(${spec.name}) · sigmoid vs linear`}
      xTickFormat={v => window.SPICE_FMT.formatEng(v, 2)}
      yTickFormat={v => v.toFixed(2)}>
      {({ sx, sy, plotW, plotH, top, left }) => (
        <>
          {/* Target marker */}
          <line x1={sx(spec.target)} x2={sx(spec.target)} y1={top} y2={top + plotH} stroke={theme.ok} strokeDasharray="3 3" />
          <text x={sx(spec.target) + 4} y={top + 10} fill={theme.ok} style={{ fontSize: 9 }}>target</text>
          {/* Current marker */}
          <line x1={sx(currentValue)} x2={sx(currentValue)} y1={top} y2={top + plotH} stroke={theme.tertiary} strokeWidth={1.2} />
          <text x={sx(currentValue) + 4} y={top + 22} fill={theme.tertiary} style={{ fontSize: 9 }}>now {window.SPICE_FMT.formatEng(currentValue, 3)}{spec.unit}</text>
          <LinePath data={linData} sx={sx} sy={sy} color={theme.secondary} width={theme.strokeWidth} dash="4 3" />
          <LinePath data={sigData} sx={sx} sy={sy} color={theme.primary} width={theme.strokeWidth} glow={theme.glow} />
          {/* Legend */}
          <g transform={`translate(${left + plotW - 110}, ${top + 8})`}>
            <rect x={-6} y={-2} width={108} height={32} fill={theme.bg} opacity={0.85} rx={3} />
            <line x1={0} x2={14} y1={6} y2={6} stroke={theme.primary} strokeWidth={theme.strokeWidth} />
            <text x={20} y={9} fill={theme.fg} style={{ fontSize: 10 }}>sigmoid · {sigP.toFixed(2)}</text>
            <line x1={0} x2={14} y1={20} y2={20} stroke={theme.secondary} strokeWidth={theme.strokeWidth} strokeDasharray="4 3" />
            <text x={20} y={23} fill={theme.fg} style={{ fontSize: 10 }}>linear · {linP.toFixed(2)}</text>
          </g>
        </>
      )}
    </ChartFrame>
  );
}

// === Scatter ===
function ScatterChart({ width, height, theme, pointsA, pointsB, xKey = "x", yKey = "y", xLabel, yLabel, xTarget, yTarget, xFmt, yFmt, xGoal = ">", yGoal = "<" }) {
  const all = [...(pointsA || []), ...(pointsB || [])];
  const xs = all.map(p => p[xKey]);
  const ys = all.map(p => p[yKey]);
  const xDomain = [Math.min(...xs) * 0.9, Math.max(...xs) * 1.05];
  const yDomain = [Math.min(...ys) * 0.9, Math.max(...ys) * 1.05];

  return (
    <ChartFrame width={width} height={height} theme={theme}
      xDomain={xDomain} yDomain={yDomain}
      title={`${xLabel} vs ${yLabel} · feasibility`}
      xTickFormat={xFmt || (v => window.SPICE_FMT.formatEng(v, 2))}
      yTickFormat={yFmt || (v => window.SPICE_FMT.formatEng(v, 2))}>
      {({ sx, sy, plotW, plotH, top, left }) => (
        <>
          {/* Feasible region shading */}
          {xTarget && yTarget && (
            <rect
              x={xGoal === ">" ? sx(xTarget) : sx(xDomain[0])}
              y={yGoal === "<" ? sy(yDomain[1]) : sy(yTarget)}
              width={xGoal === ">" ? sx(xDomain[1]) - sx(xTarget) : sx(xTarget) - sx(xDomain[0])}
              height={yGoal === "<" ? sy(yTarget) - sy(yDomain[1]) : sy(yDomain[0]) - sy(yTarget)}
              fill={theme.ok} opacity={0.05} />
          )}
          {xTarget && (
            <line x1={sx(xTarget)} x2={sx(xTarget)} y1={top} y2={top + plotH} stroke={theme.ok} strokeDasharray="3 3" opacity={0.7} />
          )}
          {yTarget && (
            <line x1={left} x2={left + plotW} y1={sy(yTarget)} y2={sy(yTarget)} stroke={theme.ok} strokeDasharray="3 3" opacity={0.7} />
          )}
          {(pointsB || []).map((p, i) => (
            <circle key={"b" + i} cx={sx(p[xKey])} cy={sy(p[yKey])} r={1.8}
              fill={p.feasible ? theme.secondary : theme.muted}
              opacity={p.feasible ? 0.8 : 0.35} />
          ))}
          {(pointsA || []).map((p, i) => (
            <circle key={"a" + i} cx={sx(p[xKey])} cy={sy(p[yKey])} r={2.2}
              fill={p.feasible ? theme.primary : theme.muted}
              opacity={p.feasible ? 0.95 : 0.5} />
          ))}
        </>
      )}
    </ChartFrame>
  );
}

// === Histogram ===
function HistogramChart({ width, height, theme, dataA, dataB, target, label, bins = 24, xFmt }) {
  const all = [...(dataA || []), ...(dataB || [])];
  const min = Math.min(...all);
  const max = Math.max(...all);
  const binW = (max - min) / bins;
  const histA = new Array(bins).fill(0);
  const histB = new Array(bins).fill(0);
  (dataA || []).forEach(v => { const idx = Math.min(bins - 1, Math.max(0, Math.floor((v - min) / binW))); histA[idx]++; });
  (dataB || []).forEach(v => { const idx = Math.min(bins - 1, Math.max(0, Math.floor((v - min) / binW))); histB[idx]++; });
  const maxCount = Math.max(...histA, ...histB);

  return (
    <ChartFrame width={width} height={height} theme={theme}
      xDomain={[min, max]} yDomain={[0, maxCount * 1.1]}
      title={label}
      xTickFormat={xFmt || (v => window.SPICE_FMT.formatEng(v, 2))}
      yTickFormat={v => Math.round(v).toString()}>
      {({ sx, sy, plotW, plotH, top, left }) => {
        const bw = Math.max(1, plotW / bins - 1);
        return (
          <>
            {target && (
              <line x1={sx(target)} x2={sx(target)} y1={top} y2={top + plotH} stroke={theme.danger} strokeDasharray="3 3" />
            )}
            {histB.map((c, i) => (
              <rect key={"hb" + i} x={sx(min + i * binW)} y={sy(c)}
                width={bw} height={sy(0) - sy(c)} fill={theme.secondary} opacity={0.55} />
            ))}
            {histA.map((c, i) => (
              <rect key={"ha" + i} x={sx(min + i * binW)} y={sy(c)}
                width={bw} height={sy(0) - sy(c)} fill={theme.primary} opacity={0.7} />
            ))}
          </>
        );
      }}
    </ChartFrame>
  );
}

// === Schematic placeholder (mini cascode OTA-ish) ===
function SchematicPanel({ width, height, theme, accent }) {
  const W = width, H = height;
  const ink = theme.fg;
  const wire = theme.muted;
  const acc = accent || theme.primary;
  // Simple stylized cascode OTA: two diff pairs + cascode + current mirror
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }}>
      <rect x={0} y={0} width={W} height={H} fill={theme.bg} />
      {/* VDD bar */}
      <line x1={20} x2={W - 20} y1={18} y2={18} stroke={wire} strokeWidth={1.2} />
      <text x={W - 18} y={14} fill={theme.muted} style={{ fontFamily: theme.monoFont, fontSize: 8 }}>VDD</text>
      {/* GND bar */}
      <line x1={20} x2={W - 20} y1={H - 18} y2={H - 18} stroke={wire} strokeWidth={1.2} />
      <text x={W - 18} y={H - 10} fill={theme.muted} style={{ fontFamily: theme.monoFont, fontSize: 8 }}>GND</text>
      {/* Current mirror (top) */}
      {[0, 1].map(i => {
        const x = 60 + i * 80;
        return (
          <g key={"top" + i}>
            <line x1={x} x2={x} y1={18} y2={38} stroke={wire} />
            <rect x={x - 8} y={38} width={16} height={20} fill="none" stroke={ink} strokeWidth={1} />
            <line x1={x - 10} x2={x - 8} y1={48} y2={48} stroke={ink} />
            <line x1={x} x2={x} y1={58} y2={78} stroke={wire} />
          </g>
        );
      })}
      {/* Cascode (top) */}
      {[0, 1].map(i => {
        const x = 60 + i * 80;
        return (
          <g key={"tcas" + i}>
            <rect x={x - 8} y={78} width={16} height={20} fill="none" stroke={ink} strokeWidth={1} />
            <line x1={x - 10} x2={x - 8} y1={88} y2={88} stroke={ink} />
            <line x1={x} x2={x} y1={98} y2={118} stroke={wire} />
          </g>
        );
      })}
      {/* Cascode (bottom) */}
      {[0, 1].map(i => {
        const x = 60 + i * 80;
        return (
          <g key={"bcas" + i}>
            <rect x={x - 8} y={118} width={16} height={20} fill="none" stroke={ink} strokeWidth={1} />
            <line x1={x - 10} x2={x - 8} y1={128} y2={128} stroke={ink} />
            <line x1={x} x2={x} y1={138} y2={158} stroke={wire} />
          </g>
        );
      })}
      {/* Diff pair */}
      {[0, 1].map(i => {
        const x = 60 + i * 80;
        return (
          <g key={"diff" + i}>
            <rect x={x - 8} y={158} width={16} height={20} fill="none" stroke={ink} strokeWidth={1} />
            <line x1={x - 10} x2={x - 8} y1={168} y2={168} stroke={ink} />
            <line x1={x} x2={x} y1={178} y2={H - 38} stroke={wire} />
          </g>
        );
      })}
      {/* Tail */}
      <line x1={100} x2={140} y1={H - 38} y2={H - 38} stroke={wire} />
      <rect x={112} y={H - 38} width={16} height={16} fill="none" stroke={ink} strokeWidth={1} />
      <line x1={120} x2={120} y1={H - 22} y2={H - 18} stroke={wire} />
      {/* Inputs */}
      <line x1={20} x2={52} y1={168} y2={168} stroke={acc} strokeWidth={1.2} />
      <line x1={W - 20} x2={W - 52} y1={168} y2={168} stroke={acc} strokeWidth={1.2} />
      <text x={22} y={163} fill={acc} style={{ fontFamily: theme.monoFont, fontSize: 8 }}>vin+</text>
      <text x={W - 36} y={163} fill={acc} style={{ fontFamily: theme.monoFont, fontSize: 8 }}>vin-</text>
      {/* Output node */}
      <line x1={60} x2={W / 2 + 30} y1={98} y2={98} stroke={acc} strokeWidth={1.2} />
      <circle cx={W / 2 + 30} cy={98} r={2.5} fill={acc} />
      <text x={W / 2 + 36} y={101} fill={acc} style={{ fontFamily: theme.monoFont, fontSize: 8 }}>vout</text>
      {/* Caption */}
      <text x={W / 2} y={H - 3} textAnchor="middle" fill={theme.muted} style={{ fontFamily: theme.monoFont, fontSize: 8 }}>cascode_ota · ihp-sg13g2</text>
    </svg>
  );
}

// === Tiny inline sparkline ===
function Sparkline({ data, width, height, color, fill, target }) {
  if (!data || data.length === 0) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const xs = i => (i / (data.length - 1)) * width;
  const ys = v => height - ((v - min) / Math.max(0.0001, max - min)) * height;
  const d = data.map((v, i) => (i === 0 ? "M" : "L") + xs(i).toFixed(1) + " " + ys(v).toFixed(1)).join(" ");
  const dFill = d + ` L ${width} ${height} L 0 ${height} Z`;
  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      {fill && <path d={dFill} fill={fill} opacity={0.25} />}
      {target !== undefined && (
        <line x1={0} x2={width} y1={ys(target)} y2={ys(target)} stroke={color} strokeDasharray="2 2" opacity={0.5} />
      )}
      <path d={d} fill="none" stroke={color} strokeWidth={1.2} />
    </svg>
  );
}

Object.assign(window, {
  ChartFrame, LinePath, ScoreConvergenceChart, MetricConvergenceChart,
  PenaltyCurveChart, ScatterChart, HistogramChart, SchematicPanel, Sparkline,
});
