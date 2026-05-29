// Shared sketchy wireframe primitives for SpiceXplorer UX exploration.
// All components/values exported to window for cross-file use.

const SK = {
  ink: '#1c1a17',
  paper: '#faf8f3',
  panel: '#fffdf7',
  mute: '#9b9588',
  soft: '#e6e1d4',
  softer: '#f1ece0',
  accent: '#d97757',
  accentSoft: '#f3d6c4',
  blue: '#4a78c4',
  green: '#5e9e6e',
  red: '#c45a4a',
  amber: '#cf9a3a',
  fontHand: '"Kalam", "Patrick Hand", "Comic Sans MS", cursive',
  fontMono: '"JetBrains Mono", ui-monospace, "SFMono-Regular", monospace',
  fontUI: '"Kalam", system-ui, sans-serif',
};

// — Hand-style title
function SkTitle({ children, size = 22, color = SK.ink, style = {} }) {
  return (
    <span style={{ fontFamily: SK.fontHand, fontWeight: 700, fontSize: size, color, lineHeight: 1.1, ...style }}>
      {children}
    </span>
  );
}

// — Hand-style body / annotation
function SkText({ children, size = 14, color = SK.ink, style = {} }) {
  return (
    <span style={{ fontFamily: SK.fontHand, fontWeight: 400, fontSize: size, color, lineHeight: 1.25, ...style }}>
      {children}
    </span>
  );
}

// — Mono for technical values
function SkMono({ children, size = 11, color = SK.ink, style = {} }) {
  return (
    <span style={{ fontFamily: SK.fontMono, fontWeight: 400, fontSize: size, color, ...style }}>
      {children}
    </span>
  );
}

// — Sketchy box
function SkBox({ children, style = {}, dashed = false, accent = false, soft = false, thick = false }) {
  const color = accent ? SK.accent : SK.ink;
  return (
    <div style={{
      border: `${thick ? 2 : 1.5}px ${dashed ? 'dashed' : 'solid'} ${color}`,
      borderRadius: 5,
      background: soft ? SK.softer : 'transparent',
      padding: 10,
      ...style,
    }}>
      {children}
    </div>
  );
}

// — Sketchy pill / tag
function SkPill({ children, color = SK.ink, bg = 'transparent', style = {} }) {
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 4,
      border: `1.2px solid ${color}`,
      borderRadius: 999,
      padding: '1px 8px',
      background: bg,
      color,
      fontFamily: SK.fontHand,
      fontSize: 12,
      ...style,
    }}>
      {children}
    </span>
  );
}

// — Sketchy button
function SkBtn({ children, accent = false, ghost = false, style = {} }) {
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 6,
      border: `1.5px solid ${accent ? SK.accent : SK.ink}`,
      background: ghost ? 'transparent' : (accent ? SK.accent : SK.ink),
      color: ghost ? (accent ? SK.accent : SK.ink) : SK.paper,
      fontFamily: SK.fontHand,
      fontSize: 14,
      padding: '4px 12px',
      borderRadius: 4,
      ...style,
    }}>
      {children}
    </span>
  );
}

// — A row of dot-leader lines (placeholder text)
function SkLines({ count = 3, width = '100%', gap = 8, color = SK.mute, style = {} }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap, width, ...style }}>
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} style={{
          height: 1.5,
          background: color,
          opacity: 0.45,
          width: `${85 - (i % 3) * 15}%`,
          borderRadius: 1,
        }} />
      ))}
    </div>
  );
}

// — Sparkline (tiny convergence curve)
function SkSparkline({ width = 80, height = 18, color = SK.ink, kind = 'conv' }) {
  const sets = {
    conv: '0,17 6,16 12,14 18,11 24,9 30,8 36,7 42,6 48,5 54,5 60,4 66,4 72,4 78,4',
    convFast: '0,17 5,12 10,9 15,7 20,6 25,5 30,5 35,4 40,4 50,4 60,4 70,4 80,4',
    noisy: '0,12 6,9 12,15 18,7 24,11 30,4 36,9 42,6 48,3 54,8 60,5 66,7 72,4 78,6',
    flat: '0,11 12,10 24,11 36,10 48,11 60,10 72,11 80,10',
  };
  return (
    <svg width={width} height={height} style={{ overflow: 'visible' }}>
      <polyline points={sets[kind]} fill="none" stroke={color} strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// — A sketchy convergence chart placeholder
function SkChart({ width, height, lines = ['ink'], title, target = true, kind = 'conv', noAxis = false }) {
  const padL = 38, padB = 26, padT = 14, padR = 14;
  const w = width - padL - padR;
  const h = height - padT - padB;
  const curves = {
    ink: { color: SK.ink, d: pathFor(w, h, 'conv', 0) },
    blue: { color: SK.blue, d: pathFor(w, h, 'conv', 1) },
    accent: { color: SK.accent, d: pathFor(w, h, 'conv', 2) },
    green: { color: SK.green, d: pathFor(w, h, 'noisy', 3) },
  };
  return (
    <svg width={width} height={height} style={{ overflow: 'visible', display: 'block' }}>
      {title && (
        <text x={padL} y={padT - 2} style={{ fontFamily: SK.fontHand, fontSize: 12, fill: SK.ink }}>{title}</text>
      )}
      {!noAxis && <>
        <line x1={padL} y1={padT + h} x2={padL + w} y2={padT + h} stroke={SK.ink} strokeWidth={1.2} />
        <line x1={padL} y1={padT} x2={padL} y2={padT + h} stroke={SK.ink} strokeWidth={1.2} />
        <text x={padL + w / 2} y={padT + h + 18} style={{ fontFamily: SK.fontHand, fontSize: 11, fill: SK.mute, textAnchor: 'middle' }}>iteration</text>
        <text x={padL - 6} y={padT + h / 2} style={{ fontFamily: SK.fontHand, fontSize: 11, fill: SK.mute, textAnchor: 'middle' }} transform={`rotate(-90 ${padL - 26} ${padT + h / 2})`}>score</text>
      </>}
      {target && (
        <line x1={padL} y1={padT + h * 0.35} x2={padL + w} y2={padT + h * 0.35}
          stroke={SK.amber} strokeWidth={1.2} strokeDasharray="4 4" />
      )}
      <g transform={`translate(${padL}, ${padT})`}>
        {lines.map((l, i) => (
          <path key={i} d={curves[l]?.d || curves.ink.d} fill="none"
            stroke={curves[l]?.color || SK.ink} strokeWidth={1.6}
            strokeLinecap="round" strokeLinejoin="round" />
        ))}
      </g>
    </svg>
  );
}

function pathFor(w, h, kind, seed = 0) {
  // generate a smooth-ish path
  const pts = [];
  const n = 30;
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const x = t * w;
    let y;
    if (kind === 'conv') {
      const base = Math.exp(-2.2 * t - seed * 0.15) * 0.85 + 0.05;
      const noise = (Math.sin(i * 0.9 + seed) + Math.sin(i * 2.3 + seed * 2)) * 0.015;
      y = (base + noise) * h;
    } else if (kind === 'noisy') {
      const base = Math.exp(-1.6 * t - seed * 0.1) * 0.7 + 0.12;
      const noise = (Math.sin(i * 1.7 + seed) * 0.05 + Math.sin(i * 0.6) * 0.04);
      y = (base + noise) * h;
    } else {
      y = (0.5 + Math.sin(i) * 0.1) * h;
    }
    y = Math.max(2, Math.min(h - 2, y));
    pts.push([x, y]);
  }
  return 'M ' + pts.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join(' L ');
}

// — A scatter cloud placeholder (feasible / infeasible)
function SkScatter({ width, height, title }) {
  const padL = 38, padB = 26, padT = 14, padR = 14;
  const w = width - padL - padR;
  const h = height - padT - padB;
  const pts = [];
  for (let i = 0; i < 90; i++) {
    const x = (Math.random() * 0.9 + 0.05) * w;
    const y = (Math.random() * 0.9 + 0.05) * h;
    const feasible = (x / w) > 0.55 && (y / h) < 0.5;
    pts.push([x, y, feasible]);
  }
  return (
    <svg width={width} height={height} style={{ overflow: 'visible' }}>
      {title && (
        <text x={padL} y={padT - 2} style={{ fontFamily: SK.fontHand, fontSize: 12, fill: SK.ink }}>{title}</text>
      )}
      <line x1={padL} y1={padT + h} x2={padL + w} y2={padT + h} stroke={SK.ink} strokeWidth={1.2} />
      <line x1={padL} y1={padT} x2={padL} y2={padT + h} stroke={SK.ink} strokeWidth={1.2} />
      <line x1={padL + w * 0.55} y1={padT} x2={padL + w * 0.55} y2={padT + h} stroke={SK.amber} strokeDasharray="4 4" strokeWidth={1.2} />
      <line x1={padL} y1={padT + h * 0.5} x2={padL + w} y2={padT + h * 0.5} stroke={SK.amber} strokeDasharray="4 4" strokeWidth={1.2} />
      {pts.map(([x, y, f], i) => (
        <circle key={i} cx={padL + x} cy={padT + y} r={2.4}
          fill={f ? SK.accent : SK.mute} opacity={f ? 0.9 : 0.45} />
      ))}
    </svg>
  );
}

// — Penalty curve (sigmoid + linear)
function SkPenaltyCurve({ width, height }) {
  const padL = 30, padB = 22, padT = 10, padR = 10;
  const w = width - padL - padR;
  const h = height - padT - padB;
  // Linear: triangle around target. Sigmoid: smooth step.
  const lin = [];
  const sig = [];
  const N = 60;
  for (let i = 0; i <= N; i++) {
    const t = i / N;
    const x = t * w;
    const d = (t - 0.5) * 2; // -1..1
    const linY = Math.min(1, Math.abs(d) * 1.4) * h;
    const sigY = (1 - 1 / (1 + Math.exp(8 * (Math.abs(d) - 0.3)))) * h * 0.95 + h * 0.04;
    lin.push([x, h - linY]);
    sig.push([x, h - sigY]);
  }
  const toPath = (pts) => 'M ' + pts.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join(' L ');
  return (
    <svg width={width} height={height} style={{ overflow: 'visible', display: 'block' }}>
      <line x1={padL} y1={padT + h} x2={padL + w} y2={padT + h} stroke={SK.ink} strokeWidth={1.2} />
      <line x1={padL} y1={padT} x2={padL} y2={padT + h} stroke={SK.ink} strokeWidth={1.2} />
      <line x1={padL + w / 2} y1={padT} x2={padL + w / 2} y2={padT + h} stroke={SK.amber} strokeWidth={1.2} strokeDasharray="3 3" />
      <g transform={`translate(${padL}, ${padT})`}>
        <path d={toPath(lin)} fill="none" stroke={SK.blue} strokeWidth={1.6} />
        <path d={toPath(sig)} fill="none" stroke={SK.accent} strokeWidth={1.6} />
      </g>
      <text x={padL + w - 4} y={padT + 14} textAnchor="end" style={{ fontFamily: SK.fontHand, fontSize: 11, fill: SK.accent }}>sigmoid</text>
      <text x={padL + w - 4} y={padT + 28} textAnchor="end" style={{ fontFamily: SK.fontHand, fontSize: 11, fill: SK.blue }}>linear</text>
      <text x={padL + w / 2} y={padT + h + 14} textAnchor="middle" style={{ fontFamily: SK.fontHand, fontSize: 10, fill: SK.amber }}>target</text>
    </svg>
  );
}

// — Cascode OTA schematic sketch
function SkSchematic({ width = 360, height = 260, highlighted = null, clickable = false }) {
  // simple boxes-and-wires version
  const mosfet = (x, y, label, type = 'n', hi = false) => {
    const w = 28, h = 22;
    const color = hi ? SK.accent : SK.ink;
    const fill = hi ? SK.accentSoft : SK.panel;
    return (
      <g key={label}>
        <rect x={x} y={y} width={w} height={h} fill={fill} stroke={color} strokeWidth={hi ? 2 : 1.4} rx={3} />
        <text x={x + w / 2} y={y + h / 2 + 3} textAnchor="middle" style={{ fontFamily: SK.fontMono, fontSize: 10, fill: color, fontWeight: 600 }}>{label}</text>
        {type === 'p' && <circle cx={x - 4} cy={y + h / 2} r={1.6} fill={SK.paper} stroke={color} strokeWidth={1} />}
      </g>
    );
  };
  const wire = (x1, y1, x2, y2) => (
    <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={SK.ink} strokeWidth={1.2} />
  );
  // Cascode telescopic OTA approximation
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: 'block' }}>
      {/* VDD rail */}
      <line x1={20} y1={28} x2={width - 20} y2={28} stroke={SK.ink} strokeWidth={1.4} />
      <text x={28} y={22} style={{ fontFamily: SK.fontMono, fontSize: 10, fill: SK.mute }}>VDD</text>
      {/* GND rail */}
      <line x1={20} y1={height - 28} x2={width - 20} y2={height - 28} stroke={SK.ink} strokeWidth={1.4} />
      <text x={28} y={height - 14} style={{ fontFamily: SK.fontMono, fontSize: 10, fill: SK.mute }}>GND</text>

      {/* PMOS top (M3/M4) */}
      {mosfet(80, 44, 'M3', 'p', highlighted === 'M3M4')}
      {mosfet(width - 110, 44, 'M4', 'p', highlighted === 'M3M4')}
      {wire(94, 44, 94, 28)} {wire(width - 96, 44, width - 96, 28)}

      {/* PMOS cascode (M3C/M4C) */}
      {mosfet(80, 80, 'M3c', 'p', highlighted === 'M3CM4C')}
      {mosfet(width - 110, 80, 'M4c', 'p', highlighted === 'M3CM4C')}
      {wire(94, 66, 94, 80)} {wire(width - 96, 66, width - 96, 80)}

      {/* NMOS cascode (M1C/M2C) */}
      {mosfet(80, 122, 'M1c', 'n', highlighted === 'M1CM2C')}
      {mosfet(width - 110, 122, 'M2c', 'n', highlighted === 'M1CM2C')}
      {wire(94, 102, 94, 122)} {wire(width - 96, 102, width - 96, 122)}

      {/* NMOS input pair (M1/M2) */}
      {mosfet(80, 158, 'M1', 'n', highlighted === 'M1M2')}
      {mosfet(width - 110, 158, 'M2', 'n', highlighted === 'M1M2')}
      {wire(94, 144, 94, 158)} {wire(width - 96, 144, width - 96, 158)}

      {/* Tail M5 */}
      {mosfet(width / 2 - 14, 200, 'M5', 'n', highlighted === 'M5')}
      {wire(94, 180, width / 2 - 14, 211)} {wire(width - 96, 180, width / 2 + 14, 211)}
      {wire(width / 2, 222, width / 2, height - 28)}

      {/* Inputs */}
      <text x={50} y={172} style={{ fontFamily: SK.fontMono, fontSize: 10, fill: SK.ink }}>Vin+</text>
      <text x={width - 78} y={172} style={{ fontFamily: SK.fontMono, fontSize: 10, fill: SK.ink }}>Vin−</text>
      <line x1={68} y1={169} x2={80} y2={169} stroke={SK.ink} strokeWidth={1.2} />
      <line x1={width - 110} y1={169} x2={width - 78} y2={169} stroke={SK.ink} strokeWidth={1.2} />

      {/* Output */}
      <text x={width / 2 - 12} y={118} style={{ fontFamily: SK.fontMono, fontSize: 10, fill: SK.ink }}>Vout</text>
      <line x1={108} y1={108} x2={width - 110} y2={108} stroke={SK.ink} strokeWidth={1.2} />
      <line x1={width / 2} y1={108} x2={width / 2} y2={100} stroke={SK.ink} strokeWidth={1.2} />
      <circle cx={width / 2} cy={108} r={2.2} fill={SK.ink} />
    </svg>
  );
}

// — A status chip for a target spec
function SkSpecChip({ name, pass = true, value, target, unit = '', style = {} }) {
  const c = pass ? SK.green : SK.red;
  return (
    <div style={{
      border: `1.2px solid ${c}`,
      borderRadius: 4,
      padding: '4px 8px',
      background: pass ? '#eaf2ec' : '#f7e3e0',
      display: 'flex',
      flexDirection: 'column',
      gap: 1,
      minWidth: 80,
      ...style,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontFamily: SK.fontMono, fontSize: 10, color: SK.ink, fontWeight: 600 }}>{name}</span>
        <span style={{ fontFamily: SK.fontHand, fontSize: 12, color: c }}>{pass ? '✓' : '✗'}</span>
      </div>
      <span style={{ fontFamily: SK.fontMono, fontSize: 10, color: SK.ink }}>{value}<span style={{ color: SK.mute }}> / {target}{unit}</span></span>
    </div>
  );
}

// — Hand-style callout with an arrow pointing at something
function SkCallout({ children, x, y, w = 180, arrow = 'left', color = SK.accent }) {
  return (
    <div style={{
      position: 'absolute',
      left: x, top: y, width: w,
      pointerEvents: 'none',
      transform: 'rotate(-1deg)',
    }}>
      <div style={{
        fontFamily: SK.fontHand, fontSize: 13, color, lineHeight: 1.25,
        padding: '4px 6px',
      }}>{children}</div>
    </div>
  );
}

// expose
Object.assign(window, {
  SK, SkTitle, SkText, SkMono, SkBox, SkPill, SkBtn, SkLines, SkSparkline,
  SkChart, SkScatter, SkPenaltyCurve, SkSchematic, SkSpecChip, SkCallout,
});
