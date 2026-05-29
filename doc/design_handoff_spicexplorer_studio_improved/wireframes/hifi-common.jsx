// Hi-fi shared primitives for SpiceXplorer UX exploration.
// Aesthetic: clean engineering tool. Inter + JetBrains Mono. Indigo accent over zinc/stone neutrals.

const HF = {
  bg:        '#fafaf9',  // app bg
  panel:     '#ffffff',
  panelAlt:  '#f5f5f4',  // stone-100
  border:    '#e7e5e4',  // stone-200
  borderDk:  '#d6d3d1',  // stone-300
  hairline:  '#f5f5f4',
  ink:       '#1c1917',  // stone-900
  text:      '#292524',  // stone-800
  textMute:  '#57534e',  // stone-600
  textDim:   '#a8a29e',  // stone-400
  accent:    '#4f46e5',  // indigo-600
  accentHov: '#4338ca',  // indigo-700
  accentSoft:'#eef2ff',  // indigo-50
  accentMid: '#c7d2fe',  // indigo-200
  success:   '#16a34a',
  successSoft:'#dcfce7',
  warn:      '#d97706',
  warnSoft:  '#fef3c7',
  error:     '#dc2626',
  errorSoft: '#fee2e2',
  // dark inverse
  inkInv:    '#fafaf9',
  panelInv:  '#1c1917',
  borderInv: '#292524',
  // fonts
  ui:   '"Inter", system-ui, -apple-system, sans-serif',
  mono: '"JetBrains Mono", ui-monospace, "SFMono-Regular", monospace',
};

// — Reset/default text wrapping for hi-fi components
function HFText({ children, size = 13, color = HF.text, weight = 400, style = {}, family }) {
  return (
    <span style={{
      fontFamily: family || HF.ui, fontSize: size, color, fontWeight: weight,
      lineHeight: 1.4, letterSpacing: size <= 11 ? 0.01 : 0,
      ...style,
    }}>{children}</span>
  );
}

function HFMono({ children, size = 12, color = HF.text, weight = 400, style = {} }) {
  return <HFText family={HF.mono} size={size} color={color} weight={weight} style={style}>{children}</HFText>;
}

function HFButton({ children, kind = 'default', size = 'md', icon, style = {}, onClick, title, type = 'button' }) {
  const bg = kind === 'primary' ? HF.accent : kind === 'ghost' ? 'transparent' : HF.panel;
  const fg = kind === 'primary' ? '#fff' : HF.text;
  const bd = kind === 'primary' ? HF.accent : kind === 'ghost' ? 'transparent' : HF.border;
  const pad = size === 'sm' ? '4px 10px' : '6px 12px';
  const fs = size === 'sm' ? 12 : 13;
  return (
    <button type={type} onClick={onClick} title={title} style={{
      display: 'inline-flex', alignItems: 'center', gap: 6,
      background: bg, color: fg, border: `1px solid ${bd}`,
      padding: pad, borderRadius: 6, fontFamily: HF.ui, fontSize: fs, fontWeight: 500,
      cursor: 'pointer', boxShadow: kind === 'primary' ? '0 1px 0 rgba(0,0,0,0.06)' : 'none',
      ...style,
    }}>
      {icon && <span style={{ display: 'inline-flex', alignItems: 'center' }}>{icon}</span>}
      {children}
    </button>
  );
}

function HFBadge({ children, tone = 'neutral', style = {} }) {
  const tones = {
    neutral: { bg: HF.panelAlt, fg: HF.textMute, bd: HF.border },
    indigo:  { bg: HF.accentSoft, fg: HF.accent, bd: HF.accentMid },
    success: { bg: HF.successSoft, fg: HF.success, bd: '#bbf7d0' },
    warn:    { bg: HF.warnSoft, fg: HF.warn, bd: '#fde68a' },
    error:   { bg: HF.errorSoft, fg: HF.error, bd: '#fecaca' },
    ink:     { bg: HF.ink, fg: '#fff', bd: HF.ink },
  };
  const t = tones[tone] || tones.neutral;
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      background: t.bg, color: t.fg, border: `1px solid ${t.bd}`,
      padding: '1px 7px', borderRadius: 999,
      fontFamily: HF.ui, fontSize: 11, fontWeight: 500, lineHeight: 1.5,
      ...style,
    }}>{children}</span>
  );
}

function HFCard({ children, style = {}, pad = 14, accent = false }) {
  return (
    <div style={{
      background: HF.panel,
      border: `1px solid ${accent ? HF.accentMid : HF.border}`,
      borderRadius: 8, padding: pad,
      boxShadow: '0 1px 2px rgba(28,25,23,0.04)',
      ...style,
    }}>{children}</div>
  );
}

// SVG icons (lucide-inspired, minimal)
function HFIcon({ name, size = 14, color = 'currentColor', strokeWidth = 1.75 }) {
  const paths = {
    play:        <polygon points="6,4 20,12 6,20" fill={color} stroke="none" />,
    stop:        <rect x="6" y="6" width="12" height="12" fill={color} stroke="none" />,
    plus:        <><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></>,
    fork:        <><circle cx="6" cy="6" r="2.5" /><circle cx="18" cy="6" r="2.5" /><circle cx="12" cy="18" r="2.5" /><path d="M6 8.5 Q6 12 12 13 Q18 14 18 8.5 M12 13 v3" fill="none" /></>,
    check:       <polyline points="20,6 9,17 4,12" />,
    x:           <><line x1="6" y1="6" x2="18" y2="18" /><line x1="18" y1="6" x2="6" y2="18" /></>,
    chevR:       <polyline points="9,6 15,12 9,18" />,
    chevD:       <polyline points="6,9 12,15 18,9" />,
    search:      <><circle cx="11" cy="11" r="6" /><line x1="20" y1="20" x2="16" y2="16" /></>,
    folder:      <path d="M3 7 a2 2 0 0 1 2 -2 h4 l2 2 h8 a2 2 0 0 1 2 2 v8 a2 2 0 0 1 -2 2 h-14 a2 2 0 0 1 -2 -2 z" />,
    file:        <><path d="M14 3 H7 a2 2 0 0 0 -2 2 v14 a2 2 0 0 0 2 2 h10 a2 2 0 0 0 2 -2 V8 z" /><polyline points="14,3 14,8 19,8" /></>,
    chip:        <><rect x="6" y="6" width="12" height="12" rx="1" /><line x1="9" y1="3" x2="9" y2="6" /><line x1="15" y1="3" x2="15" y2="6" /><line x1="9" y1="18" x2="9" y2="21" /><line x1="15" y1="18" x2="15" y2="21" /><line x1="3" y1="9" x2="6" y2="9" /><line x1="3" y1="15" x2="6" y2="15" /><line x1="18" y1="9" x2="21" y2="9" /><line x1="18" y1="15" x2="21" y2="15" /></>,
    graph:       <><circle cx="6" cy="6" r="2.2" /><circle cx="18" cy="6" r="2.2" /><circle cx="12" cy="18" r="2.2" /><line x1="7.5" y1="7" x2="11" y2="16" /><line x1="16.5" y1="7" x2="13" y2="16" /></>,
    activity:    <polyline points="3,12 7,12 10,5 14,19 17,12 21,12" />,
    sliders:     <><line x1="4" y1="6" x2="20" y2="6" /><circle cx="9" cy="6" r="2" /><line x1="4" y1="12" x2="20" y2="12" /><circle cx="15" cy="12" r="2" /><line x1="4" y1="18" x2="20" y2="18" /><circle cx="7" cy="18" r="2" /></>,
    settings:    <><circle cx="12" cy="12" r="3" /><path d="M12 3 v2 M12 19 v2 M3 12 h2 M19 12 h2 M5.5 5.5 l1.4 1.4 M17.1 17.1 l1.4 1.4 M5.5 18.5 l1.4 -1.4 M17.1 6.9 l1.4 -1.4" /></>,
    git:         <><circle cx="6" cy="6" r="2" /><circle cx="6" cy="18" r="2" /><circle cx="18" cy="12" r="2" /><line x1="6" y1="8" x2="6" y2="16" /><path d="M6 12 h6 a4 4 0 0 0 4 -4 v-1" fill="none" /></>,
    diff:        <><polyline points="14,3 14,7 18,7" /><line x1="6" y1="11" x2="18" y2="11" /><line x1="6" y1="15" x2="14" y2="15" /></>,
    terminal:    <><polyline points="4,7 8,11 4,15" /><line x1="11" y1="15" x2="18" y2="15" /></>,
    flag:        <><line x1="5" y1="22" x2="5" y2="4" /><path d="M5 4 h11 l-2 4 l2 4 H5" fill="none" /></>,
    pin:         <path d="M12 2 v8 m-4 4 h8 l-4 8 z" />,
    dot:         <circle cx="12" cy="12" r="4" fill={color} stroke="none" />,
  };
  return (
    <svg width={size} height={size} viewBox="0 0 24 24"
      fill="none" stroke={color} strokeWidth={strokeWidth}
      strokeLinecap="round" strokeLinejoin="round" style={{ display: 'inline-block' }}>
      {paths[name]}
    </svg>
  );
}

// Spec status row (mini status with bar)
function HFSpecRow({ name, value, target, unit = '', pass = true, progress = 0.7 }) {
  const c = pass ? HF.success : HF.error;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 6, height: 6, borderRadius: 999, background: c }} />
          <HFMono size={12} color={HF.text} weight={500}>{name}</HFMono>
        </span>
        <HFMono size={11} color={HF.textMute}>{target}{unit}</HFMono>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <div style={{ flex: 1, height: 4, background: HF.panelAlt, borderRadius: 999, overflow: 'hidden' }}>
          <div style={{ width: `${progress * 100}%`, height: '100%', background: c }} />
        </div>
        <HFMono size={11} color={c} weight={500}>{value}</HFMono>
      </div>
    </div>
  );
}

function HFSpecChip({ name, value, target, pass = true, unit = '' }) {
  const c = pass ? HF.success : HF.error;
  const bg = pass ? HF.successSoft : HF.errorSoft;
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', gap: 2,
      padding: '6px 8px',
      background: HF.panel,
      border: `1px solid ${HF.border}`,
      borderRadius: 6,
      borderLeft: `3px solid ${c}`,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <HFMono size={11} color={HF.textMute} weight={500}>{name}</HFMono>
        <span style={{
          color: c, background: bg, padding: '0 5px',
          borderRadius: 999, fontSize: 10, fontFamily: HF.ui, fontWeight: 600,
        }}>{pass ? 'PASS' : 'FAIL'}</span>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <HFMono size={13} weight={600}>{value}</HFMono>
        <HFMono size={10} color={HF.textMute}>{target}{unit}</HFMono>
      </div>
    </div>
  );
}

// Tiny sparkline (cleaner than the sketchy one)
function HFSpark({ width = 96, height = 24, color = HF.accent, kind = 'conv', fill = false }) {
  const presets = {
    conv:     '0,22 4,21 9,19 14,16 20,13 26,10 32,8 38,7 44,6 50,5 56,5 64,4 72,4 80,4 90,4',
    convFast: '0,22 3,15 7,10 12,7 18,6 24,5 32,5 40,4 50,4 60,4 72,4 84,4 96,4',
    noisy:    '0,16 6,12 12,18 18,8 24,14 30,5 36,11 42,7 48,4 54,9 60,6 66,8 72,5 80,7 90,4',
    flat:     '0,14 12,13 24,14 36,13 48,14 60,13 72,14 84,13 96,14',
  };
  const pts = presets[kind] || presets.conv;
  // Build a filled version if needed
  let fillPath = null;
  if (fill) {
    const lastX = Math.max(...pts.split(' ').map(p => +p.split(',')[0]));
    fillPath = `M 0 ${height} L ${pts.split(' ').join(' L ')} L ${lastX} ${height} Z`;
  }
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: 'block' }}>
      {fill && <path d={fillPath} fill={color} opacity={0.12} />}
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.6}
        strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// Bigger convergence chart
function HFChart({ width, height, lines = [{ kind: 'conv', color: HF.text }], target = false, title, yLabel, xLabel = 'iteration', noAxis = false, bg = HF.panel }) {
  const padL = 42, padR = 14, padT = title ? 32 : 14, padB = 28;
  const w = width - padL - padR;
  const h = height - padT - padB;

  // Build path
  const pathFor = (kind, seed = 0) => {
    const pts = [];
    const n = 60;
    for (let i = 0; i <= n; i++) {
      const t = i / n;
      const x = t * w;
      let y;
      if (kind === 'conv') {
        const base = Math.exp(-2.4 * t - seed * 0.25) * 0.82 + 0.06;
        const noise = (Math.sin(i * 0.7 + seed * 1.5) * 0.012 + Math.sin(i * 2.1) * 0.006);
        y = (base + noise) * h;
      } else if (kind === 'raw') {
        const base = Math.exp(-1.6 * t - seed * 0.1) * 0.6 + 0.18;
        const noise = (Math.sin(i * 1.7 + seed) * 0.08 + Math.sin(i * 0.6) * 0.05);
        y = (base + noise) * h;
      } else if (kind === 'flat') {
        y = (0.55 + Math.sin(i * 0.4) * 0.04) * h;
      } else { y = h * 0.5; }
      y = Math.max(3, Math.min(h - 3, y));
      pts.push([x, y]);
    }
    return 'M ' + pts.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join(' L ');
  };

  // Tick marks
  const yTicks = [0, 0.25, 0.5, 0.75, 1];
  const xTicks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}
      style={{ display: 'block', fontFamily: HF.ui }}>
      <rect x={0} y={0} width={width} height={height} fill={bg} />
      {title && <text x={padL} y={20} fill={HF.text} fontSize={12} fontWeight={500}>{title}</text>}

      {/* gridlines */}
      {!noAxis && yTicks.map((t, i) => (
        <line key={i} x1={padL} y1={padT + h * (1 - t)} x2={padL + w} y2={padT + h * (1 - t)}
          stroke={HF.hairline} strokeWidth={1} />
      ))}

      {target && (
        <>
          <line x1={padL} y1={padT + h * 0.25} x2={padL + w} y2={padT + h * 0.25}
            stroke={HF.warn} strokeWidth={1} strokeDasharray="3 3" />
          <text x={padL + w - 4} y={padT + h * 0.25 - 4} fill={HF.warn} fontSize={10}
            textAnchor="end">target 200M</text>
        </>
      )}

      {/* axes */}
      {!noAxis && <>
        <line x1={padL} y1={padT + h} x2={padL + w} y2={padT + h} stroke={HF.borderDk} strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={padT + h} stroke={HF.borderDk} strokeWidth={1} />
        {yTicks.map((t, i) => (
          <text key={i} x={padL - 6} y={padT + h * (1 - t) + 3} fill={HF.textDim} fontSize={10} textAnchor="end" fontFamily={HF.mono}>
            {t.toFixed(2)}
          </text>
        ))}
        {xTicks.map((t, i) => (
          <text key={i} x={padL + w * t} y={padT + h + 14} fill={HF.textDim} fontSize={10} textAnchor="middle" fontFamily={HF.mono}>
            {Math.round(t * 2000)}
          </text>
        ))}
        <text x={padL + w / 2} y={height - 4} fill={HF.textMute} fontSize={10} textAnchor="middle">{xLabel}</text>
        {yLabel && (
          <text x={12} y={padT + h / 2} fill={HF.textMute} fontSize={10} textAnchor="middle"
            transform={`rotate(-90, 12, ${padT + h / 2})`}>{yLabel}</text>
        )}
      </>}

      {/* lines */}
      <g transform={`translate(${padL}, ${padT})`}>
        {lines.map((l, i) => (
          <path key={i} d={pathFor(l.kind || 'conv', l.seed ?? i * 0.3)} fill="none"
            stroke={l.color || HF.text} strokeWidth={l.width || 1.6}
            strokeOpacity={l.opacity ?? 1}
            strokeDasharray={l.dash ? '4 3' : 'none'}
            strokeLinecap="round" strokeLinejoin="round" />
        ))}
      </g>

      {/* legend */}
      {lines.some(l => l.label) && (
        <g transform={`translate(${padL + 10}, ${padT + 10})`}>
          {lines.filter(l => l.label).map((l, i) => (
            <g key={i} transform={`translate(0, ${i * 16})`}>
              <line x1={0} y1={6} x2={16} y2={6} stroke={l.color} strokeWidth={1.6} strokeDasharray={l.dash ? '4 3' : 'none'} />
              <text x={20} y={9} fill={HF.text} fontSize={11}>{l.label}</text>
            </g>
          ))}
        </g>
      )}
    </svg>
  );
}

// Cascode OTA schematic — hi-fi version with cleaner symbols
function HFSchematic({ width = 340, height = 260, highlighted = null, theme = 'light' }) {
  const ink = theme === 'dark' ? '#e7e5e4' : HF.text;
  const fill = theme === 'dark' ? HF.panelInv : HF.panel;
  const mute = theme === 'dark' ? '#78716c' : HF.textMute;

  const fet = (x, y, label, type = 'n', hi = false) => {
    const w = 30, h = 22;
    const stroke = hi ? HF.accent : ink;
    const bg = hi ? HF.accentSoft : fill;
    return (
      <g key={label}>
        <rect x={x} y={y} width={w} height={h} fill={bg} stroke={stroke} strokeWidth={hi ? 1.8 : 1.2} rx={2} />
        <text x={x + w / 2} y={y + h / 2 + 3.5} textAnchor="middle" fontSize={9.5} fontFamily={HF.mono} fontWeight={600} fill={stroke}>{label}</text>
        {type === 'p' && <circle cx={x - 3} cy={y + h / 2} r={1.4} fill={fill} stroke={stroke} strokeWidth={1} />}
      </g>
    );
  };
  const wire = (x1, y1, x2, y2) => <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={ink} strokeWidth={1.1} />;

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: 'block' }}>
      <line x1={20} y1={26} x2={width - 20} y2={26} stroke={ink} strokeWidth={1.4} />
      <text x={24} y={20} fontFamily={HF.mono} fontSize={9} fill={mute}>VDD</text>
      <line x1={20} y1={height - 26} x2={width - 20} y2={height - 26} stroke={ink} strokeWidth={1.4} />
      <text x={24} y={height - 12} fontFamily={HF.mono} fontSize={9} fill={mute}>GND</text>

      {fet(72, 40, 'M3', 'p', highlighted === 'M3M4')}
      {fet(width - 102, 40, 'M4', 'p', highlighted === 'M3M4')}
      {wire(87, 40, 87, 26)} {wire(width - 87, 40, width - 87, 26)}

      {fet(72, 74, 'M3c', 'p', highlighted === 'M3CM4C')}
      {fet(width - 102, 74, 'M4c', 'p', highlighted === 'M3CM4C')}
      {wire(87, 62, 87, 74)} {wire(width - 87, 62, width - 87, 74)}

      {fet(72, 114, 'M1c', 'n', highlighted === 'M1CM2C')}
      {fet(width - 102, 114, 'M2c', 'n', highlighted === 'M1CM2C')}
      {wire(87, 96, 87, 114)} {wire(width - 87, 96, width - 87, 114)}

      {fet(72, 148, 'M1', 'n', highlighted === 'M1M2')}
      {fet(width - 102, 148, 'M2', 'n', highlighted === 'M1M2')}
      {wire(87, 136, 87, 148)} {wire(width - 87, 136, width - 87, 148)}

      {fet(width / 2 - 15, 188, 'M5', 'n', highlighted === 'M5')}
      {wire(87, 170, width / 2 - 15, 200)}
      {wire(width - 87, 170, width / 2 + 15, 200)}
      {wire(width / 2, 210, width / 2, height - 26)}

      <text x={42} y={161} fontFamily={HF.mono} fontSize={9} fill={ink}>Vin+</text>
      <text x={width - 64} y={161} fontFamily={HF.mono} fontSize={9} fill={ink}>Vin−</text>
      <line x1={60} y1={159} x2={72} y2={159} stroke={ink} strokeWidth={1.1} />
      <line x1={width - 102} y1={159} x2={width - 64} y2={159} stroke={ink} strokeWidth={1.1} />

      <line x1={102} y1={100} x2={width - 102} y2={100} stroke={ink} strokeWidth={1.1} />
      <line x1={width / 2} y1={100} x2={width / 2} y2={92} stroke={ink} strokeWidth={1.1} />
      <circle cx={width / 2} cy={100} r={1.8} fill={ink} />
      <text x={width / 2 - 12} y={88} fontFamily={HF.mono} fontSize={9} fill={ink}>Vout</text>
    </svg>
  );
}

// Penalty curve (sigmoid vs linear), hi-fi
function HFPenalty({ width, height, theme = 'light' }) {
  const padL = 32, padR = 12, padT = 8, padB = 18;
  const w = width - padL - padR;
  const h = height - padT - padB;
  const lin = [], sig = [];
  const N = 80;
  for (let i = 0; i <= N; i++) {
    const t = i / N;
    const d = (t - 0.5) * 2;
    const x = t * w;
    const linY = Math.min(1, Math.abs(d) * 1.6) * h * 0.92 + h * 0.04;
    const sigY = (1 - 1 / (1 + Math.exp(9 * (Math.abs(d) - 0.32)))) * h * 0.92 + h * 0.04;
    lin.push([x, h - linY]);
    sig.push([x, h - sigY]);
  }
  const toPath = (pts) => 'M ' + pts.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join(' L ');
  const ink = theme === 'dark' ? '#e7e5e4' : HF.text;
  const dim = theme === 'dark' ? '#78716c' : HF.textDim;
  const lineCol = theme === 'dark' ? '#404040' : HF.hairline;
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: 'block', fontFamily: HF.ui }}>
      {[0.25, 0.5, 0.75, 1].map((t, i) => (
        <line key={i} x1={padL} y1={padT + h * (1 - t)} x2={padL + w} y2={padT + h * (1 - t)} stroke={lineCol} strokeWidth={1} />
      ))}
      <line x1={padL} y1={padT + h} x2={padL + w} y2={padT + h} stroke={ink} strokeOpacity={0.4} />
      <line x1={padL + w / 2} y1={padT} x2={padL + w / 2} y2={padT + h} stroke={HF.warn} strokeWidth={1} strokeDasharray="3 3" />
      <g transform={`translate(${padL}, ${padT})`}>
        <path d={toPath(lin)} fill="none" stroke="#0ea5e9" strokeWidth={1.8} />
        <path d={toPath(sig)} fill="none" stroke={HF.accent} strokeWidth={1.8} />
      </g>
      <text x={padL + 6} y={padT + 12} fontSize={10} fill={HF.accent} fontWeight={500}>sigmoid</text>
      <text x={padL + 6} y={padT + 24} fontSize={10} fill="#0ea5e9" fontWeight={500}>linear</text>
      <text x={padL + w / 2} y={padT + h + 12} fontSize={9} fill={HF.warn} textAnchor="middle">target</text>
      <text x={padL - 6} y={padT + h + 3} fontSize={9} fill={dim} textAnchor="end" fontFamily={HF.mono}>0</text>
      <text x={padL - 6} y={padT + 6} fontSize={9} fill={dim} textAnchor="end" fontFamily={HF.mono}>1</text>
    </svg>
  );
}

// Section header (small label) — used inside panels
function HFSectionLabel({ children, action, style = {} }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', ...style }}>
      <span style={{
        fontFamily: HF.ui, fontSize: 11, fontWeight: 600, letterSpacing: 0.4,
        color: HF.textMute, textTransform: 'uppercase',
      }}>{children}</span>
      {action}
    </div>
  );
}

Object.assign(window, {
  HF, HFText, HFMono, HFButton, HFBadge, HFCard, HFIcon,
  HFSpecRow, HFSpecChip, HFSpark, HFChart, HFSchematic, HFPenalty, HFSectionLabel,
});
