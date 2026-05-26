// Studio center-tab views. Each reads state from useStudio() and provides interactivity.

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────
function studioCompPenalty(spec, tryValue) {
  // Approximate the penalty for a try-value, given the spec's target & range.
  // Returns { linear, sigmoid } in [0..1].
  const t = spec.target;
  const r = spec.range || Math.abs(t) * 0.3 || 1;
  let signedErr;
  if (spec.goal === 'exceed')      signedErr = Math.max(0, t - tryValue) / r;
  else if (spec.goal === 'minimize') signedErr = Math.max(0, tryValue - t) / r;
  else                              signedErr = Math.abs(tryValue - t) / r;
  const lin = Math.min(1, Math.max(0, signedErr));
  const sig = 1 - 1 / (1 + Math.exp(9 * (signedErr - 0.32)));
  return { linear: lin, sigmoid: Math.max(0, Math.min(1, sig)) };
}

function studioSliderValue(spec, ratio /* 0..1 */) {
  // Map slider 0..1 to a numeric try-value around the target (target - 1.5*range to target + 1.5*range)
  const r = spec.range || Math.abs(spec.target) * 0.5 || 1;
  return spec.target - 1.5 * r + ratio * 3 * r;
}

function studioFormatValue(spec, val) {
  if (spec.unit === 'Hz')   return `${(val / 1e6).toFixed(1)} MHz`;
  if (spec.unit === 'dB')   return `${val.toFixed(2)} dB`;
  if (spec.unit === '°')    return `${val.toFixed(1)}°`;
  if (spec.unit === 'V/√Hz') return `${(val * 1000).toFixed(2)} mV/√Hz`;
  if (spec.unit === 'A')    return `${(val * 1e6).toFixed(2)} µA`;
  if (spec.unit === 's')    return `${(val * 1e6).toFixed(2)} µs`;
  return `${val.toFixed(3)}`;
}

// Penalty curve renderer that responds to try-value position
function StudioPenaltyCurve({ width, height, spec, tryValue }) {
  const padL = 36, padR = 12, padT = 12, padB = 22;
  const w = width - padL - padR;
  const h = height - padT - padB;
  const r = spec.range || Math.abs(spec.target) * 0.5 || 1;
  const lo = spec.target - 1.5 * r;
  const hi = spec.target + 1.5 * r;
  const lin = [], sig = [];
  const N = 80;
  for (let i = 0; i <= N; i++) {
    const t = i / N;
    const v = lo + t * (hi - lo);
    const p = studioCompPenalty(spec, v);
    lin.push([t * w, h - p.linear * h * 0.95 - h * 0.025]);
    sig.push([t * w, h - p.sigmoid * h * 0.95 - h * 0.025]);
  }
  const toPath = (pts) => 'M ' + pts.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join(' L ');
  const targetX = padL + (spec.target - lo) / (hi - lo) * w;
  const tryX    = padL + Math.max(0, Math.min(1, (tryValue - lo) / (hi - lo))) * w;
  const cur = studioCompPenalty(spec, tryValue);

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: 'block' }}>
      {/* gridlines */}
      {[0.25, 0.5, 0.75, 1].map((t, i) => (
        <line key={i} x1={padL} y1={padT + h * (1 - t)} x2={padL + w} y2={padT + h * (1 - t)} stroke={HF.hairline} strokeWidth={1} />
      ))}
      <line x1={padL} y1={padT + h} x2={padL + w} y2={padT + h} stroke={HF.borderDk} />

      {/* target marker */}
      <line x1={targetX} y1={padT} x2={targetX} y2={padT + h} stroke={HF.warn} strokeWidth={1} strokeDasharray="3 3" />
      <text x={targetX} y={padT - 2} fill={HF.warn} fontSize={10} textAnchor="middle">target</text>

      {/* try-value marker */}
      <line x1={tryX} y1={padT} x2={tryX} y2={padT + h} stroke={HF.text} strokeWidth={1.4} />
      <circle cx={tryX} cy={padT + h - cur.linear * h * 0.95 - h * 0.025} r={3.5} fill="#0ea5e9" stroke="#fff" strokeWidth={1.4} />
      <circle cx={tryX} cy={padT + h - cur.sigmoid * h * 0.95 - h * 0.025} r={3.5} fill={HF.accent} stroke="#fff" strokeWidth={1.4} />

      <g transform={`translate(${padL}, ${padT})`}>
        <path d={toPath(lin)} fill="none" stroke="#0ea5e9" strokeWidth={1.8} />
        <path d={toPath(sig)} fill="none" stroke={HF.accent} strokeWidth={1.8} />
      </g>

      {/* legend */}
      <g transform={`translate(${padL + 8}, ${padT + 6})`}>
        <line x1={0} y1={6} x2={14} y2={6} stroke={HF.accent} strokeWidth={2} />
        <text x={18} y={9} fontSize={10} fill={HF.text}>sigmoid</text>
        <line x1={66} y1={6} x2={80} y2={6} stroke="#0ea5e9" strokeWidth={2} />
        <text x={84} y={9} fontSize={10} fill={HF.text}>linear</text>
      </g>

      {/* axis label */}
      <text x={padL} y={padT + h + 14} fill={HF.textDim} fontSize={10} fontFamily={HF.mono}>{studioFormatValue(spec, lo)}</text>
      <text x={padL + w} y={padT + h + 14} fill={HF.textDim} fontSize={10} fontFamily={HF.mono} textAnchor="end">{studioFormatValue(spec, hi)}</text>
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────
// Pipeline view (interactive: click spec nodes to select)
// ─────────────────────────────────────────────────────────────
function StudioPipelineView({ width, height }) {
  const { s, set, specs } = useStudio();

  const PNode = ({ x, y, w = 168, kind, title, status, accent = false, selected = false, sub, onClick }) => (
    <div onClick={onClick} style={{
      position: 'absolute', left: x, top: y, width: w,
      background: HF.panel,
      border: `1px solid ${selected ? HF.accent : accent ? HF.accentMid : HF.border}`,
      borderRadius: 7,
      boxShadow: selected ? `0 0 0 2px ${HF.accentSoft}` : '0 1px 2px rgba(28,25,23,0.04)',
      padding: 8, display: 'flex', flexDirection: 'column', gap: 2,
      cursor: onClick ? 'pointer' : 'default',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
        <HFMono size={9} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase' }}>{kind}</HFMono>
        <div style={{ flex: 1 }} />
        {status === 'live' && <span style={{ width: 6, height: 6, borderRadius: 999, background: HF.accent }} />}
        {status === 'ok'   && <span style={{ width: 6, height: 6, borderRadius: 999, background: HF.success }} />}
      </div>
      <HFText size={12} weight={600}>{title}</HFText>
      {sub && <HFMono size={9} color={HF.textMute}>{sub}</HFMono>}
    </div>
  );

  const wire = (x1, y1, x2, y2, color = HF.borderDk, dashed = false) => {
    const dx = Math.max(20, Math.abs(x2 - x1) * 0.45);
    return <path d={`M ${x1} ${y1} C ${x1 + dx} ${y1} ${x2 - dx} ${y2} ${x2} ${y2}`}
      fill="none" stroke={color} strokeWidth={1.4} strokeDasharray={dashed ? '4 3' : 'none'} />;
  };

  const selSpec = specs.find(sp => sp.name === s.selectedSpec) || specs[0];

  return (
    <div style={{
      width, height, position: 'relative', overflow: 'hidden',
      backgroundImage: `radial-gradient(${HF.borderDk} 0.7px, transparent 0.7px)`,
      backgroundSize: '16px 16px', backgroundColor: HF.bg,
    }}>
      <svg width="100%" height="100%" style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
        {wire(190, 76, 232, 196)}{wire(190, 156, 232, 206)}{wire(190, 236, 232, 216)}
        {wire(400, 206, 442, 76)}{wire(400, 206, 442, 156)}{wire(400, 206, 442, 236)}
        {wire(610, 76, 654, 56, HF.accent)}{wire(610, 76, 654, 116, HF.accent)}{wire(610, 76, 654, 176, HF.accent)}
        {wire(610, 156, 654, 236)}{wire(610, 236, 654, 296)}{wire(610, 236, 654, 356)}
        {wire(822, 56, 866, 196)}{wire(822, 116, 866, 196)}{wire(822, 176, 866, 196)}
        {wire(822, 236, 866, 206)}{wire(822, 296, 866, 216)}{wire(822, 356, 866, 216)}
        {wire(1034, 206, 1078, 206)}
        <path d={`M 1140 240 C 1170 320 660 410 232 250`} fill="none" stroke={HF.accent} strokeWidth={1.6} strokeDasharray="5 4" opacity={0.85} />
      </svg>

      <PNode x={64} y={50}  kind="input" title="Netlist" status="ok" sub="ota-improved.spice" />
      <PNode x={64} y={130} kind="input" title="PDK rules" status="ok" sub="ihp-sg13g2" />
      <PNode x={64} y={210} kind="input" title="PVT corner" status="ok" sub="tt · 25°C · 1.5 V" />
      <PNode x={232} y={180} kind="variables" title="DUT parameters" status="ok" sub="13 vars · 1 integer"
        onClick={() => set({ activeActivity: 'setup', activeTab: 'yaml' })} />

      <PNode x={442} y={50}  kind="simulation" title="tb_ac"    status="ok" sub="→ ugf · gain · pm" />
      <PNode x={442} y={130} kind="simulation" title="tb_noise" status="ok" sub="→ inoise · idd" />
      <PNode x={442} y={210} kind="simulation" title="tb_tran"  status="ok" sub="→ tsettle" />

      {specs.map((sp, i) => (
        <PNode key={sp.name} x={654} y={30 + i * 60} w={168} kind={`spec · ${sp.shape}`}
          title={sp.name} status="ok" accent={sp.name === s.selectedSpec}
          selected={sp.name === s.selectedSpec}
          sub={sp.goal === 'exceed' ? `≥ ${sp.target_str}` : sp.goal === 'minimize' ? `≤ ${sp.target_str}` : `= ${sp.target_str}`}
          onClick={() => set({ selectedSpec: sp.name })} />
      ))}

      <PNode x={866} y={180} kind="scoring" title="Aggregate F(x)" status="ok" sub="weighted log sum  ·  0.412" />
      <PNode x={1078} y={180} kind="optimizer" title={`${s.algo} · seed 48`} status={s.isRunning ? 'live' : 'idle'} accent sub={`${847} / ${s.budget} · 42%`}
        onClick={() => set({ activeActivity: 'pipeline' })} />

      <div style={{ position: 'absolute', left: 580, top: 410 }}>
        <HFBadge tone="indigo" style={{ background: HF.panel }}>↺ optimizer rewrites DUT params each iter</HFBadge>
      </div>

      {/* Inline spec inspector */}
      <div style={{
        position: 'absolute', right: 16, bottom: 16, width: 316,
        background: HF.panel, border: `1px solid ${HF.accentMid}`, borderRadius: 9,
        boxShadow: '0 6px 18px rgba(28,25,23,0.10)', padding: 12,
        display: 'flex', flexDirection: 'column', gap: 8,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <div>
            <HFText size={10} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase', letterSpacing: 0.4 }}>Spec node</HFText>
            <HFText size={14} weight={600}>{selSpec.name} · {selSpec.desc}</HFText>
          </div>
          <HFButton size="sm" kind="ghost" style={{ padding: '2px 6px' }} onClick={() => set({ activeTab: 'shaping' })}>Open shaping →</HFButton>
        </div>
        <div style={{ display: 'flex', gap: 5 }}>
          {['sigmoid', 'linear', 'none'].map((sh) => (
            <span key={sh} onClick={() => set({ specShape: sh })} style={{ cursor: 'pointer' }}>
              <HFBadge tone={s.specShape === sh ? 'indigo' : 'neutral'}>{sh}</HFBadge>
            </span>
          ))}
        </div>
        <StudioPenaltyCurve width={290} height={80} spec={selSpec} tryValue={selSpec.value_num} />
        <HFMono size={10} color={HF.textMute}>
          {selSpec.goal} {selSpec.target_str} · w={selSpec.weight} · reward log
        </HFMono>
      </div>

      {/* canvas controls */}
      <div style={{ position: 'absolute', top: 12, left: 12, display: 'flex', flexDirection: 'column', gap: 4 }}>
        {['+', '−', '⊙'].map((c) => (
          <div key={c} style={{
            width: 28, height: 28, background: HF.panel, border: `1px solid ${HF.border}`,
            borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontFamily: HF.ui, fontSize: 14, color: HF.text,
          }}>{c}</div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Convergence view (current-run charts)
// ─────────────────────────────────────────────────────────────
function StudioConvergenceView({ width, height }) {
  const { s, set, specs, runs } = useStudio();
  const run = runs.find(r => r.id === s.selectedRunId) || runs[0];
  const [metric, setMetric] = React.useState('ugf');
  const [overlay, setOverlay] = React.useState(true);
  const [logScale, setLogScale] = React.useState(true);

  return (
    <div style={{ width, height, padding: 16, overflow: 'hidden', display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <HFText size={14} weight={600}>{run.id.toUpperCase()} · {run.name}</HFText>
        <HFBadge tone={run.status === 'live' ? 'indigo' : run.status === 'fail' ? 'error' : 'success'}>{run.status}</HFBadge>
        <div style={{ flex: 1 }} />
        <span onClick={() => setLogScale(v => !v)} style={{ cursor: 'pointer' }}>
          <HFBadge tone={logScale ? 'indigo' : 'neutral'}>log scale</HFBadge>
        </span>
        <span onClick={() => setOverlay(v => !v)} style={{ cursor: 'pointer' }}>
          <HFBadge tone={overlay ? 'indigo' : 'neutral'}>overlay r11</HFBadge>
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.6fr 1fr', gap: 12, flex: 1, minHeight: 0 }}>
        <HFCard pad={10} style={{ display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0 0 6px' }}>
            <HFText size={13} weight={600}>Score convergence</HFText>
          </div>
          <div style={{ flex: 1, minHeight: 0 }}>
            <HFChart width={Math.max(360, width * 0.55)} height={Math.max(280, height - 80)} lines={[
              { kind: 'raw',  color: HF.textDim, opacity: 0.55, label: `${run.id} · raw`,  seed: 0 },
              { kind: 'conv', color: HF.accent,  width: 2,       label: `${run.id} · best`, seed: 0 },
              ...(overlay ? [{ kind: 'conv', color: '#0ea5e9', dash: true, label: 'r11 · best', seed: 1.4 }] : []),
            ]} xLabel="iteration" yLabel="score F(x)" />
          </div>
        </HFCard>
        <HFCard pad={10} style={{ display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0 0 6px', gap: 8 }}>
            <HFText size={13} weight={600}>Metric</HFText>
            <select value={metric} onChange={(e) => setMetric(e.target.value)}
              style={{ flex: 1, padding: '3px 8px', fontFamily: HF.mono, fontSize: 12, border: `1px solid ${HF.border}`, borderRadius: 6, background: HF.panel, color: HF.text }}>
              {specs.map(sp => <option key={sp.name} value={sp.name}>{sp.name}</option>)}
            </select>
          </div>
          <div style={{ flex: 1, minHeight: 0 }}>
            <HFChart width={Math.max(280, width * 0.35)} height={Math.max(280, height - 80)}
              lines={[{ kind: 'conv', color: specs.find(sp => sp.name === metric)?.pass ? HF.success : HF.error, width: 2, label: `best ${metric}` }]}
              target xLabel="iteration" yLabel={specs.find(sp => sp.name === metric)?.unit || ''} />
          </div>
        </HFCard>
      </div>

      <HFCard pad={8} style={{ background: HF.panelAlt }}>
        <HFMono size={11} color={HF.text}>
          iter {847} · ugf=<span style={{color:HF.success}}>214M</span>  dcgain=<span style={{color:HF.success}}>46.2dB</span>  pm=<span style={{color:HF.error}}>52°</span>  inoise=<span style={{color:HF.success}}>0.98m</span>  idd=<span style={{color:HF.error}}>31µ</span>  tsettle=<span style={{color:HF.success}}>12.4µ</span>  ·  F=<span style={{color:HF.accent, fontWeight:600}}>0.412 ↑</span>
        </HFMono>
      </HFCard>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// YAML setup view (Monaco-like editor + Load Demo + Validate + Apply)
// ─────────────────────────────────────────────────────────────
function StudioYAMLView({ width, height }) {
  const { s, set, demos, testbenches, params } = useStudio();
  const [demoOpen, setDemoOpen] = React.useState(false);

  const yamlLines = [
    ['project:'], ['  name: CASCODE-OTA'], ['  simulator: ngspice'],
    ['  netlist: spice/ota-improved.spice'], ['  ws_root: examples/OTA/cascode/ihp-sg13g2/'], [''],
    ['  tech_spec:'], ['    name: ihp-sg13g2'], ['    constraints:'],
    ['      min_nfet_w: 0.18u', 'm'], ['      max_nfet_w: 200u',  'm'],
    ['      min_nfet_l: 0.18u', 'm'], ['      max_nfet_l: 10u',   'm'], [''],
    ['  dut_params:'], ['    - name: X_DUT_M1M2_W'], ['      min_val: min_nfet_w', 'm'], ['      max_val: max_nfet_w', 'm'],
    ['    - name: X_DUT_M5_NG'], ['      min_val: 1', 'm'], ['      max_val: 5', 'm'], ['      is_integer: true', 'a'], [''],
    ['  optimizer_config:'], [`    type: nevergrad`], [`    name: ${s.algo}`, 'w'], [`    budget: ${s.budget}`], [''],
    ['    target_specs:'], ['      - name: ugf'], ['        testbench: tb_ac', 'm'],
    ['        goal: exceed'], ['        target: 200e6', 'h'], ['        tolerance: 10e6', 'm'],
    ['        weight: 100.0', 'm'], ['        error_type: relative-absolute', 'm'], ['        reward_type: log', 'm'],
  ];
  const colorFor = (kind) => ({ m: HF.textMute, a: HF.accent, h: HF.text, w: HF.warn }[kind] || HF.text);

  return (
    <div style={{ width, height, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Toolbar */}
      <div style={{ height: 44, borderBottom: `1px solid ${HF.border}`, background: HF.panel, padding: '0 14px', display: 'flex', alignItems: 'center', gap: 10 }}>
        <HFMono size={12} color={HF.textMute}>sizing/project_setup.yaml</HFMono>
        {s.yamlDirty && <HFBadge tone="warn">● unsaved</HFBadge>}
        <div style={{ flex: 1 }} />
        <div style={{ position: 'relative' }}>
          <HFButton size="sm" onClick={() => setDemoOpen(v => !v)} icon={<HFIcon name="folder" size={11} />}>
            Load demo  ▾
          </HFButton>
          {demoOpen && (
            <div style={{
              position: 'absolute', top: 32, right: 0, width: 280,
              background: HF.panel, border: `1px solid ${HF.border}`, borderRadius: 7,
              boxShadow: '0 4px 12px rgba(28,25,23,0.10)', zIndex: 10, padding: 4,
            }}>
              {demos.map(d => (
                <div key={d.id} onClick={() => { set({ yamlDirty: false }); setDemoOpen(false); }} style={{
                  padding: '6px 8px', borderRadius: 5, cursor: 'pointer',
                }}
                  onMouseEnter={(e) => e.currentTarget.style.background = HF.accentSoft}
                  onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}>
                  <HFText size={12} weight={500}>{d.label}</HFText>
                  <HFMono size={10} color={HF.textMute} style={{ display: 'block' }}>{d.path}</HFMono>
                </div>
              ))}
            </div>
          )}
        </div>
        <HFButton size="sm" icon={<HFIcon name="file" size={11} />}>Upload netlist…</HFButton>
        <HFButton size="sm" icon={<HFIcon name="check" size={11} />}>Validate</HFButton>
        <HFButton size="sm" kind="primary" onClick={() => set({ yamlDirty: false })}>Apply</HFButton>
      </div>

      {/* Body */}
      <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
        {/* Editor */}
        <div style={{ flex: 1, background: HF.bg, padding: '12px 0', overflow: 'hidden', position: 'relative', minWidth: 0 }}>
          <div style={{ fontFamily: HF.mono, fontSize: 12.5, lineHeight: 1.7, color: HF.text }}>
            {yamlLines.map((l, i) => (
              <div key={i} style={{ display: 'flex', gap: 14, padding: '0 12px' }}>
                <span style={{ color: HF.textDim, width: 24, textAlign: 'right', flexShrink: 0 }}>{i + 1}</span>
                <span style={{ color: colorFor(l[1]), whiteSpace: 'pre' }}>{l[0]}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Right summary */}
        <div style={{ width: 320, borderLeft: `1px solid ${HF.border}`, background: HF.panel, display: 'flex', flexDirection: 'column' }}>
          <div style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 12, overflow: 'hidden' }}>
            <HFSectionLabel action={<HFBadge tone="success"><HFIcon name="check" size={10} />valid</HFBadge>}>Parsed summary</HFSectionLabel>
            <HFCard pad={10}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', rowGap: 4 }}>
                {[
                  ['simulator', 'ngspice'],
                  ['tech', 'ihp-sg13g2'],
                  ['testbenches', '3 enabled'],
                  ['dut params', '13 (1 integer)'],
                  ['target specs', '6'],
                  ['optimizer', `${s.algo} · ${s.budget}`],
                ].map(([k, v]) => (<React.Fragment key={k}>
                  <HFMono size={11} color={HF.textMute}>{k}</HFMono>
                  <HFMono size={11} weight={500} style={{ textAlign: 'right' }}>{v}</HFMono>
                </React.Fragment>))}
              </div>
            </HFCard>

            <HFSectionLabel>Testbenches</HFSectionLabel>
            {testbenches.map(tb => (
              <div key={tb.name} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '5px 8px', border: `1px solid ${HF.border}`, borderRadius: 6 }}>
                <div>
                  <HFMono size={11} weight={500}>{tb.name}</HFMono>
                  <HFMono size={9} color={HF.textMute} style={{ display: 'block' }}>{tb.desc}</HFMono>
                </div>
                <HFBadge tone="success" style={{ fontSize: 9, padding: '0 5px' }}>enabled</HFBadge>
              </div>
            ))}

            <HFSectionLabel action={<HFMono size={10} color={HF.textMute}>{params.length}</HFMono>}>DUT parameters</HFSectionLabel>
            <div style={{ maxHeight: 180, overflow: 'auto', border: `1px solid ${HF.border}`, borderRadius: 6 }}>
              {params.slice(0, 6).map((p, i) => (
                <div key={p.name} style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', padding: '4px 8px', borderBottom: i < 5 ? `1px dashed ${HF.border}` : 'none' }}>
                  <HFMono size={10}>{p.name}</HFMono>
                  <HFMono size={10} color={HF.textMute} style={{ textAlign: 'right' }}>{p.bounds}</HFMono>
                </div>
              ))}
              <div style={{ padding: '4px 8px', background: HF.panelAlt }}>
                <HFMono size={10} color={HF.textMute}>+ {params.length - 6} more rows</HFMono>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Schematic view (click-to-select transistor groups)
// ─────────────────────────────────────────────────────────────
function StudioSchematicView({ width, height }) {
  const { s, set, params } = useStudio();
  const groups = ['M1M2', 'M1CM2C', 'M3M4', 'M3CM4C', 'M5', 'M6'];

  return (
    <div style={{ width, height, display: 'flex', overflow: 'hidden' }}>
      {/* Schematic canvas */}
      <div style={{
        flex: 1, position: 'relative',
        background: `repeating-linear-gradient(0deg, ${HF.bg} 0 23px, ${HF.panelAlt} 23px 24px), repeating-linear-gradient(90deg, ${HF.bg} 0 23px, ${HF.panelAlt} 23px 24px)`,
        display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 32,
      }}>
        <HFCard pad={20} style={{ background: HF.panel }}>
          <HFSchematic width={460} height={320} highlighted={s.selectedDevice} />
        </HFCard>

        <div style={{ position: 'absolute', top: 14, left: 14, display: 'flex', gap: 6 }}>
          <HFBadge tone="neutral">click a device → set bounds</HFBadge>
          <HFBadge tone="neutral">hover → which specs depend</HFBadge>
        </div>

        {/* Quick device picker (overlay) */}
        <div style={{ position: 'absolute', bottom: 16, left: 16, display: 'flex', gap: 6 }}>
          {groups.map(g => (
            <span key={g} onClick={() => set({ selectedDevice: g })} style={{ cursor: 'pointer' }}>
              <HFBadge tone={s.selectedDevice === g ? 'indigo' : 'neutral'}>{g}</HFBadge>
            </span>
          ))}
        </div>
      </div>

      {/* Inspector */}
      <div style={{ width: 320, borderLeft: `1px solid ${HF.border}`, background: HF.panel, padding: 16, display: 'flex', flexDirection: 'column', gap: 12, overflow: 'hidden' }}>
        <div>
          <HFText size={11} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase', letterSpacing: 0.4 }}>Selected device</HFText>
          <HFText size={18} weight={600}>{s.selectedDevice || 'M5'} · tail current</HFText>
          <HFMono size={11} color={HF.textMute}>NMOS · multiplier NG=3</HFMono>
        </div>

        <HFCard pad={10}>
          <HFText size={11} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase' }}>Width  W</HFText>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 }}>
            <HFMono size={10}>0.18µ</HFMono>
            <div style={{ flex: 1, position: 'relative', height: 6, background: HF.panelAlt, borderRadius: 999, overflow: 'visible' }}>
              <div style={{ position: 'absolute', left: '8%', right: '20%', top: 1, height: 4, background: HF.accentMid, borderRadius: 999 }} />
              <div style={{ position: 'absolute', left: '8%', top: -3, width: 10, height: 12, background: HF.accent, borderRadius: 2 }} />
              <div style={{ position: 'absolute', left: '80%', top: -3, width: 10, height: 12, background: HF.accent, borderRadius: 2 }} />
            </div>
            <HFMono size={10}>200µ</HFMono>
          </div>
          <HFMono size={10} color={HF.textMute} style={{ display: 'block', marginTop: 4 }}>best so far: 12.0 µ</HFMono>

          <HFText size={11} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase', marginTop: 8, display: 'block' }}>Length  L</HFText>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 }}>
            <HFMono size={10}>0.18µ</HFMono>
            <div style={{ flex: 1, position: 'relative', height: 6, background: HF.panelAlt, borderRadius: 999 }}>
              <div style={{ position: 'absolute', left: '4%', right: '60%', top: 1, height: 4, background: HF.accentMid, borderRadius: 999 }} />
              <div style={{ position: 'absolute', left: '4%', top: -3, width: 10, height: 12, background: HF.accent, borderRadius: 2 }} />
              <div style={{ position: 'absolute', left: '40%', top: -3, width: 10, height: 12, background: HF.accent, borderRadius: 2 }} />
            </div>
            <HFMono size={10}>10µ</HFMono>
          </div>
          <HFMono size={10} color={HF.textMute} style={{ display: 'block', marginTop: 4 }}>best so far: 0.30 µ</HFMono>

          <div style={{ display: 'flex', gap: 5, marginTop: 8 }}>
            <HFBadge tone="neutral">freeze</HFBadge>
            <HFBadge tone="neutral">log scale</HFBadge>
            <HFBadge tone="indigo">integer NG</HFBadge>
          </div>
        </HFCard>

        <HFSectionLabel>Sensitivity · last run</HFSectionLabel>
        <HFCard pad={10}>
          {[
            ['idd', 0.92, HF.accent],
            ['ugf', 0.55, HF.warn],
            ['inoise', 0.41, HF.warn],
            ['pm', 0.12, HF.textMute],
            ['dcgain', 0.07, HF.textMute],
          ].map(([m, v, c]) => (
            <div key={m} style={{ display: 'grid', gridTemplateColumns: '60px 1fr 36px', alignItems: 'center', gap: 8, padding: '3px 0' }}>
              <HFMono size={11}>{m}</HFMono>
              <div style={{ height: 4, background: HF.panelAlt, borderRadius: 999, overflow: 'hidden' }}>
                <div style={{ width: `${v * 100}%`, height: '100%', background: c }} />
              </div>
              <HFMono size={10} color={HF.textMute}>{(v * 100).toFixed(0)}%</HFMono>
            </div>
          ))}
        </HFCard>

        <HFMono size={10} color={HF.textMute}>
          ↑ sizing this device most strongly trades off <span style={{color:HF.text}}>idd ↔ ugf</span>
        </HFMono>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Score shaping view (sigmoid/linear toggle, slider, breakdown, F(x))
// ─────────────────────────────────────────────────────────────
function StudioShapingView({ width, height }) {
  const { s, set, specs } = useStudio();
  const selSpec = specs.find(sp => sp.name === s.selectedSpec) || specs[0];

  // Try-value slider: 0..1 maps to numeric value around target
  const r = selSpec.range || Math.abs(selSpec.target) * 0.5 || 1;
  const lo = selSpec.target - 1.5 * r;
  const hi = selSpec.target + 1.5 * r;
  const tryValue = s.specTryValue ?? selSpec.target;
  const ratio = Math.max(0, Math.min(1, (tryValue - lo) / (hi - lo)));
  const cur = studioCompPenalty(selSpec, tryValue);

  // Aggregate F(x) using current shape, summing the per-spec penalties
  const Fsig = specs.reduce((a, sp) => a + sp.P_sig * (sp.weight / 100), 0).toFixed(3);
  const Flin = specs.reduce((a, sp) => a + sp.P_lin * (sp.weight / 100), 0).toFixed(3);
  const worst = [...specs].sort((a, b) => (s.specShape === 'sigmoid' ? b.P_sig - a.P_sig : b.P_lin - a.P_lin))[0];

  return (
    <div style={{ width, height, padding: 16, overflow: 'hidden', display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <HFText size={14} weight={600}>Score shaping</HFText>
        <HFText size={12} color={HF.textMute}>Sigmoid vs linear penalty · per spec, summed into F(x)</HFText>
        <div style={{ flex: 1 }} />
        <HFBadge tone={worst.P_sig > worst.P_lin ? 'error' : 'warn'}>
          worst spec ({s.specShape}): {worst.name} · P̂ {(s.specShape === 'sigmoid' ? worst.P_sig : worst.P_lin).toFixed(2)}
        </HFBadge>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 14, flex: 1, minHeight: 0 }}>
        {/* Spec inspector w/ slider + chart */}
        <HFCard pad={14} style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
            <select value={s.selectedSpec} onChange={(e) => set({ selectedSpec: e.target.value, specTryValue: specs.find(sp => sp.name === e.target.value).target })}
              style={{ padding: '4px 10px', fontFamily: HF.mono, fontSize: 13, fontWeight: 600, border: `1px solid ${HF.border}`, borderRadius: 6, background: HF.panel, color: HF.text }}>
              {specs.map(sp => <option key={sp.name} value={sp.name}>{sp.name}</option>)}
            </select>
            <HFText size={12} color={HF.textMute}>{selSpec.desc} · {selSpec.testbench}</HFText>
            <div style={{ flex: 1 }} />
            <HFMono size={11} color={HF.textMute}>w={selSpec.weight} · {selSpec.goal} {selSpec.target_str}</HFMono>
          </div>

          {/* Shape toggle */}
          <div style={{ display: 'flex', gap: 0, background: HF.panelAlt, border: `1px solid ${HF.border}`, borderRadius: 7, padding: 3, width: 'fit-content' }}>
            {['sigmoid', 'linear', 'none'].map(sh => (
              <span key={sh} onClick={() => set({ specShape: sh })} style={{
                padding: '5px 14px', borderRadius: 5, cursor: 'pointer',
                background: s.specShape === sh ? HF.panel : 'transparent',
                color: s.specShape === sh ? HF.accent : HF.textMute,
                fontFamily: HF.ui, fontSize: 12, fontWeight: 600,
                boxShadow: s.specShape === sh ? '0 1px 2px rgba(0,0,0,0.05)' : 'none',
              }}>{sh}</span>
            ))}
          </div>

          {/* Slider */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
              <HFText size={11} color={HF.textMute}>Try value</HFText>
              <HFMono size={13} weight={600}>{studioFormatValue(selSpec, tryValue)}</HFMono>
            </div>
            <input type="range" min={0} max={1000} value={Math.round(ratio * 1000)}
              onChange={(e) => set({ specTryValue: studioSliderValue(selSpec, +e.target.value / 1000) })}
              style={{ width: '100%', accentColor: HF.accent }} />
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <HFMono size={9} color={HF.textDim}>{studioFormatValue(selSpec, lo)}</HFMono>
              <HFMono size={9} color={HF.warn}>target {selSpec.target_str}</HFMono>
              <HFMono size={9} color={HF.textDim}>{studioFormatValue(selSpec, hi)}</HFMono>
            </div>
          </div>

          {/* Live penalty values */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <div style={{ padding: 10, border: `1px solid ${HF.border}`, borderRadius: 6 }}>
              <HFText size={11} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase' }}>Sigmoid penalty</HFText>
              <HFText size={22} weight={600} color={HF.accent}>{cur.sigmoid.toFixed(2)}</HFText>
            </div>
            <div style={{ padding: 10, border: `1px solid ${HF.border}`, borderRadius: 6 }}>
              <HFText size={11} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase' }}>Linear penalty</HFText>
              <HFText size={22} weight={600} color="#0ea5e9">{cur.linear.toFixed(2)}</HFText>
            </div>
          </div>

          {/* Penalty curve */}
          <div style={{ flex: 1, minHeight: 0 }}>
            <StudioPenaltyCurve width={Math.max(420, width * 0.48)} height={Math.max(170, height - 360)} spec={selSpec} tryValue={tryValue} />
          </div>
        </HFCard>

        {/* Breakdown table + aggregate */}
        <HFCard pad={14} style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <HFSectionLabel>Per-spec breakdown</HFSectionLabel>
          <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr 0.8fr 0.8fr', padding: '4px 6px', borderBottom: `1px solid ${HF.border}` }}>
            {['spec', 'value', 'P̂ sig', 'P̂ lin'].map(h =>
              <HFMono key={h} size={10} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase' }}>{h}</HFMono>
            )}
          </div>
          {specs.map((sp) => {
            const isSel = sp.name === s.selectedSpec;
            return (
              <div key={sp.name}
                onClick={() => set({ selectedSpec: sp.name, specTryValue: sp.target })}
                style={{
                  display: 'grid', gridTemplateColumns: '1.2fr 0.8fr 0.8fr 0.8fr',
                  padding: '5px 6px', borderRadius: 5, cursor: 'pointer',
                  background: isSel ? HF.accentSoft : 'transparent', alignItems: 'center',
                }}>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                  <span style={{ width: 6, height: 6, borderRadius: 999, background: sp.pass ? HF.success : HF.error }} />
                  <HFMono size={11} weight={isSel ? 600 : 500}>{sp.name}</HFMono>
                </span>
                <HFMono size={11} color={HF.text}>{sp.value}</HFMono>
                <HFMono size={11} color={sp.P_sig > 0.3 ? HF.error : HF.text}>{sp.P_sig.toFixed(2)}</HFMono>
                <HFMono size={11} color={sp.P_lin > 0.3 ? HF.error : HF.text}>{sp.P_lin.toFixed(2)}</HFMono>
              </div>
            );
          })}

          <div style={{ height: 1, background: HF.border, margin: '6px 0' }} />

          <HFSectionLabel>Aggregate F(x)</HFSectionLabel>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <div style={{ padding: 10, border: `1px solid ${s.specShape === 'sigmoid' ? HF.accent : HF.border}`, borderRadius: 6, background: s.specShape === 'sigmoid' ? HF.accentSoft : HF.panel }}>
              <HFText size={11} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase' }}>Sigmoid · F(x)</HFText>
              <HFText size={22} weight={600} color={HF.accent}>{Fsig}</HFText>
              <HFMono size={10} color={HF.textMute}>worst spec: {worst.name}</HFMono>
            </div>
            <div style={{ padding: 10, border: `1px solid ${s.specShape === 'linear' ? '#0ea5e9' : HF.border}`, borderRadius: 6, background: s.specShape === 'linear' ? '#e0f2fe' : HF.panel }}>
              <HFText size={11} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase' }}>Linear · F(x)</HFText>
              <HFText size={22} weight={600} color="#0ea5e9">{Flin}</HFText>
              <HFMono size={10} color={HF.textMute}>spread compressed</HFMono>
            </div>
          </div>
          <HFText size={11} color={HF.textMute} style={{ lineHeight: 1.4 }}>
            Sigmoid concentrates penalty on near-target misses, so the optimizer keeps caring about <HFMono color={HF.text}>{worst.name}</HFMono> even after others pass. Linear lets the worst spec mask the rest.
          </HFText>
        </HFCard>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Compare view (Explorer-equivalent: Run A/B selectors, overlay, scatter, envelope, histogram, best params, spec summary)
// ─────────────────────────────────────────────────────────────
function StudioCompareView({ width, height }) {
  const { s, set, runs, specs } = useStudio();

  const runOpts = runs.filter(r => r.status !== 'live' || r.id === s.compareRunA || r.id === s.compareRunB);

  return (
    <div style={{ width, height, padding: 16, overflow: 'auto', display: 'flex', flexDirection: 'column', gap: 12 }}>
      {/* Run A / B selectors */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <HFText size={14} weight={600}>Compare runs</HFText>
        <div style={{ flex: 1 }} />
        <HFText size={11} color={HF.textMute}>Run A</HFText>
        <select value={s.compareRunA} onChange={(e) => set({ compareRunA: e.target.value })}
          style={{ padding: '4px 8px', fontFamily: HF.mono, fontSize: 12, border: `1px solid ${HF.border}`, borderRadius: 6, background: HF.panel, color: HF.accent, fontWeight: 600 }}>
          {runs.map(r => <option key={r.id} value={r.id}>{r.id.toUpperCase()} · {r.name}</option>)}
        </select>
        <HFText size={11} color={HF.textMute}>vs Run B</HFText>
        <select value={s.compareRunB} onChange={(e) => set({ compareRunB: e.target.value })}
          style={{ padding: '4px 8px', fontFamily: HF.mono, fontSize: 12, border: `1px solid ${HF.border}`, borderRadius: 6, background: HF.panel, color: '#0ea5e9', fontWeight: 600 }}>
          {runs.map(r => <option key={r.id} value={r.id}>{r.id.toUpperCase()} · {r.name}</option>)}
        </select>
      </div>

      {/* Row 1: Score convergence overlay + metric overlay */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 12 }}>
        <HFCard pad={10}>
          <HFText size={13} weight={600}>Score convergence · overlay</HFText>
          <HFChart width={Math.max(420, width * 0.5)} height={180}
            lines={[
              { kind: 'conv', color: HF.accent, width: 2, label: `${s.compareRunA} · best`, seed: 0 },
              { kind: 'conv', color: '#0ea5e9', width: 2, label: `${s.compareRunB} · best`, seed: 1.4, dash: true },
            ]} xLabel="iteration" yLabel="F(x)" />
        </HFCard>
        <HFCard pad={10}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
            <HFText size={13} weight={600}>Metric overlay</HFText>
            <select defaultValue="ugf" style={{ padding: '3px 8px', fontFamily: HF.mono, fontSize: 12, border: `1px solid ${HF.border}`, borderRadius: 6, background: HF.panel, color: HF.text }}>
              {specs.map(sp => <option key={sp.name} value={sp.name}>{sp.name}</option>)}
            </select>
          </div>
          <HFChart width={Math.max(280, width * 0.35)} height={170}
            lines={[
              { kind: 'conv', color: HF.accent, width: 2, label: s.compareRunA, seed: 0.2 },
              { kind: 'conv', color: '#0ea5e9', width: 2, label: s.compareRunB, seed: 1.2, dash: true },
            ]} target xLabel="iteration" yLabel="ugf" />
        </HFCard>
      </div>

      {/* Row 2: Scatter + Envelope */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 12 }}>
        <HFCard pad={10}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6, gap: 8 }}>
            <HFText size={13} weight={600}>Metric scatter</HFText>
            <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              <HFText size={11} color={HF.textMute}>X</HFText>
              <select value={s.scatterX} onChange={(e) => set({ scatterX: e.target.value })}
                style={{ padding: '3px 8px', fontFamily: HF.mono, fontSize: 12, border: `1px solid ${HF.border}`, borderRadius: 6, background: HF.panel, color: HF.text }}>
                {specs.map(sp => <option key={sp.name} value={sp.name}>{sp.name}</option>)}
              </select>
              <HFText size={11} color={HF.textMute}>Y</HFText>
              <select value={s.scatterY} onChange={(e) => set({ scatterY: e.target.value })}
                style={{ padding: '3px 8px', fontFamily: HF.mono, fontSize: 12, border: `1px solid ${HF.border}`, borderRadius: 6, background: HF.panel, color: HF.text }}>
                {specs.map(sp => <option key={sp.name} value={sp.name}>{sp.name}</option>)}
              </select>
            </div>
          </div>
          {/* Scatter SVG */}
          <StudioScatter width={Math.max(420, width * 0.5)} height={220} xLabel={s.scatterX} yLabel={s.scatterY} />
        </HFCard>
        <HFCard pad={10}>
          <HFText size={13} weight={600}>Performance envelope</HFText>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', padding: '6px 6px', borderBottom: `1px solid ${HF.border}` }}>
            <HFMono size={10} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase' }}>spec</HFMono>
            <HFMono size={10} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase' }}>{s.compareRunA} best</HFMono>
            <HFMono size={10} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase' }}>{s.compareRunB} best</HFMono>
          </div>
          {specs.map((sp, i) => (
            <div key={sp.name} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', padding: '5px 6px', borderBottom: i < specs.length - 1 ? `1px dashed ${HF.border}` : 'none', alignItems: 'center' }}>
              <HFMono size={11}>{sp.name}</HFMono>
              <HFMono size={11} color={HF.accent} weight={500}>{sp.value}</HFMono>
              <HFMono size={11} color="#0ea5e9" weight={500}>{i % 2 === 0 ? sp.value : '—'}</HFMono>
            </div>
          ))}
        </HFCard>
      </div>

      {/* Row 3: Histogram + Best params */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 12 }}>
        <HFCard pad={10}>
          <HFText size={13} weight={600}>Metric histogram</HFText>
          <StudioHistogram width={Math.max(420, width * 0.5)} height={170} />
        </HFCard>
        <HFCard pad={10}>
          <HFText size={13} weight={600}>Best design params</HFText>
          <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr 1fr', padding: '4px 0', borderBottom: `1px solid ${HF.border}` }}>
            {['param', s.compareRunA, s.compareRunB].map(h => <HFMono key={h} size={10} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase' }}>{h}</HFMono>)}
          </div>
          {[
            ['M1M2 W',    '24.2µ',  '19.8µ'],
            ['M1M2 L',    '0.42µ',  '0.55µ'],
            ['M3M4 W',    '46.8µ',  '52.4µ'],
            ['M5 W',      '12.0µ',  '8.4µ'],
            ['M5 NG',     '3',      '2'],
            ['V_BIAS_1',  '0.62 V', '0.58 V'],
          ].map(([k, a, b]) => (
            <div key={k} style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr 1fr', padding: '4px 0', borderBottom: `1px dashed ${HF.border}` }}>
              <HFMono size={11}>{k}</HFMono>
              <HFMono size={11} color={HF.accent}>{a}</HFMono>
              <HFMono size={11} color="#0ea5e9">{b}</HFMono>
            </div>
          ))}
        </HFCard>
      </div>

      {/* Row 4: Spec summary */}
      <HFCard pad={10}>
        <HFText size={13} weight={600}>Spec summary</HFText>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 0.8fr 1fr 1fr 0.6fr 0.6fr', padding: '6px 4px', borderBottom: `1px solid ${HF.border}` }}>
          {['spec', 'goal', 'target', `${s.compareRunA} best`, `${s.compareRunB} best`, `${s.compareRunA}`, `${s.compareRunB}`].map(h =>
            <HFMono key={h} size={10} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase' }}>{h}</HFMono>
          )}
        </div>
        {specs.map((sp, i) => (
          <div key={sp.name} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 0.8fr 1fr 1fr 0.6fr 0.6fr', padding: '5px 4px', borderBottom: i < specs.length - 1 ? `1px dashed ${HF.border}` : 'none', alignItems: 'center' }}>
            <HFMono size={11} weight={500}>{sp.name}</HFMono>
            <HFMono size={11} color={HF.textMute}>{sp.goal}</HFMono>
            <HFMono size={11} color={HF.textMute}>{sp.target_str}</HFMono>
            <HFMono size={11}>{sp.value}</HFMono>
            <HFMono size={11}>{i % 2 === 0 ? sp.value : '—'}</HFMono>
            <HFBadge tone={sp.pass ? 'success' : 'error'} style={{ fontSize: 9, padding: '0 5px' }}>{sp.pass ? 'PASS' : 'FAIL'}</HFBadge>
            <HFBadge tone={i % 2 === 0 ? 'success' : 'warn'} style={{ fontSize: 9, padding: '0 5px' }}>{i % 2 === 0 ? 'PASS' : '—'}</HFBadge>
          </div>
        ))}
      </HFCard>
    </div>
  );
}

// Scatter chart (for compare view)
function StudioScatter({ width, height, xLabel, yLabel }) {
  const padL = 42, padR = 12, padT = 10, padB = 28;
  const w = width - padL - padR;
  const h = height - padT - padB;
  // Deterministic pts based on labels
  const seed = (xLabel.length * 13 + yLabel.length * 7) % 100;
  const pts = [];
  for (let i = 0; i < 120; i++) {
    const v = (Math.sin(i * 1.9 + seed) * 0.5 + 0.5);
    const u = (Math.cos(i * 1.3 + seed * 0.7) * 0.5 + 0.5);
    const x = u * w;
    const y = (1 - v) * h;
    const feasible = u > 0.55 && v > 0.55;
    pts.push([x, y, feasible, i % 2 === 0]);
  }
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: 'block' }}>
      {[0.25, 0.5, 0.75, 1].map((t, i) => (
        <line key={i} x1={padL} y1={padT + h * (1 - t)} x2={padL + w} y2={padT + h * (1 - t)} stroke={HF.hairline} strokeWidth={1} />
      ))}
      <line x1={padL} y1={padT + h} x2={padL + w} y2={padT + h} stroke={HF.borderDk} />
      <line x1={padL} y1={padT} x2={padL} y2={padT + h} stroke={HF.borderDk} />
      <line x1={padL + w * 0.55} y1={padT} x2={padL + w * 0.55} y2={padT + h} stroke={HF.warn} strokeDasharray="3 3" />
      <line x1={padL} y1={padT + h * 0.45} x2={padL + w} y2={padT + h * 0.45} stroke={HF.warn} strokeDasharray="3 3" />
      {pts.map(([x, y, f, runA], i) => (
        <circle key={i} cx={padL + x} cy={padT + y} r={2.6}
          fill={runA ? HF.accent : '#0ea5e9'} opacity={f ? 0.85 : 0.3} />
      ))}
      <text x={padL + w / 2} y={padT + h + 16} fill={HF.textMute} fontSize={11} textAnchor="middle">{xLabel}</text>
      <text x={12} y={padT + h / 2} fill={HF.textMute} fontSize={11} textAnchor="middle"
        transform={`rotate(-90, 12, ${padT + h / 2})`}>{yLabel}</text>
    </svg>
  );
}

// Histogram
function StudioHistogram({ width, height }) {
  const padL = 36, padR = 12, padT = 10, padB = 28;
  const w = width - padL - padR;
  const h = height - padT - padB;
  const bins = 14;
  const bw = w / bins;
  const aBars = Array.from({ length: bins }, (_, i) => Math.max(2, Math.sin(i * 0.55) * 30 + 35 - i * 0.8));
  const bBars = Array.from({ length: bins }, (_, i) => Math.max(2, Math.cos(i * 0.45 + 1) * 22 + 28 + i * 0.4));
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <line x1={padL} y1={padT + h} x2={padL + w} y2={padT + h} stroke={HF.borderDk} />
      <line x1={padL} y1={padT} x2={padL} y2={padT + h} stroke={HF.borderDk} />
      <line x1={padL + w * 0.7} y1={padT} x2={padL + w * 0.7} y2={padT + h} stroke={HF.warn} strokeDasharray="3 3" />
      {aBars.map((v, i) => (
        <rect key={'a' + i} x={padL + i * bw + 1} y={padT + h - v} width={bw - 2} height={v}
          fill={HF.accent} opacity={0.55} />
      ))}
      {bBars.map((v, i) => (
        <rect key={'b' + i} x={padL + i * bw + 2} y={padT + h - v} width={bw - 4} height={v}
          fill="#0ea5e9" opacity={0.45} />
      ))}
      <text x={padL + w / 2} y={padT + h + 16} fill={HF.textMute} fontSize={11} textAnchor="middle">value</text>
      <text x={12} y={padT + h / 2} fill={HF.textMute} fontSize={11} textAnchor="middle"
        transform={`rotate(-90, 12, ${padT + h / 2})`}>count</text>
    </svg>
  );
}

Object.assign(window, {
  StudioPipelineView, StudioConvergenceView, StudioYAMLView,
  StudioSchematicView, StudioShapingView, StudioCompareView,
  StudioPenaltyCurve, studioCompPenalty, studioFormatValue,
});
