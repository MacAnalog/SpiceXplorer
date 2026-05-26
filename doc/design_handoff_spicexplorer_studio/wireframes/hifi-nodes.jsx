// Hi-fi D — Pipeline / node graph.

function HFNodes() {
  const W = 1440, H = 900;

  // Reusable Node component
  const Node = ({ x, y, w = 220, title, kind, status, badge, body, accent = false, selected = false }) => (
    <div style={{
      position: 'absolute', left: x, top: y, width: w,
      background: HF.panel,
      border: `1px solid ${selected ? HF.accent : accent ? HF.accentMid : HF.border}`,
      borderRadius: 8,
      boxShadow: selected ? `0 0 0 3px ${HF.accentSoft}, 0 4px 8px rgba(28,25,23,0.06)` : '0 1px 3px rgba(28,25,23,0.06)',
      overflow: 'hidden',
    }}>
      <div style={{
        padding: '8px 10px',
        background: accent ? HF.accentSoft : HF.panelAlt,
        borderBottom: `1px solid ${HF.border}`,
        display: 'flex', alignItems: 'center', gap: 8,
      }}>
        <HFMono size={10} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase' }}>{kind}</HFMono>
        <div style={{ flex: 1 }} />
        {status === 'live' && <HFBadge tone="indigo" style={{ fontSize: 9, padding: '0 5px' }}>● live</HFBadge>}
        {status === 'ok'   && <HFBadge tone="success" style={{ fontSize: 9, padding: '0 5px' }}>✓</HFBadge>}
        {status === 'idle' && <HFBadge tone="neutral" style={{ fontSize: 9, padding: '0 5px' }}>idle</HFBadge>}
      </div>
      <div style={{ padding: '8px 10px 10px', display: 'flex', flexDirection: 'column', gap: 6 }}>
        <HFText size={13} weight={600}>{title}</HFText>
        {badge}
        {body}
      </div>
      {/* I/O ports (visual) */}
      <span style={{ position: 'absolute', left: -5, top: '50%', width: 8, height: 8, borderRadius: 999, background: HF.panel, border: `1.5px solid ${HF.borderDk}` }} />
      <span style={{ position: 'absolute', right: -5, top: '50%', width: 8, height: 8, borderRadius: 999, background: HF.panel, border: `1.5px solid ${HF.borderDk}` }} />
    </div>
  );

  const wire = (x1, y1, x2, y2, color = HF.borderDk, w = 1.4, dashed = false, opacity = 1) => {
    const dx = Math.max(40, Math.abs(x2 - x1) * 0.5);
    return (
      <path
        d={`M ${x1} ${y1} C ${x1 + dx} ${y1} ${x2 - dx} ${y2} ${x2} ${y2}`}
        fill="none" stroke={color} strokeWidth={w} strokeOpacity={opacity}
        strokeDasharray={dashed ? '5 4' : 'none'}
        markerEnd={`url(#arr-${color.replace('#', '')})`}
      />
    );
  };

  return (
    <div style={{ width: W, height: H, background: HF.bg, fontFamily: HF.ui, color: HF.text, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Top bar */}
      <div style={{
        height: 56, borderBottom: `1px solid ${HF.border}`, background: HF.panel,
        display: 'flex', alignItems: 'center', padding: '0 20px', gap: 14,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 24, height: 24, borderRadius: 6, background: `linear-gradient(135deg, ${HF.accent}, ${HF.accentHov})`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <HFIcon name="graph" size={14} color="#fff" />
          </div>
          <HFText size={15} weight={600}>SpiceXplorer</HFText>
        </div>
        <HFText size={13} color={HF.textDim}>/</HFText>
        <HFText size={13} weight={500}>OTA Cascode</HFText>
        <HFText size={13} color={HF.textDim}>/</HFText>
        <HFText size={13}>Pipeline</HFText>
        <HFBadge tone="neutral">cascode_ota.graph</HFBadge>
        <div style={{ flex: 1 }} />
        <div style={{ display: 'flex', gap: 6, padding: 4, background: HF.panelAlt, borderRadius: 8, border: `1px solid ${HF.border}` }}>
          {[['Graph', true], ['YAML', false], ['Schematic', false]].map(([l, sel]) => (
            <span key={l} style={{
              padding: '4px 10px', borderRadius: 5,
              background: sel ? HF.panel : 'transparent',
              border: sel ? `1px solid ${HF.border}` : '1px solid transparent',
              fontFamily: HF.ui, fontSize: 12, fontWeight: sel ? 600 : 500,
              color: sel ? HF.text : HF.textMute,
              boxShadow: sel ? '0 1px 2px rgba(0,0,0,0.04)' : 'none',
            }}>{l}</span>
          ))}
        </div>
        <HFButton icon={<HFIcon name="fork" size={13} />}>Fork branch</HFButton>
        <HFButton kind="primary" icon={<HFIcon name="play" size={11} color="#fff" />}>Run pipeline</HFButton>
      </div>

      <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
        {/* Left palette */}
        <div style={{
          width: 230, borderRight: `1px solid ${HF.border}`, background: HF.panel,
          display: 'flex', flexDirection: 'column', minHeight: 0,
        }}>
          <div style={{ padding: '14px 14px 8px' }}>
            <HFSectionLabel>Add node</HFSectionLabel>
            <div style={{ marginTop: 8, padding: '5px 8px', border: `1px solid ${HF.border}`, borderRadius: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
              <HFIcon name="search" size={12} color={HF.textDim} />
              <HFText size={12} color={HF.textDim}>Filter…</HFText>
            </div>
          </div>
          <div style={{ overflow: 'hidden', padding: '0 14px 14px', display: 'flex', flexDirection: 'column', gap: 12 }}>
            {[
              ['Inputs',     [['file', 'Netlist file'], ['chip', 'PDK rules'], ['flag', 'PVT corner']]],
              ['Variables',  [['sliders', 'DUT params'], ['sliders', 'Sweep set']]],
              ['Simulation', [['activity', 'Testbench · AC'], ['activity', 'Testbench · Noise'], ['activity', 'Testbench · Tran']]],
              ['Scoring',    [['dot', 'Spec · sigmoid'], ['dot', 'Spec · linear'], ['dot', 'Aggregate F(x)']]],
              ['Optimizers', [['fork', 'LhsDE'], ['fork', 'CMA-ES'], ['fork', 'LBFGSB'], ['fork', 'Blind LHS']]],
              ['Outputs',    [['file', 'Checkpoint'], ['file', 'Pareto / scatter'], ['file', 'Best YAML']]],
            ].map(([g, items]) => (
              <div key={g}>
                <HFText size={10} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase', letterSpacing: 0.4 }}>{g}</HFText>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 6 }}>
                  {items.map(([ic, n]) => (
                    <div key={n} style={{
                      display: 'flex', alignItems: 'center', gap: 7,
                      padding: '5px 8px', border: `1px solid ${HF.border}`, borderRadius: 6,
                      background: HF.panel,
                      cursor: 'grab',
                    }}>
                      <HFIcon name={ic} size={13} color={HF.textMute} />
                      <HFText size={12}>{n}</HFText>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Canvas */}
        <div style={{
          flex: 1, position: 'relative', overflow: 'hidden',
          backgroundImage: `radial-gradient(${HF.borderDk} 0.7px, transparent 0.7px)`,
          backgroundSize: '18px 18px', backgroundPosition: '0 0',
          backgroundColor: HF.bg,
        }}>
          {/* zoom controls */}
          <div style={{ position: 'absolute', top: 12, left: 12, display: 'flex', flexDirection: 'column', gap: 4 }}>
            {['+', '−', '⊙'].map((c) => (
              <div key={c} style={{
                width: 28, height: 28, background: HF.panel,
                border: `1px solid ${HF.border}`, borderRadius: 6,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontFamily: HF.ui, fontSize: 14, color: HF.text,
                boxShadow: '0 1px 2px rgba(0,0,0,0.04)',
              }}>{c}</div>
            ))}
          </div>
          <div style={{ position: 'absolute', top: 12, right: 12, display: 'flex', gap: 6 }}>
            <HFBadge tone="neutral">⌗ snap 20</HFBadge>
            <HFBadge tone="neutral">⚡ auto-layout</HFBadge>
          </div>

          {/* Wires */}
          <svg width="100%" height="100%" style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
            <defs>
              <marker id={`arr-${HF.borderDk.replace('#', '')}`} markerWidth="10" markerHeight="10" refX="6" refY="5" orient="auto" markerUnits="strokeWidth">
                <path d="M0,1 L7,5 L0,9 Z" fill={HF.borderDk} />
              </marker>
              <marker id={`arr-${HF.accent.replace('#', '')}`} markerWidth="10" markerHeight="10" refX="6" refY="5" orient="auto" markerUnits="strokeWidth">
                <path d="M0,1 L7,5 L0,9 Z" fill={HF.accent} />
              </marker>
            </defs>

            {/* fanout: inputs -> dut params */}
            {wire(265, 130, 305, 270)}
            {wire(265, 240, 305, 280)}
            {wire(265, 350, 305, 290)}
            {/* dut params -> testbenches */}
            {wire(525, 280, 568, 130)}
            {wire(525, 285, 568, 270)}
            {wire(525, 290, 568, 410)}
            {/* testbenches -> spec nodes */}
            {wire(788, 130, 832, 130, HF.accent, 1.8)}
            {wire(788, 270, 832, 240)}
            {wire(788, 270, 832, 370)}
            {wire(788, 410, 832, 500)}
            {/* specs -> aggregate */}
            {wire(1052, 130, 1095, 285)}
            {wire(1052, 240, 1095, 290)}
            {wire(1052, 370, 1095, 295)}
            {wire(1052, 500, 1095, 300)}
            {/* aggregate -> optimizer */}
            {wire(1315, 285, 1100, 285, HF.borderDk, 0, false, 0)} {/* placeholder, drawn below */}
            <path d={`M 1315 285 C 1340 285 1340 600 1100 600`} fill="none" stroke={HF.accent} strokeWidth={1.8} strokeDasharray="5 4" />
            {/* feedback loop annotation */}
            <path d={`M 1100 600 C 600 620 600 380 305 380`} fill="none" stroke={HF.accent} strokeWidth={1.6} strokeDasharray="5 4" opacity={0.8}
              markerEnd={`url(#arr-${HF.accent.replace('#', '')})`} />
          </svg>

          {/* Nodes */}
          <Node x={75} y={100} w={190} kind="input" title="Netlist" status="ok"
            body={<HFMono size={10} color={HF.textMute}>spice/ota-improved.spice<br />6 transistors · 3 .param</HFMono>} />
          <Node x={75} y={210} w={190} kind="input" title="PDK rules · ihp-sg13g2" status="ok"
            body={<HFMono size={10} color={HF.textMute}>min/max W,L per type<br />8 named constraints</HFMono>} />
          <Node x={75} y={320} w={190} kind="input" title="PVT · 1 corner" status="ok"
            body={<HFMono size={10} color={HF.textMute}>tt · 25°C · 1.5 V</HFMono>} />

          <Node x={305} y={240} w={220} kind="variables" title="DUT parameters" status="ok"
            badge={<HFBadge tone="neutral">13 vars · 1 integer</HFBadge>}
            body={<div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
              {['M1M2 W,L', 'M3M4 W,L', 'M1cM2c', 'M3cM4c', 'M5 W,L,NG', 'M6 W,L', 'V_BIAS_1,2'].map((p) => (
                <HFBadge key={p} tone="neutral" style={{ fontSize: 9, padding: '0 5px' }}>{p}</HFBadge>
              ))}
            </div>} />

          <Node x={568} y={100} w={220} kind="simulation" title="Testbench · AC" status="ok"
            badge={<HFBadge tone="neutral">tb_ac · CL=50f</HFBadge>}
            body={<HFMono size={10} color={HF.textMute}>→ ugf · dcgain · pm<br />typ 1.4 s · parallel</HFMono>} />
          <Node x={568} y={240} w={220} kind="simulation" title="Testbench · Noise + DC" status="ok"
            badge={<HFBadge tone="neutral">tb_noise · 1M–1G</HFBadge>}
            body={<HFMono size={10} color={HF.textMute}>→ inoise · idd</HFMono>} />
          <Node x={568} y={380} w={220} kind="simulation" title="Testbench · Transient" status="ok"
            badge={<HFBadge tone="neutral">tb_tran · step</HFBadge>}
            body={<HFMono size={10} color={HF.textMute}>→ tsettle</HFMono>} />

          <Node x={832} y={100} w={220} kind="spec · sigmoid" title="ugf" accent status="ok" selected
            badge={<HFBadge tone="indigo">≥ 200 MHz · w=100</HFBadge>}
            body={<HFMono size={10} color={HF.textMute}>tol ±10M · reward log</HFMono>} />
          <Node x={832} y={210} w={220} kind="spec · sigmoid" title="dcgain" status="ok"
            badge={<HFBadge tone="neutral">≥ 40 dB · w=100</HFBadge>}
            body={<HFMono size={10} color={HF.textMute}>tol ±1 dB</HFMono>} />
          <Node x={832} y={340} w={220} kind="spec · sigmoid" title="pm" status="ok"
            badge={<HFBadge tone="error">= 60° ± 10° · failing</HFBadge>}
            body={<HFMono size={10} color={HF.textMute}>current 52° · reward none</HFMono>} />
          <Node x={832} y={470} w={220} kind="spec bundle" title="inoise · idd · tsettle" status="ok"
            badge={<HFBadge tone="neutral">3 specs · minimize</HFBadge>}
            body={<HFMono size={10} color={HF.textMute}>idd failing · noise/settle ok</HFMono>} />

          <Node x={1095} y={250} w={220} kind="scoring" title="Aggregate F(x)" status="ok"
            badge={<HFBadge tone="neutral">weighted log sum</HFBadge>}
            body={<HFMono size={10} color={HF.textMute}>6 specs in · scalar out<br />current best 0.412</HFMono>} />

          <Node x={1095} y={550} w={220} kind="optimizer" title="LhsDE · seed 48" accent status="live"
            badge={<HFBadge tone="indigo">iter 847 / 2 000</HFBadge>}
            body={
              <div>
                <div style={{ height: 4, background: HF.panelAlt, borderRadius: 999, overflow: 'hidden', marginTop: 2 }}>
                  <div style={{ width: '42%', height: '100%', background: HF.accent }} />
                </div>
                <HFMono size={10} color={HF.textMute} style={{ marginTop: 4 }}>ETA 14 min</HFMono>
              </div>
            } />

          {/* feedback loop label */}
          <div style={{ position: 'absolute', left: 650, top: 595 }}>
            <HFBadge tone="indigo" style={{ background: HF.panel, padding: '2px 7px' }}>
              ↺ optimizer rewrites DUT params each iteration
            </HFBadge>
          </div>

          {/* Inspector panel — selected node = ugf */}
          <div style={{
            position: 'absolute', right: 16, bottom: 16, width: 320,
            background: HF.panel, border: `1px solid ${HF.accentMid}`,
            borderRadius: 10, boxShadow: '0 6px 20px rgba(28,25,23,0.10)',
            padding: 14, display: 'flex', flexDirection: 'column', gap: 10,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <HFText size={10} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase', letterSpacing: 0.4 }}>Spec node</HFText>
                <div><HFText size={14} weight={600}>ugf  ·  unity gain freq</HFText></div>
              </div>
              <HFIcon name="x" size={14} color={HF.textMute} />
            </div>

            <div style={{ display: 'flex', gap: 6 }}>
              <HFBadge tone="indigo">sigmoid</HFBadge>
              <HFBadge tone="neutral">linear</HFBadge>
              <HFBadge tone="neutral">none</HFBadge>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <div>
                <HFText size={10} color={HF.textMute}>goal</HFText>
                <div><HFMono size={12} weight={500}>exceed 200 MHz</HFMono></div>
              </div>
              <div>
                <HFText size={10} color={HF.textMute}>tolerance</HFText>
                <div><HFMono size={12} weight={500}>± 10 MHz</HFMono></div>
              </div>
              <div>
                <HFText size={10} color={HF.textMute}>weight</HFText>
                <div><HFMono size={12} weight={500}>100</HFMono></div>
              </div>
              <div>
                <HFText size={10} color={HF.textMute}>reward</HFText>
                <div><HFMono size={12} weight={500}>log</HFMono></div>
              </div>
            </div>

            <HFPenalty width={292} height={90} />
            <HFText size={11} color={HF.textMute}>
              Drag the dashed midline to retarget · the upstream <HFMono color={HF.text}>tb_ac</HFMono> port lights up while editing.
            </HFText>
          </div>
        </div>

        {/* Right panel: outline + minimap */}
        <div style={{
          width: 260, borderLeft: `1px solid ${HF.border}`, background: HF.panel,
          padding: 16, display: 'flex', flexDirection: 'column', gap: 14, minHeight: 0,
        }}>
          <HFSectionLabel>Graph outline</HFSectionLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {[
              { ind: 0, l: '◆ Inputs', n: '3' },
              { ind: 1, l: 'Netlist', n: '1' },
              { ind: 1, l: 'PDK', n: '1' },
              { ind: 1, l: 'PVT', n: '1' },
              { ind: 0, l: '⬢ Variables', n: '1' },
              { ind: 0, l: '◐ Simulation', n: '3' },
              { ind: 0, l: '● Scoring', n: '4' },
              { ind: 1, l: 'ugf', sel: true },
              { ind: 1, l: 'dcgain' },
              { ind: 1, l: 'pm' },
              { ind: 1, l: 'noise · idd · settle' },
              { ind: 0, l: '⚙ Optimizer', n: '1' },
            ].map((r, i) => (
              <div key={i} style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                padding: '3px 6px', paddingLeft: 6 + r.ind * 12, borderRadius: 4,
                background: r.sel ? HF.accentSoft : 'transparent',
              }}>
                <HFText size={12} color={r.sel ? HF.accent : HF.text} weight={r.sel ? 600 : (r.ind === 0 ? 500 : 400)}>{r.l}</HFText>
                {r.n && <HFMono size={10} color={HF.textMute}>{r.n}</HFMono>}
              </div>
            ))}
          </div>

          <div style={{ height: 1, background: HF.border, margin: '4px -16px' }} />

          <HFSectionLabel>Branches</HFSectionLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {[
              { name: 'main · sigmoid', sel: true, run: 'r12 live' },
              { name: 'compare · linear loss', sel: false, run: 'r11 done' },
              { name: 'wider · M5 NG ∈ 1..7', sel: false, run: '—' },
            ].map((b) => (
              <div key={b.name} style={{
                padding: '6px 8px',
                border: `1px solid ${b.sel ? HF.accentMid : HF.border}`,
                background: b.sel ? HF.accentSoft : HF.panel,
                borderRadius: 6,
                display: 'flex', flexDirection: 'column', gap: 2,
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <HFText size={12} weight={b.sel ? 600 : 500}>{b.name}</HFText>
                  {b.sel && <HFIcon name="check" size={12} color={HF.accent} />}
                </div>
                <HFMono size={10} color={HF.textMute}>{b.run}</HFMono>
              </div>
            ))}
          </div>

          <div style={{ flex: 1 }} />

          <HFSectionLabel>Minimap</HFSectionLabel>
          <div style={{
            height: 110, background: HF.bg, border: `1px solid ${HF.border}`, borderRadius: 6,
            position: 'relative', overflow: 'hidden',
          }}>
            {[
              [10, 20, 16], [10, 40, 16], [10, 60, 16],
              [38, 40, 18], [70, 18, 18], [70, 40, 18], [70, 62, 18],
              [102, 18, 18, true], [102, 40, 18], [102, 62, 18], [102, 84, 18],
              [134, 40, 18], [134, 90, 14],
            ].map(([x, y, w, hi], i) => (
              <div key={i} style={{
                position: 'absolute', left: x, top: y, width: w, height: 12,
                background: hi ? HF.accent : HF.borderDk, borderRadius: 2, opacity: hi ? 1 : 0.5,
              }} />
            ))}
            <div style={{ position: 'absolute', left: 60, top: 8, width: 110, height: 80, border: `1.5px solid ${HF.accent}`, borderRadius: 3 }} />
          </div>
        </div>
      </div>
    </div>
  );
}

window.HFNodes = HFNodes;
