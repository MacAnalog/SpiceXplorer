// Direction B — "Notebook" cells.
// Pattern: Jupyter / Observable. A linear, top-down story.
// Each step is a collapsible cell with an editable input and an inline output.
// Great for reproducibility, demos, and "show your work" engineering.

function NotebookWireframe() {
  const W = 1440, H = 900;
  const headerH = 52;

  const Cell = ({ idx, label, expanded = true, status = 'ok', children, kind = 'config' }) => (
    <div style={{
      display: 'grid',
      gridTemplateColumns: '52px 1fr',
      borderRadius: 5,
      background: expanded ? '#fff' : SK.softer,
      border: `1.5px solid ${SK.ink}`,
    }}>
      {/* gutter */}
      <div style={{
        borderRight: `1px solid ${SK.soft}`, padding: 8,
        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6,
      }}>
        <SkMono size={10} color={SK.mute}>[{idx}]</SkMono>
        <div style={{
          width: 22, height: 22, borderRadius: 999,
          border: `1.4px solid ${status === 'ok' ? SK.green : status === 'todo' ? SK.amber : SK.ink}`,
          background: status === 'ok' ? '#eaf2ec' : status === 'todo' ? '#fbf2db' : '#fff',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontFamily: SK.fontHand, fontSize: 12,
          color: status === 'ok' ? SK.green : status === 'todo' ? SK.amber : SK.ink,
        }}>{status === 'ok' ? '✓' : status === 'todo' ? '!' : '▷'}</div>
        <div style={{ flex: 1, width: 1.5, background: SK.soft }} />
      </div>
      {/* body */}
      <div style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <SkText size={15} style={{ fontWeight: 700 }}>{label}</SkText>
          <div style={{ flex: 1 }} />
          <SkPill bg={SK.softer} style={{ fontSize: 10 }}>{kind}</SkPill>
          <SkText size={11} color={SK.mute}>{expanded ? '⌄ collapse' : '› expand'}</SkText>
        </div>
        {expanded && children}
      </div>
    </div>
  );

  return (
    <div style={{ width: W, height: H, background: SK.paper, position: 'relative', overflow: 'hidden' }}>
      {/* header */}
      <div style={{
        height: headerH, borderBottom: `1.5px solid ${SK.ink}`, padding: '0 22px',
        display: 'flex', alignItems: 'center', gap: 14,
      }}>
        <SkTitle size={20}>SpiceXplorer · Notebook</SkTitle>
        <SkText size={13} color={SK.mute}>· cascode_ota.spx</SkText>
        <div style={{ flex: 1 }} />
        <SkBtn ghost>＋ Add cell</SkBtn>
        <SkBtn ghost>⤓ Export YAML</SkBtn>
        <SkBtn accent>▶ Run all</SkBtn>
      </div>

      {/* notebook scroll area */}
      <div style={{
        padding: '18px 80px', height: H - headerH, overflow: 'hidden',
        display: 'flex', flexDirection: 'column', gap: 14,
      }}>
        {/* Cell 1: Netlist + project */}
        <Cell idx={1} label="Project · netlist + workspace" kind="setup">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
            <SkBox dashed style={{ padding: 10, display: 'flex', flexDirection: 'column', gap: 4 }}>
              <SkText size={12} color={SK.mute}>Drag &amp; drop a .spice netlist  ·  or pick from workspace</SkText>
              <SkMono size={10}>spice/ota-improved.spice  ✓ parsed</SkMono>
              <SkMono size={10} color={SK.mute}>found 3 .param  ·  6 transistors  ·  2 bias nodes</SkMono>
            </SkBox>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <SkMono size={11}>simulator: ngspice</SkMono>
              <SkMono size={11}>tech: ihp-sg13g2</SkMono>
              <SkMono size={11}>ws_root: examples/OTA/cascode/ihp-sg13g2</SkMono>
              <SkMono size={11} color={SK.mute}>parallel_sim: true · save_sim: false</SkMono>
            </div>
          </div>
        </Cell>

        {/* Cell 2: DUT params */}
        <Cell idx={2} label="DUT parameters · 13 variables" kind="variables">
          <SkBox style={{ padding: 0, overflow: 'hidden' }}>
            <div style={{
              display: 'grid', gridTemplateColumns: '1.6fr 1fr 1fr 0.6fr 0.6fr 0.7fr',
              background: SK.softer, padding: '6px 10px',
              borderBottom: `1px solid ${SK.soft}`,
            }}>
              {['name', 'min', 'max', 'integer', 'log', 'freeze'].map((h) => (
                <SkMono key={h} size={10} color={SK.mute} style={{ textTransform: 'uppercase', letterSpacing: 0.5 }}>{h}</SkMono>
              ))}
            </div>
            {[
              ['X_DUT_M1M2_W', 'min_nfet_w', 'max_nfet_w'],
              ['X_DUT_M1M2_L', 'min_nfet_l', 'max_nfet_l'],
              ['X_DUT_M3M4_W', 'min_pfet_w', 'max_pfet_w'],
              ['X_DUT_M5_NG', '1', '5', 'int'],
              ['X_DUT_V_BIAS_1', '0', '1.2'],
            ].map((row, i) => (
              <div key={i} style={{
                display: 'grid', gridTemplateColumns: '1.6fr 1fr 1fr 0.6fr 0.6fr 0.7fr',
                padding: '5px 10px', borderBottom: i < 4 ? `1px dashed ${SK.soft}` : 'none',
                alignItems: 'center',
              }}>
                <SkMono size={11}>{row[0]}</SkMono>
                <SkMono size={11} color={SK.mute}>{row[1]}</SkMono>
                <SkMono size={11} color={SK.mute}>{row[2]}</SkMono>
                <SkText size={11} color={SK.mute}>{row[3] === 'int' ? '☑' : '☐'}</SkText>
                <SkText size={11} color={SK.mute}>☐</SkText>
                <SkText size={11} color={SK.mute}>☐</SkText>
              </div>
            ))}
            <div style={{ padding: '5px 10px' }}>
              <SkText size={11} color={SK.mute}>+ 8 more rows · click to expand</SkText>
            </div>
          </SkBox>
        </Cell>

        {/* Cell 3: Testbenches + PVT (collapsed) */}
        <Cell idx={3} label="Testbenches · 3 enabled  ·  PVT · 1 corner" expanded={false} kind="simulation" />

        {/* Cell 4: Target specs + shaping inline */}
        <Cell idx={4} label="Target specs · 6 metrics · score shaping" kind="objective" status="todo">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10 }}>
            {[
              { name: 'ugf', goal: '≥ 200 MHz', tol: '±10M', w: 100, sel: true },
              { name: 'dcgain', goal: '≥ 40 dB', tol: '±1', w: 100 },
              { name: 'pm', goal: '= 60° ±10', tol: '±10', w: 100 },
              { name: 'inoise', goal: '≤ 1.2 mV/√Hz', tol: '—', w: 100 },
              { name: 'idd', goal: '≤ 25 µA', tol: '—', w: 100 },
              { name: 'tsettle', goal: '≤ 15 µs', tol: '—', w: 10 },
            ].map((s) => (
              <div key={s.name} style={{
                border: `${s.sel ? 2 : 1.2}px solid ${s.sel ? SK.accent : SK.ink}`,
                borderRadius: 4, padding: 8,
                display: 'flex', flexDirection: 'column', gap: 4,
                background: s.sel ? '#fff' : 'transparent',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <SkMono size={11}>{s.name}</SkMono>
                  <SkMono size={10} color={SK.mute}>w={s.w}</SkMono>
                </div>
                <SkMono size={10} color={SK.mute}>{s.goal}  ·  tol {s.tol}</SkMono>
              </div>
            ))}
          </div>
          {/* shaping mini-panel inline */}
          <SkBox style={{ padding: 10, marginTop: 8, display: 'grid', gridTemplateColumns: '1fr 280px', gap: 14, alignItems: 'center' }}>
            <div>
              <SkText size={13}>Score shape for <SkMono size={11} color={SK.accent}>ugf</SkMono></SkText>
              <div style={{ display: 'flex', gap: 8, marginTop: 6 }}>
                <SkPill color={SK.accent} bg={SK.accentSoft}>● sigmoid</SkPill>
                <SkPill color={SK.mute}>○ linear</SkPill>
              </div>
              <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 8 }}>
                <SkText size={11} color={SK.mute}>try value</SkText>
                <div style={{ height: 6, flex: 1, border: `1.2px solid ${SK.ink}`, borderRadius: 999, position: 'relative' }}>
                  <div style={{ position: 'absolute', left: '36%', top: -4, width: 12, height: 14, background: SK.accent, borderRadius: 3 }} />
                </div>
                <SkMono size={11}>187 MHz</SkMono>
              </div>
              <SkText size={11} color={SK.mute} style={{ marginTop: 4 }}>
                sigmoid penalty <SkMono color={SK.accent}>0.18</SkMono>  ·  linear <SkMono color={SK.blue}>0.87</SkMono>  →  same miss, very different blame
              </SkText>
            </div>
            <SkPenaltyCurve width={280} height={120} />
          </SkBox>
        </Cell>

        {/* Cell 5: Optimizer run */}
        <Cell idx={5} label="Optimize · LhsDE · budget 2000" kind="run" status="run">
          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <SkBtn accent>▶ Run</SkBtn>
            <SkBtn ghost>↻ Replay sigmoid_de</SkBtn>
            <div style={{ flex: 1, height: 8, border: `1.2px solid ${SK.ink}`, borderRadius: 2, position: 'relative' }}>
              <div style={{ width: '42%', height: '100%', background: SK.accent }} />
            </div>
            <SkMono size={11} color={SK.mute}>847 / 2000  ·  best 0.412</SkMono>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 12 }}>
            <SkBox style={{ padding: 6 }}>
              <SkChart width={620} height={180} lines={['ink', 'accent']} title="score convergence" target={false} />
            </SkBox>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, alignContent: 'start' }}>
              <SkSpecChip name="ugf" pass value="214M" target="≥ 200M" unit="" />
              <SkSpecChip name="dcgain" pass value="46.2" target="≥ 40" unit="dB" />
              <SkSpecChip name="pm" pass={false} value="52°" target="60±10°" unit="" />
              <SkSpecChip name="inoise" pass value="0.98m" target="≤1.2m" unit="" />
              <SkSpecChip name="idd" pass={false} value="31µ" target="≤25µ" unit="" />
              <SkSpecChip name="tsettle" pass value="12.4µ" target="≤15µ" unit="" />
            </div>
          </div>
        </Cell>
      </div>

      {/* annotations */}
      <SkCallout x={W - 320} y={130} w={300}>
        ↑ every step a re-runnable cell · setup, shape, run, explore in one scroll
      </SkCallout>
      <SkCallout x={20} y={H - 50} w={500}>
        ↑ score shaping lives next to the spec it shapes · not in a separate tab
      </SkCallout>
    </div>
  );
}

window.NotebookWireframe = NotebookWireframe;
