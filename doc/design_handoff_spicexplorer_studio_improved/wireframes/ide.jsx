// Direction E — "IDE three-pane".
// Pattern: VS Code. Engineer comfort food.
// Activity bar · file tree · split editor (YAML + visual form) · right preview · bottom panel (terminal + SSE).
// Lowest learning curve for someone already living in editors.

function IDEWireframe() {
  const W = 1440, H = 900;
  const topH = 36;
  const tabH = 32;
  const statusH = 22;
  const activityW = 44;
  const treeW = 220;
  const rightW = 320;
  const bottomH = 200;

  return (
    <div style={{
      width: W, height: H, background: SK.paper, position: 'relative', overflow: 'hidden',
      display: 'flex', flexDirection: 'column',
    }}>
      {/* === Title bar === */}
      <div style={{
        height: topH, borderBottom: `1.5px solid ${SK.ink}`, padding: '0 14px',
        display: 'flex', alignItems: 'center', gap: 14,
      }}>
        <SkTitle size={15}>SpiceXplorer Studio</SkTitle>
        <div style={{ display: 'flex', gap: 14, marginLeft: 14 }}>
          {['File', 'Edit', 'Project', 'Run', 'Compare', 'Help'].map((m) => (
            <SkText key={m} size={12} color={SK.mute}>{m}</SkText>
          ))}
        </div>
        <div style={{ flex: 1 }} />
        <SkBox dashed style={{ padding: '2px 10px', width: 320 }}>
          <SkMono size={10} color={SK.mute}>⌘K  ·  go to spec · device · run …</SkMono>
        </SkBox>
        <div style={{ flex: 1 }} />
        <SkPill color={SK.accent} bg={SK.accentSoft}>● run · iter 847</SkPill>
        <SkBtn accent>▶ Run</SkBtn>
      </div>

      <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
        {/* === Activity bar === */}
        <div style={{
          width: activityW, borderRight: `1.5px solid ${SK.ink}`,
          display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '10px 0', gap: 14,
        }}>
          {[
            ['☰', true, 'Explorer'],
            ['◬', false, 'Schematic'],
            ['∿', false, 'Specs'],
            ['▶', false, 'Run'],
            ['◫', false, 'Compare'],
            ['⌗', false, 'Settings'],
          ].map(([icon, sel, _l], i) => (
            <div key={i} style={{
              width: 30, height: 30,
              borderLeft: sel ? `2px solid ${SK.accent}` : '2px solid transparent',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              color: sel ? SK.ink : SK.mute,
              fontFamily: SK.fontHand, fontSize: 20,
            }}>{icon}</div>
          ))}
        </div>

        {/* === File tree === */}
        <div style={{ width: treeW, borderRight: `1.5px solid ${SK.ink}`, padding: 10, display: 'flex', flexDirection: 'column', gap: 4 }}>
          <SkText size={11} color={SK.mute} style={{ textTransform: 'uppercase', letterSpacing: 0.5 }}>Cascode OTA</SkText>
          {[
            ['📁 spice/', 0, false],
            ['  ota-improved.spice', 1, false],
            ['  ota-improved_tb-loopgain.spice', 1, false],
            ['  ota-improved_tb-noise.spice', 1, false],
            ['  ota-improved_tb-tran.spice', 1, false],
            ['📁 sizing/', 0, false],
            ['  project_setup.yaml', 1, true],
            ['📁 runs/', 0, false],
            ['  r12 · sigmoid_de  ●', 1, false],
            ['  r11 · linear_de   ✓', 1, false],
            ['  r10 · CMA wider   ✓', 1, false],
            ['  r09 · blind       ✓', 1, false],
            ['📁 checkpoints/', 0, false],
          ].map(([n, ind, sel], i) => (
            <div key={i} style={{
              display: 'flex', alignItems: 'center',
              padding: '2px 4px', paddingLeft: 4 + ind * 8,
              background: sel ? SK.accentSoft : 'transparent',
              borderRadius: 3,
            }}>
              <SkMono size={11} color={sel ? SK.ink : (n.includes('📁') ? SK.ink : SK.mute)}>{n}</SkMono>
            </div>
          ))}
          <div style={{ flex: 1 }} />
          <SkText size={10} color={SK.mute}>git · dev/ui · clean</SkText>
        </div>

        {/* === Center editor + right preview === */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
          {/* tabs */}
          <div style={{
            height: tabH, borderBottom: `1px solid ${SK.soft}`,
            display: 'flex', alignItems: 'flex-end', gap: 2, paddingLeft: 8,
          }}>
            {[
              { name: 'project_setup.yaml', sel: true, dirty: true },
              { name: 'ota-improved.spice', sel: false },
              { name: 'r12 · live', sel: false, dot: SK.accent },
            ].map((t) => (
              <div key={t.name} style={{
                padding: '6px 12px',
                borderTop: t.sel ? `2px solid ${SK.accent}` : '2px solid transparent',
                borderRight: `1px solid ${SK.soft}`,
                background: t.sel ? '#fff' : SK.softer,
                display: 'flex', alignItems: 'center', gap: 6,
              }}>
                {t.dot && <span style={{ width: 6, height: 6, borderRadius: 999, background: t.dot }} />}
                <SkMono size={11}>{t.name}{t.dirty ? '  •' : ''}</SkMono>
                <SkText size={10} color={SK.mute}>×</SkText>
              </div>
            ))}
            <div style={{ flex: 1 }} />
            <div style={{ display: 'flex', gap: 0, padding: '4px 6px' }}>
              <SkPill bg={SK.softer} style={{ fontSize: 10, padding: '1px 8px' }}>YAML</SkPill>
              <SkPill style={{ fontSize: 10, padding: '1px 8px', marginLeft: 4 }}>Form</SkPill>
              <SkPill style={{ fontSize: 10, padding: '1px 8px', marginLeft: 4 }}>Split</SkPill>
            </div>
          </div>

          {/* editor + right preview row */}
          <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
            {/* editor */}
            <div style={{
              flex: 1, background: '#fdfaf2', padding: 12, minWidth: 0,
              fontFamily: SK.fontMono, fontSize: 12, color: SK.ink,
              lineHeight: 1.65, overflow: 'hidden', position: 'relative',
            }}>
              {[
                ['project:', SK.ink],
                ['  name: CASCODE-OTA', SK.ink],
                ['  simulator: ngspice', SK.ink],
                ['  netlist: spice/ota-improved.spice', SK.ink],
                ['  ', SK.ink],
                ['  dut_params:', SK.ink],
                ['    - name: X_DUT_M1M2_W', SK.ink],
                ['      min_val: min_nfet_w', SK.mute],
                ['      max_val: max_nfet_w', SK.mute],
                ['    - name: X_DUT_M5_NG', SK.ink],
                ['      min_val: 1', SK.mute],
                ['      max_val: 5', SK.mute],
                ['      is_integer: true', SK.accent],
                ['  ', SK.ink],
                ['  optimizer_config:', SK.ink],
                ['    type: nevergrad', SK.ink],
                ['    name: LhsDE   # was LogBFGSCMAPlus', SK.amber],
                ['    budget: 2000', SK.ink],
                ['    target_specs:', SK.ink],
                ['      - name: ugf', SK.ink],
                ['        goal: exceed', SK.ink],
                ['        target: 200e6', SK.ink],
                ['        error_type: relative-absolute', SK.mute],
                ['        reward_type: log', SK.mute],
              ].map(([line, color], i) => (
                <div key={i} style={{ display: 'flex', gap: 10 }}>
                  <span style={{ color: SK.mute, width: 24, textAlign: 'right' }}>{i + 1}</span>
                  <span style={{ color }}>{line}</span>
                </div>
              ))}
              {/* gutter hint badge */}
              <div style={{ position: 'absolute', right: 18, top: 12 }}>
                <SkBox accent style={{ padding: '4px 8px', background: '#fff' }}>
                  <SkText size={11} color={SK.accent}>① unsaved · cmd-S to apply</SkText>
                </SkBox>
              </div>
            </div>

            {/* right preview */}
            <div style={{ width: rightW, borderLeft: `1.5px solid ${SK.ink}`, display: 'flex', flexDirection: 'column' }}>
              {/* preview switch */}
              <div style={{ borderBottom: `1px solid ${SK.soft}`, padding: 8, display: 'flex', gap: 6 }}>
                {['Live preview', 'Schematic', 'Spec status'].map((t, i) => (
                  <SkPill key={t} bg={i === 0 ? SK.ink : 'transparent'} color={i === 0 ? SK.paper : SK.ink}>{t}</SkPill>
                ))}
              </div>
              <div style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 10, overflow: 'hidden' }}>
                <SkText size={12}>Parsed summary</SkText>
                <SkBox soft style={{ padding: 8, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
                  {[
                    ['simulator', 'ngspice'],
                    ['tech', 'ihp-sg13g2'],
                    ['testbenches', '3 enabled'],
                    ['dut params', '13 (1 int)'],
                    ['target specs', '6'],
                    ['optimizer', 'LhsDE · 2000'],
                  ].map(([k, v]) => (<React.Fragment key={k}>
                    <SkMono size={10} color={SK.mute}>{k}</SkMono>
                    <SkMono size={10}>{v}</SkMono>
                  </React.Fragment>))}
                </SkBox>

                <SkText size={12}>Live · r12</SkText>
                <SkBox style={{ padding: 6 }}>
                  <SkChart width={290} height={130} lines={['ink', 'accent']} target={false} noAxis />
                </SkBox>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 5 }}>
                  {[
                    ['ugf', true, '214M', '≥200M'],
                    ['dcgain', true, '46.2', '≥40'],
                    ['pm', false, '52°', '60±10°'],
                    ['inoise', true, '0.98m', '≤1.2m'],
                    ['idd', false, '31µ', '≤25µ'],
                    ['tsettle', true, '12.4µ', '≤15µ'],
                  ].map((s) => (
                    <SkSpecChip key={s[0]} name={s[0]} pass={s[1]} value={s[2]} target={s[3]} unit="" />
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* bottom panel */}
          <div style={{ height: bottomH, borderTop: `1.5px solid ${SK.ink}`, display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', gap: 14, borderBottom: `1px solid ${SK.soft}`, padding: '4px 12px' }}>
              {['Terminal', 'Optimizer log', 'Problems', 'Compare (r12 vs r11)'].map((t, i) => (
                <SkText key={t} size={12} color={i === 1 ? SK.accent : SK.mute}
                  style={i === 1 ? { borderBottom: `2px solid ${SK.accent}`, paddingBottom: 2 } : {}}>{t}</SkText>
              ))}
              <div style={{ flex: 1 }} />
              <SkText size={11} color={SK.mute}>tail · stream</SkText>
            </div>
            <div style={{ flex: 1, padding: 10, overflow: 'hidden', display: 'flex', flexDirection: 'column', gap: 2 }}>
              {[
                ['845', 'F=0.408', 'ugf=210M dcgain=45.9dB pm=51° inoise=1.01m idd=32µ tsettle=12.8µ'],
                ['846', 'F=0.410', 'ugf=212M dcgain=46.0dB pm=51° inoise=0.99m idd=31µ tsettle=12.6µ'],
                ['847', 'F=0.412', 'ugf=214M dcgain=46.2dB pm=52° inoise=0.98m idd=31µ tsettle=12.4µ  ← new best', SK.accent],
                ['', '', '↳ pm and idd still failing · sensitivity → M5 NG, M1M2 W', SK.mute],
              ].map((row, i) => (
                <div key={i} style={{ display: 'flex', gap: 12 }}>
                  <SkMono size={10} color={SK.mute} style={{ width: 28 }}>{row[0]}</SkMono>
                  <SkMono size={10} color={row[3] || SK.ink} style={{ width: 80 }}>{row[1]}</SkMono>
                  <SkMono size={10} color={row[3] || SK.ink}>{row[2]}</SkMono>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* status bar */}
      <div style={{
        height: statusH, background: SK.ink, color: SK.paper,
        padding: '0 12px', display: 'flex', alignItems: 'center', gap: 12,
      }}>
        <SkMono size={10} color={SK.paper}>⎇ dev/ui</SkMono>
        <SkMono size={10} color={SK.paper}>● r12 live  ·  42 min left</SkMono>
        <SkMono size={10} color={SK.paper}>best F = 0.412</SkMono>
        <div style={{ flex: 1 }} />
        <SkMono size={10} color={SK.paper}>YAML  ·  Ln 17  ·  spaces 2</SkMono>
      </div>

      <SkCallout x={activityW + treeW + 12} y={H - bottomH - 80} w={500}>
        ↑ keep your YAML · but give it a live preview, a schematic, and a terminal log next to it
      </SkCallout>
    </div>
  );
}

window.IDEWireframe = IDEWireframe;
