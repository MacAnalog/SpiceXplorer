// Hi-fi E — IDE three-pane (VS Code pattern).

function HFIDE() {
  const W = 1440, H = 900;
  const titleH = 36;
  const tabH = 34;
  const statusH = 24;
  const activityW = 48;
  const treeW = 240;
  const rightW = 340;
  const bottomH = 220;

  // YAML editor lines (with syntax color)
  const yamlLines = [
    { t: 'project:', c: HF.text },
    { t: '  name: CASCODE-OTA', c: HF.text, hl: 'name' },
    { t: '  simulator: ngspice', c: HF.text },
    { t: '  netlist: spice/ota-improved.spice', c: HF.text },
    { t: '  ws_root: examples/OTA/cascode/ihp-sg13g2/', c: HF.text },
    { t: '', c: HF.text },
    { t: '  dut_params:', c: HF.text },
    { t: '    - name: X_DUT_M1M2_W', c: HF.text },
    { t: '      min_val: min_nfet_w', c: HF.textMute },
    { t: '      max_val: max_nfet_w', c: HF.textMute },
    { t: '    - name: X_DUT_M5_NG', c: HF.text },
    { t: '      min_val: 1', c: HF.textMute },
    { t: '      max_val: 5', c: HF.textMute },
    { t: '      is_integer: true', c: HF.accent, hl: 'badge' },
    { t: '', c: HF.text },
    { t: '  optimizer_config:', c: HF.text },
    { t: '    type: nevergrad', c: HF.text },
    { t: '    name: LhsDE   # was LogBFGSCMAPlus', c: HF.warn },
    { t: '    budget: 2000', c: HF.text },
    { t: '    target_specs:', c: HF.text },
    { t: '      - name: ugf', c: HF.text },
    { t: '        testbench: tb_ac', c: HF.textMute },
    { t: '        goal: exceed', c: HF.text },
    { t: '        target: 200e6', c: HF.text, hl: 'target' },
    { t: '        tolerance: 10e6', c: HF.textMute },
    { t: '        error_type: relative-absolute', c: HF.textMute },
    { t: '        reward_type: log', c: HF.textMute },
  ];

  // Live log lines
  const logLines = [
    { it: 845, F: '0.408', payload: 'ugf=210M  dcgain=45.9dB  pm=51°  inoise=1.01m  idd=32µ  tsettle=12.8µ' },
    { it: 846, F: '0.410', payload: 'ugf=212M  dcgain=46.0dB  pm=51°  inoise=0.99m  idd=31µ  tsettle=12.6µ' },
    { it: 847, F: '0.412', payload: 'ugf=214M  dcgain=46.2dB  pm=52°  inoise=0.98m  idd=31µ  tsettle=12.4µ',
      tag: 'new best', tagTone: HF.accent },
    { it: null, F: null, payload: '↳ blame: pm sensitivity high on M5 NG · idd sensitivity high on M1M2 W', tag: 'hint', tagTone: HF.textMute },
  ];

  return (
    <div style={{ width: W, height: H, background: HF.bg, fontFamily: HF.ui, color: HF.text, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Title bar */}
      <div style={{
        height: titleH, background: HF.panel, borderBottom: `1px solid ${HF.border}`,
        display: 'flex', alignItems: 'center', padding: '0 12px', gap: 16,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 14, height: 14, borderRadius: 999, background: HF.accent }} />
          <HFText size={13} weight={600}>SpiceXplorer Studio</HFText>
        </div>
        {['File', 'Edit', 'Project', 'Run', 'Compare', 'View', 'Help'].map((m) => (
          <HFText key={m} size={12} color={HF.textMute}>{m}</HFText>
        ))}
        <div style={{ flex: 1 }} />
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          background: HF.panelAlt, border: `1px solid ${HF.border}`,
          borderRadius: 6, padding: '3px 10px', width: 360,
        }}>
          <HFIcon name="search" size={12} color={HF.textDim} />
          <HFText size={12} color={HF.textDim}>Go to spec · device · run …</HFText>
          <div style={{ flex: 1 }} />
          <HFMono size={10} color={HF.textDim}>⌘K</HFMono>
        </div>
        <div style={{ flex: 1 }} />
        <HFBadge tone="indigo"><span style={{ width: 6, height: 6, borderRadius: 999, background: HF.accent, marginRight: 3 }} />r12 · iter 847</HFBadge>
        <HFButton kind="primary" size="sm" icon={<HFIcon name="play" size={11} color="#fff" />}>Run</HFButton>
      </div>

      <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
        {/* Activity bar */}
        <div style={{
          width: activityW, borderRight: `1px solid ${HF.border}`, background: HF.panel,
          display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '10px 0', gap: 6,
        }}>
          {[
            ['file', true, 'Explorer'],
            ['chip', false, 'Schematic'],
            ['sliders', false, 'Specs'],
            ['graph', false, 'Pipeline'],
            ['activity', false, 'Runs'],
            ['diff', false, 'Compare'],
          ].map(([icon, sel, l], i) => (
            <div key={i} title={l} style={{
              width: 34, height: 34, borderRadius: 6,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: sel ? HF.accentSoft : 'transparent',
              borderLeft: sel ? `2px solid ${HF.accent}` : '2px solid transparent',
              marginLeft: -2,
            }}>
              <HFIcon name={icon} size={18} color={sel ? HF.accent : HF.textMute} />
            </div>
          ))}
          <div style={{ flex: 1 }} />
          <HFIcon name="git" size={18} color={HF.textMute} />
          <HFIcon name="settings" size={18} color={HF.textMute} />
        </div>

        {/* File tree */}
        <div style={{ width: treeW, borderRight: `1px solid ${HF.border}`, background: HF.panel, display: 'flex', flexDirection: 'column' }}>
          <div style={{ padding: '10px 12px', borderBottom: `1px solid ${HF.border}`, display: 'flex', justifyContent: 'space-between' }}>
            <HFText size={11} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase', letterSpacing: 0.4 }}>Cascode OTA</HFText>
            <HFIcon name="plus" size={12} color={HF.textMute} />
          </div>
          <div style={{ flex: 1, padding: '6px 6px', display: 'flex', flexDirection: 'column', gap: 1, overflow: 'hidden' }}>
            {[
              { l: 'spice', ind: 0, ic: 'folder', open: true },
              { l: 'ota-improved.spice', ind: 1, ic: 'file' },
              { l: 'ota-improved_tb-loopgain.spice', ind: 1, ic: 'file' },
              { l: 'ota-improved_tb-noise.spice', ind: 1, ic: 'file' },
              { l: 'ota-improved_tb-tran.spice', ind: 1, ic: 'file' },
              { l: 'sizing', ind: 0, ic: 'folder', open: true },
              { l: 'project_setup.yaml', ind: 1, ic: 'file', sel: true, dirty: true },
              { l: 'runs', ind: 0, ic: 'folder', open: true },
              { l: 'r12 · sigmoid_de', ind: 1, ic: 'dot', state: 'live' },
              { l: 'r11 · linear_de', ind: 1, ic: 'dot', state: 'done' },
              { l: 'r10 · CMA wider', ind: 1, ic: 'dot', state: 'done' },
              { l: 'r09 · blind LHS', ind: 1, ic: 'dot', state: 'done' },
              { l: 'r08 · pm hard', ind: 1, ic: 'dot', state: 'fail' },
              { l: 'checkpoints', ind: 0, ic: 'folder' },
            ].map((r, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '3px 6px', paddingLeft: 6 + r.ind * 12, borderRadius: 4,
                background: r.sel ? HF.accentSoft : 'transparent',
              }}>
                {r.ic === 'folder' && <HFIcon name="folder" size={12} color={HF.textMute} />}
                {r.ic === 'file' && <HFIcon name="file" size={12} color={r.sel ? HF.accent : HF.textMute} />}
                {r.ic === 'dot' && <span style={{
                  width: 8, height: 8, borderRadius: 999,
                  background: r.state === 'live' ? HF.accent : r.state === 'fail' ? HF.error : HF.success,
                }} />}
                <HFText size={12} weight={r.sel ? 600 : 400} color={r.sel ? HF.accent : HF.text}>{r.l}</HFText>
                <div style={{ flex: 1 }} />
                {r.dirty && <span style={{ color: HF.warn, fontFamily: HF.mono, fontSize: 10 }}>●</span>}
              </div>
            ))}
          </div>
          <div style={{ borderTop: `1px solid ${HF.border}`, padding: '6px 10px', display: 'flex', alignItems: 'center', gap: 6 }}>
            <HFIcon name="git" size={12} color={HF.textMute} />
            <HFText size={11} color={HF.textMute}>dev/ui · clean</HFText>
          </div>
        </div>

        {/* Editor + bottom panel */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
          {/* Tab bar */}
          <div style={{
            height: tabH, borderBottom: `1px solid ${HF.border}`, background: HF.panel,
            display: 'flex', alignItems: 'stretch',
          }}>
            {[
              { name: 'project_setup.yaml', sel: true, dirty: true, icon: 'file' },
              { name: 'ota-improved.spice', sel: false, icon: 'file' },
              { name: 'r12 · live', sel: false, icon: 'dot', state: 'live' },
            ].map((t, i) => (
              <div key={t.name} style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '0 12px',
                borderRight: `1px solid ${HF.border}`,
                background: t.sel ? HF.bg : HF.panel,
                borderTop: t.sel ? `2px solid ${HF.accent}` : '2px solid transparent',
                marginTop: t.sel ? -1 : 0,
              }}>
                {t.icon === 'file' && <HFIcon name="file" size={12} color={t.sel ? HF.accent : HF.textMute} />}
                {t.icon === 'dot' && <span style={{ width: 8, height: 8, borderRadius: 999, background: HF.accent }} />}
                <HFText size={12} weight={t.sel ? 600 : 400} color={t.sel ? HF.text : HF.textMute}>{t.name}</HFText>
                {t.dirty && <span style={{ color: HF.warn, fontFamily: HF.mono, fontSize: 11 }}>●</span>}
                <HFIcon name="x" size={10} color={HF.textDim} />
              </div>
            ))}
            <div style={{ flex: 1 }} />
            <div style={{ padding: '4px 8px', display: 'flex', gap: 6, alignItems: 'center' }}>
              <div style={{ display: 'flex', background: HF.panelAlt, border: `1px solid ${HF.border}`, borderRadius: 5, overflow: 'hidden' }}>
                {[['YAML', true], ['Form', false], ['Split', false]].map(([l, sel]) => (
                  <span key={l} style={{
                    padding: '3px 10px', background: sel ? HF.panel : 'transparent',
                    borderRight: l !== 'Split' ? `1px solid ${HF.border}` : 'none',
                    fontFamily: HF.ui, fontSize: 11, fontWeight: sel ? 600 : 500,
                    color: sel ? HF.text : HF.textMute,
                  }}>{l}</span>
                ))}
              </div>
            </div>
          </div>

          {/* Editor + right preview */}
          <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
            {/* Code editor */}
            <div style={{
              flex: 1, background: HF.bg, padding: '14px 0 14px 8px',
              fontFamily: HF.mono, fontSize: 12.5, lineHeight: 1.7, color: HF.text,
              overflow: 'hidden', minWidth: 0, position: 'relative',
            }}>
              {yamlLines.map((l, i) => (
                <div key={i} style={{
                  display: 'flex', gap: 14, padding: '0 12px',
                  background: i === 13 ? HF.accentSoft : 'transparent',
                  borderLeft: i === 13 ? `2px solid ${HF.accent}` : '2px solid transparent',
                  marginLeft: -8, paddingLeft: 14,
                }}>
                  <span style={{ color: HF.textDim, width: 24, textAlign: 'right', flexShrink: 0 }}>{i + 1}</span>
                  <span style={{ color: l.c, whiteSpace: 'pre' }}>{l.t}</span>
                </div>
              ))}
              {/* Minimap (right edge of editor) */}
              <div style={{
                position: 'absolute', top: 8, right: 8, width: 56, height: '90%',
                background: HF.panelAlt, border: `1px solid ${HF.border}`, borderRadius: 4,
                padding: 4, display: 'flex', flexDirection: 'column', gap: 2,
              }}>
                {yamlLines.map((l, i) => (
                  <div key={i} style={{
                    height: 2.5,
                    width: l.t ? `${Math.min(95, l.t.length * 1.4)}%` : '20%',
                    background: i === 13 ? HF.accent : HF.borderDk,
                    opacity: l.t ? 0.7 : 0.3,
                    borderRadius: 1,
                  }} />
                ))}
              </div>
            </div>

            {/* Right preview */}
            <div style={{ width: rightW, borderLeft: `1px solid ${HF.border}`, background: HF.panel, display: 'flex', flexDirection: 'column' }}>
              {/* Preview tabs */}
              <div style={{ display: 'flex', borderBottom: `1px solid ${HF.border}`, padding: '4px 8px', gap: 4 }}>
                {[['Preview', true], ['Schematic', false], ['Live', false]].map(([l, sel]) => (
                  <span key={l} style={{
                    padding: '5px 10px', borderRadius: 5,
                    background: sel ? HF.accentSoft : 'transparent',
                    color: sel ? HF.accent : HF.textMute,
                    fontFamily: HF.ui, fontSize: 12, fontWeight: sel ? 600 : 500,
                  }}>{l}</span>
                ))}
              </div>

              <div style={{ flex: 1, overflow: 'hidden', padding: 14, display: 'flex', flexDirection: 'column', gap: 12 }}>
                <HFSectionLabel>Parsed summary</HFSectionLabel>
                <HFCard pad={10}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
                    {[
                      ['simulator', 'ngspice'],
                      ['tech', 'ihp-sg13g2'],
                      ['testbenches', '3 enabled'],
                      ['dut params', '13 (1 int)'],
                      ['target specs', '6'],
                      ['optimizer', 'LhsDE · 2000'],
                    ].map(([k, v]) => (<React.Fragment key={k}>
                      <HFMono size={11} color={HF.textMute}>{k}</HFMono>
                      <HFMono size={11} weight={500} style={{ textAlign: 'right' }}>{v}</HFMono>
                    </React.Fragment>))}
                  </div>
                </HFCard>

                <HFSectionLabel action={<HFBadge tone="indigo">r12 · live</HFBadge>}>Convergence</HFSectionLabel>
                <HFCard pad={6}>
                  <HFChart width={300} height={120} lines={[
                    { kind: 'raw',  color: HF.textDim, opacity: 0.6 },
                    { kind: 'conv', color: HF.accent, width: 1.8 },
                  ]} noAxis />
                </HFCard>

                <HFSectionLabel action={<HFBadge tone="warn">2 failing</HFBadge>}>Spec status</HFSectionLabel>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                  {[
                    ['ugf',     true,  '214M', '≥200M'],
                    ['dcgain',  true,  '46.2', '≥40dB'],
                    ['pm',      false, '52°',  '60±10°'],
                    ['inoise',  true,  '0.98m','≤1.2m'],
                    ['idd',     false, '31µ',  '≤25µ'],
                    ['tsettle', true,  '12.4µ','≤15µ'],
                  ].map((s) => <HFSpecChip key={s[0]} name={s[0]} pass={s[1]} value={s[2]} target={s[3]} />)}
                </div>
              </div>
            </div>
          </div>

          {/* Bottom panel */}
          <div style={{ height: bottomH, borderTop: `1px solid ${HF.border}`, background: HF.panelInv, color: HF.inkInv, display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', padding: '0 12px', borderBottom: `1px solid ${HF.borderInv}`, alignItems: 'stretch', height: 32 }}>
              {[
                ['Terminal', false],
                ['Optimizer log', true, <HFBadge tone="indigo" style={{ fontSize: 9, padding: '0 5px' }}>847</HFBadge>],
                ['Problems', false, <HFBadge tone="warn" style={{ fontSize: 9, padding: '0 5px' }}>2</HFBadge>],
                ['Diff · r12 ↔ r11', false],
              ].map(([l, sel, extra]) => (
                <div key={l} style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  padding: '0 12px',
                  borderBottom: sel ? `2px solid ${HF.accent}` : '2px solid transparent',
                  color: sel ? '#fafaf9' : '#a8a29e',
                }}>
                  <HFText size={12} weight={sel ? 600 : 500} color={sel ? '#fafaf9' : '#a8a29e'}>{l}</HFText>
                  {extra}
                </div>
              ))}
              <div style={{ flex: 1 }} />
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#a8a29e', fontSize: 11 }}>
                <HFIcon name="terminal" size={12} color="#a8a29e" />
                <span style={{ fontFamily: HF.ui }}>tail · streaming</span>
              </div>
            </div>
            <div style={{ flex: 1, padding: '10px 14px', overflow: 'hidden', display: 'flex', flexDirection: 'column', gap: 4 }}>
              {logLines.map((row, i) => (
                <div key={i} style={{ display: 'flex', gap: 14, alignItems: 'baseline' }}>
                  {row.it != null
                    ? <HFMono size={11} color="#78716c" style={{ width: 36 }}>{row.it}</HFMono>
                    : <span style={{ width: 36 }} />}
                  {row.F
                    ? <HFMono size={11} color={row.tag === 'new best' ? HF.accent : '#fafaf9'} weight={500} style={{ width: 70 }}>F={row.F}</HFMono>
                    : <span style={{ width: 70 }} />}
                  <HFMono size={11} color={row.tag === 'hint' ? '#a8a29e' : '#e7e5e4'}>{row.payload}</HFMono>
                  {row.tag && <span style={{ marginLeft: 'auto', color: row.tagTone, fontSize: 10, fontFamily: HF.mono }}>← {row.tag}</span>}
                </div>
              ))}
              {/* prompt */}
              <div style={{ display: 'flex', gap: 14, marginTop: 4 }}>
                <span style={{ width: 36 }} />
                <HFMono size={11} color={HF.accent}>$</HFMono>
                <HFMono size={11} color="#fafaf9">spx run --algo LhsDE --budget 2000 --loss sigmoid<span style={{ background: HF.accent, marginLeft: 1 }}>&nbsp;</span></HFMono>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Status bar */}
      <div style={{
        height: statusH, background: HF.accent, color: '#fff',
        display: 'flex', alignItems: 'center', padding: '0 12px', gap: 14, fontSize: 11,
      }}>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
          <HFIcon name="git" size={11} color="#fff" /> dev/ui
        </span>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
          <span style={{ width: 8, height: 8, borderRadius: 999, background: '#fff' }} />
          r12 live · 14 min left
        </span>
        <HFMono size={11} color="#fff">F best = 0.412</HFMono>
        <HFMono size={11} color="#fff">4/6 specs passing</HFMono>
        <div style={{ flex: 1 }} />
        <HFMono size={11} color="#fff">YAML · Ln 17, Col 11</HFMono>
        <HFMono size={11} color="#fff">UTF-8</HFMono>
        <HFMono size={11} color="#fff">spaces: 2</HFMono>
      </div>
    </div>
  );
}

window.HFIDE = HFIDE;
