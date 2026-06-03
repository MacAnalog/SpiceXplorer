// Studio shell — top-level component, title bar, activity bar, tab strip, status bar, overlays.

function StudioActivityBar() {
  const { s, setActivity } = useStudio();
  const items = [
    ['activity', 'runs',      'Runs',      12],
    ['file',     'setup',     'Setup'],
    ['graph',    'pipeline',  'Pipeline'],
    ['chip',     'schematic', 'Schematic'],
    ['sliders',  'specs',     'Specs',     6],
    ['diff',     'compare',   'Compare'],
  ];
  return (
    <div style={{
      width: 52, borderRight: `1px solid ${HF.border}`, background: HF.panel,
      display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '10px 0', gap: 4,
    }}>
      {items.map(([icon, id, label, badge]) => {
        const sel = s.activeActivity === id;
        return (
          <div key={id} title={label} onClick={() => setActivity(id)} style={{
            position: 'relative', width: 36, height: 36, borderRadius: 7,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: sel ? HF.accentSoft : 'transparent',
            borderLeft: sel ? `2px solid ${HF.accent}` : '2px solid transparent',
            marginLeft: -2, cursor: 'pointer',
          }}>
            <HFIcon name={icon} size={18} color={sel ? HF.accent : HF.textMute} />
            {badge && (
              <span style={{
                position: 'absolute', top: 2, right: 2,
                background: HF.accent, color: '#fff', borderRadius: 999,
                padding: '0 4px', fontSize: 9, fontFamily: HF.ui, fontWeight: 600, lineHeight: '13px',
              }}>{badge}</span>
            )}
          </div>
        );
      })}
      <div style={{ flex: 1 }} />
      <HFIcon name="git" size={18} color={HF.textMute} />
      <HFIcon name="settings" size={18} color={HF.textMute} />
    </div>
  );
}

function StudioTitleBar() {
  const { s, set, demos, replays, algos } = useStudio();
  const [runOpen, setRunOpen] = React.useState(false);
  const [demoOpen, setDemoOpen] = React.useState(false);

  return (
    <div style={{
      height: 38, background: HF.panel, borderBottom: `1px solid ${HF.border}`,
      display: 'flex', alignItems: 'center', padding: '0 14px', gap: 14,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
        <div style={{ width: 18, height: 18, borderRadius: 5, background: `linear-gradient(135deg, ${HF.accent}, ${HF.accentHov})`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <HFIcon name="activity" size={11} color="#fff" />
        </div>
        <HFText size={13} weight={600}>SpiceXplorer Studio</HFText>
      </div>
      <HFText size={12} color={HF.textDim}>·</HFText>
      <div style={{ position: 'relative' }}>
        <span onClick={() => setDemoOpen(v => !v)} style={{ cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 6 }}>
          <HFText size={12}>OTA Cascode · ihp-sg13g2</HFText>
          <HFIcon name="chevD" size={11} color={HF.textMute} />
        </span>
        {demoOpen && (
          <div style={{ position: 'absolute', top: 26, left: 0, width: 300, background: HF.panel, border: `1px solid ${HF.border}`, borderRadius: 7, boxShadow: '0 6px 18px rgba(0,0,0,0.10)', padding: 6, zIndex: 20 }}>
            <HFText size={10} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase', letterSpacing: 0.4, padding: '4px 8px' }}>Recent projects</HFText>
            {demos.map(d => (
              <div key={d.id} onClick={() => setDemoOpen(false)} style={{ padding: '6px 8px', borderRadius: 5, cursor: 'pointer' }}
                onMouseEnter={(e) => e.currentTarget.style.background = HF.accentSoft}
                onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}>
                <HFText size={12} weight={500}>{d.label}</HFText>
                <HFMono size={10} color={HF.textMute} style={{ display: 'block' }}>{d.path}</HFMono>
              </div>
            ))}
            <div style={{ height: 1, background: HF.border, margin: '4px 0' }} />
            <div onClick={() => { setDemoOpen(false); set({ wizardOpen: true }); }} style={{ padding: '6px 8px', borderRadius: 5, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6 }}>
              <HFIcon name="plus" size={12} color={HF.accent} />
              <HFText size={12} weight={500} color={HF.accent}>New project · wizard</HFText>
            </div>
          </div>
        )}
      </div>

      <div style={{ flex: 1 }} />

      <div onClick={() => set({ commandOpen: true })} style={{
        display: 'flex', alignItems: 'center', gap: 8,
        background: HF.panelAlt, border: `1px solid ${HF.border}`,
        borderRadius: 6, padding: '4px 10px', width: 420, cursor: 'text',
      }}>
        <HFIcon name="search" size={12} color={HF.textDim} />
        <HFText size={12} color={HF.textDim}>Search runs · specs · params · devices · files</HFText>
        <div style={{ flex: 1 }} />
        <HFMono size={10} color={HF.textDim}>⌘K</HFMono>
      </div>

      <div style={{ flex: 1 }} />

      {s.isRunning && <HFBadge tone="indigo"><span style={{ width: 6, height: 6, borderRadius: 999, background: HF.accent, marginRight: 3 }} />r12 · iter 847</HFBadge>}
      <HFButton size="sm" icon={<HFIcon name="fork" size={11} />}>Fork</HFButton>

      <div style={{ position: 'relative' }}>
        <HFButton size="sm" kind="primary" onClick={() => setRunOpen(v => !v)}
          icon={<HFIcon name={s.isRunning ? 'stop' : 'play'} size={11} color="#fff" />}>
          {s.isRunning ? 'Stop' : 'Run'}  ▾
        </HFButton>
        {runOpen && (
          <div style={{
            position: 'absolute', top: 34, right: 0, width: 320, zIndex: 20,
            background: HF.panel, border: `1px solid ${HF.border}`, borderRadius: 8,
            boxShadow: '0 10px 26px rgba(0,0,0,0.14)', padding: 14,
            display: 'flex', flexDirection: 'column', gap: 10,
          }}>
            <HFText size={13} weight={600}>{s.isRunning ? 'Stop run · r12' : 'Configure run'}</HFText>
            <div>
              <HFText size={11} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase' }}>Algorithm</HFText>
              <select value={s.algo} onChange={(e) => set({ algo: e.target.value })}
                style={{ width: '100%', marginTop: 4, padding: '6px 8px', fontFamily: HF.mono, fontSize: 12, border: `1px solid ${HF.border}`, borderRadius: 6, background: HF.panel, color: HF.text }}>
                {algos.map(a => <option key={a.id} value={a.id}>{a.label} — {a.desc}</option>)}
              </select>
            </div>
            <div>
              <HFText size={11} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase' }}>Budget</HFText>
              <input type="number" value={s.budget} onChange={(e) => set({ budget: +e.target.value })}
                style={{ width: '100%', marginTop: 4, padding: '6px 8px', fontFamily: HF.mono, fontSize: 12, border: `1px solid ${HF.border}`, borderRadius: 6, background: HF.panel, color: HF.text }} />
            </div>
            <div>
              <HFText size={11} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase' }}>Loss shape</HFText>
              <div style={{ display: 'flex', gap: 4, marginTop: 4, background: HF.panelAlt, padding: 3, border: `1px solid ${HF.border}`, borderRadius: 6 }}>
                {['sigmoid', 'linear'].map(sh => (
                  <span key={sh} onClick={() => set({ specShape: sh })} style={{
                    flex: 1, textAlign: 'center', padding: '4px 0', borderRadius: 4, cursor: 'pointer',
                    background: s.specShape === sh ? HF.panel : 'transparent',
                    color: s.specShape === sh ? HF.accent : HF.textMute,
                    fontFamily: HF.ui, fontSize: 12, fontWeight: 600,
                  }}>{sh}</span>
                ))}
              </div>
            </div>
            <div>
              <HFText size={11} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase' }}>Or replay demo checkpoint</HFText>
              <select value={s.replaySource || ''} onChange={(e) => set({ replaySource: e.target.value || null })}
                style={{ width: '100%', marginTop: 4, padding: '6px 8px', fontFamily: HF.mono, fontSize: 12, border: `1px solid ${HF.border}`, borderRadius: 6, background: HF.panel, color: HF.text }}>
                <option value="">— live run —</option>
                {replays.map(r => <option key={r.id} value={r.id}>{r.label}</option>)}
              </select>
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              <HFButton size="sm" style={{ flex: 1 }} onClick={() => setRunOpen(false)}>Cancel</HFButton>
              <HFButton size="sm" kind="primary" style={{ flex: 1 }} onClick={() => { setRunOpen(false); set({ isRunning: !s.isRunning }); }}>
                {s.isRunning ? 'Stop' : 'Start run'}
              </HFButton>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function StudioTabStrip() {
  const { s, set } = useStudio();
  const tabs = [
    { id: 'pipeline',    ic: 'graph',    l: 'Pipeline' },
    { id: 'convergence', ic: 'activity', l: 'Convergence' },
    { id: 'yaml',        ic: 'file',     l: 'YAML setup', dirty: s.yamlDirty },
    { id: 'schematic',   ic: 'chip',     l: 'Schematic' },
    { id: 'shaping',     ic: 'sliders',  l: 'Score shaping' },
    { id: 'compare',     ic: 'diff',     l: 'Compare', badge: '3' },
  ];
  return (
    <div style={{
      height: 36, borderBottom: `1px solid ${HF.border}`, background: HF.panel,
      display: 'flex', alignItems: 'stretch',
    }}>
      {tabs.map(t => {
        const sel = s.activeTab === t.id;
        return (
          <div key={t.id} onClick={() => set({ activeTab: t.id })} style={{
            display: 'flex', alignItems: 'center', gap: 6, padding: '0 14px',
            borderRight: `1px solid ${HF.border}`,
            borderTop: sel ? `2px solid ${HF.accent}` : '2px solid transparent',
            marginTop: sel ? -1 : 0,
            background: sel ? HF.bg : HF.panel, cursor: 'pointer',
          }}>
            <HFIcon name={t.ic} size={12} color={sel ? HF.accent : HF.textMute} />
            <HFText size={12} weight={sel ? 600 : 500} color={sel ? HF.text : HF.textMute}>{t.l}</HFText>
            {t.dirty && <span style={{ color: HF.warn, fontFamily: HF.mono, fontSize: 10 }}>●</span>}
            {t.badge && <HFBadge tone="neutral" style={{ fontSize: 9, padding: '0 5px' }}>{t.badge}</HFBadge>}
          </div>
        );
      })}
      <div style={{ flex: 1 }} />
    </div>
  );
}

function StudioCenter({ width, height }) {
  const { s, set } = useStudio();
  const views = {
    pipeline:    StudioPipelineView,
    convergence: StudioConvergenceView,
    yaml:        StudioYAMLView,
    schematic:   StudioSchematicView,
    shaping:     StudioShapingView,
    compare:     StudioCompareView,
  };
  const View = views[s.activeTab] || StudioPipelineView;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minWidth: 0 }}>
      <StudioTabStrip />
      <div style={{ flex: 1, minHeight: 0, position: 'relative' }}>
        <View width={width} height={height} />
      </div>
      {s.bottomOpen && <StudioBottomPanel height={184} />}
      {!s.bottomOpen && (
        <div onClick={() => set({ bottomOpen: true })} style={{
          height: 24, background: HF.panelInv, color: '#a8a29e', cursor: 'pointer',
          display: 'flex', alignItems: 'center', padding: '0 14px', gap: 8,
        }}>
          <HFIcon name="terminal" size={12} color="#a8a29e" />
          <HFText size={11} color="#a8a29e">show panel · log · terminal · problems · diff</HFText>
        </div>
      )}
    </div>
  );
}

function StudioStatusBar() {
  const { s, specs } = useStudio();
  const failing = specs.filter(sp => !sp.pass).length;
  return (
    <div style={{
      height: 24, background: s.isRunning ? HF.accent : HF.text, color: '#fff',
      display: 'flex', alignItems: 'center', padding: '0 12px', gap: 14, fontSize: 11,
    }}>
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
        <HFIcon name="git" size={11} color="#fff" /> dev/ui
      </span>
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
        <span style={{ width: 8, height: 8, borderRadius: 999, background: '#fff' }} />
        {s.isRunning ? `r12 live · 14 min left` : 'idle'}
      </span>
      <HFMono size={11} color="#fff">F best = 0.412</HFMono>
      <HFMono size={11} color="#fff">{specs.length - failing}/{specs.length} specs passing</HFMono>
      <HFMono size={11} color="#fff">{s.algo} · {s.specShape} · budget {s.budget}</HFMono>
      <div style={{ flex: 1 }} />
      <HFMono size={11} color="#fff">ngspice · 1.41 s / iter</HFMono>
      <HFMono size={11} color="#fff">parallel · 4 workers</HFMono>
    </div>
  );
}

// ─── Wizard overlay ───
function StudioWizard() {
  const { s, set } = useStudio();
  if (!s.wizardOpen) return null;
  const steps = ['Basic info', 'PDK rules', 'DUT params', 'PVT', 'Testbenches', 'Target specs', 'Optimizer'];
  const step = s.wizardStep;

  const stepBody = () => {
    if (step === 0) return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <Field label="Project name" defaultValue="MY_OTA_PROJECT" />
        <Field label="Description" defaultValue="Cascode OTA sizing in ihp-sg13g2" multiline />
        <FieldSelect label="Simulator" options={['ngspice', 'xyce', 'spectre']} />
        <Field label="Workspace root" defaultValue="examples/OTA/cascode/ihp-sg13g2/" />
      </div>
    );
    if (step === 1) return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <Field label="Tech name" defaultValue="ihp-sg13g2" />
        <HFSectionLabel action={<HFButton size="sm" icon={<HFIcon name="plus" size={11} />}>Add constraint</HFButton>}>Named constraints</HFSectionLabel>
        {[
          ['min_nfet_w', '0.18u'], ['max_nfet_w', '200u'],
          ['min_nfet_l', '0.18u'], ['max_nfet_l', '10u'],
          ['min_pfet_w', '0.18u'], ['max_pfet_w', '200u'],
        ].map(([k, v]) => (
          <div key={k} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
            <input defaultValue={k} style={inputStyle} />
            <input defaultValue={v} style={{ ...inputStyle, fontFamily: HF.mono }} />
          </div>
        ))}
      </div>
    );
    if (step === 2) return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <HFCard pad={10} style={{ flex: 1, background: HF.accentSoft, border: `1px dashed ${HF.accentMid}`, textAlign: 'center' }}>
            <HFIcon name="file" size={20} color={HF.accent} /><br />
            <HFText size={12} color={HF.accent} weight={500}>Drop DUT netlist (.spice) here · or browse</HFText>
            <HFMono size={10} color={HF.textMute} style={{ display: 'block' }}>auto-parses .param lines · fills the table below</HFMono>
          </HFCard>
        </div>
        <HFSectionLabel action={<HFButton size="sm" icon={<HFIcon name="plus" size={11} />}>Add row</HFButton>}>Parameters · 13 found</HFSectionLabel>
        <div style={{ border: `1px solid ${HF.border}`, borderRadius: 6, overflow: 'hidden' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1.6fr 1fr 1fr 0.6fr 0.6fr', padding: '6px 10px', borderBottom: `1px solid ${HF.border}`, background: HF.panelAlt }}>
            {['name', 'min', 'max', 'int', 'log'].map(h => <HFMono key={h} size={10} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase' }}>{h}</HFMono>)}
          </div>
          {['X_DUT_M1M2_W','X_DUT_M1M2_L','X_DUT_M5_NG','X_DUT_V_BIAS_1'].map((n, i) => (
            <div key={n} style={{ display: 'grid', gridTemplateColumns: '1.6fr 1fr 1fr 0.6fr 0.6fr', padding: '5px 10px', borderBottom: i < 3 ? `1px dashed ${HF.border}` : 'none', alignItems: 'center' }}>
              <HFMono size={11}>{n}</HFMono>
              <input defaultValue={i < 2 ? 'min_nfet_w' : i === 2 ? '1' : '0'} style={inputStyleSm} />
              <input defaultValue={i < 2 ? 'max_nfet_w' : i === 2 ? '5' : '1.2'} style={inputStyleSm} />
              <input type="checkbox" defaultChecked={i === 2} />
              <input type="checkbox" />
            </div>
          ))}
        </div>
      </div>
    );
    if (step === 5) return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <HFSectionLabel action={<HFButton size="sm" icon={<HFIcon name="plus" size={11} />}>Add spec</HFButton>}>Target specs · 6</HFSectionLabel>
        {['ugf', 'dcgain', 'pm'].map(n => (
          <HFCard key={n} pad={10}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 0.8fr 0.6fr 0.6fr 0.5fr 0.5fr', gap: 6, alignItems: 'center' }}>
              <input defaultValue={n} style={{ ...inputStyleSm, fontWeight: 600 }} />
              <FieldSelect label="" small options={['tb_ac', 'tb_noise', 'tb_tran']} />
              <FieldSelect label="" small options={['exceed', 'minimize', 'exact']} />
              <input defaultValue="200e6" style={inputStyleSm} />
              <input defaultValue="10e6" style={inputStyleSm} />
              <input defaultValue="100" style={inputStyleSm} />
            </div>
          </HFCard>
        ))}
        <HFText size={11} color={HF.textMute}>+ 3 more (inoise · idd · tsettle)</HFText>
      </div>
    );
    if (step === 6) return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <FieldSelect label="Algorithm" options={['LhsDE', 'LogBFGSCMAPlus', 'TinyCMA', 'LHSSearch']} />
        <Field label="Budget" defaultValue="2000" mono />
        <Field label="Random seed" defaultValue="48" mono />
        <FieldSelect label="Default loss shape" options={['sigmoid', 'linear']} />
      </div>
    );
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <HFCard pad={20} style={{ textAlign: 'center', background: HF.panelAlt }}>
          <HFText size={13} color={HF.textMute}>Step body for "{steps[step]}" — fields as per the planned wizard</HFText>
        </HFCard>
      </div>
    );
  };

  return (
    <div onClick={() => set({ wizardOpen: false })} style={{
      position: 'absolute', inset: 0, background: 'rgba(28,25,23,0.45)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100,
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        width: 880, height: 620, background: HF.panel, borderRadius: 12,
        boxShadow: '0 24px 60px rgba(0,0,0,0.25)', display: 'flex', overflow: 'hidden',
      }}>
        {/* Side stepper */}
        <div style={{ width: 220, background: HF.panelAlt, borderRight: `1px solid ${HF.border}`, padding: 18, display: 'flex', flexDirection: 'column', gap: 4 }}>
          <HFText size={13} weight={600}>New project wizard</HFText>
          <HFText size={11} color={HF.textMute} style={{ marginBottom: 10 }}>generates project_setup.yaml</HFText>
          {steps.map((label, i) => (
            <div key={label} onClick={() => set({ wizardStep: i })} style={{
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '6px 8px', borderRadius: 6, cursor: 'pointer',
              background: i === step ? HF.accentSoft : 'transparent',
            }}>
              <span style={{
                width: 20, height: 20, borderRadius: 999,
                background: i < step ? HF.success : i === step ? HF.accent : HF.panel,
                color: i <= step ? '#fff' : HF.textMute,
                border: `1px solid ${i <= step ? 'transparent' : HF.border}`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontFamily: HF.ui, fontSize: 11, fontWeight: 600,
              }}>{i < step ? '✓' : i + 1}</span>
              <HFText size={12} weight={i === step ? 600 : 500} color={i === step ? HF.accent : HF.text}>{label}</HFText>
            </div>
          ))}
        </div>

        {/* Step body */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
          <div style={{ padding: '14px 22px', borderBottom: `1px solid ${HF.border}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <HFText size={15} weight={600}>{step + 1}. {steps[step]}</HFText>
            <span onClick={() => set({ wizardOpen: false })} style={{ cursor: 'pointer' }}>
              <HFIcon name="x" size={14} color={HF.textMute} />
            </span>
          </div>
          <div style={{ flex: 1, padding: 22, overflow: 'auto' }}>
            {stepBody()}
          </div>
          <div style={{ padding: '12px 22px', borderTop: `1px solid ${HF.border}`, display: 'flex', alignItems: 'center', gap: 8, background: HF.panelAlt }}>
            <HFText size={11} color={HF.textMute}>Step {step + 1} of {steps.length}</HFText>
            <div style={{ flex: 1 }} />
            <HFButton size="sm" onClick={() => set({ wizardStep: Math.max(0, step - 1) })}>Back</HFButton>
            {step < steps.length - 1 ? (
              <HFButton size="sm" kind="primary" onClick={() => set({ wizardStep: step + 1 })}>Next →</HFButton>
            ) : (
              <HFButton size="sm" kind="primary" onClick={() => set({ wizardOpen: false, yamlDirty: false, activeActivity: 'setup', activeTab: 'yaml' })}
                icon={<HFIcon name="check" size={11} color="#fff" />}>Generate YAML</HFButton>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

const inputStyle = {
  padding: '6px 8px', fontSize: 12, border: `1px solid ${HF.border}`,
  borderRadius: 5, background: HF.panel, color: HF.text, fontFamily: HF.ui, outline: 'none',
};
const inputStyleSm = { ...inputStyle, padding: '4px 6px', fontSize: 11, fontFamily: HF.mono };

function Field({ label, defaultValue, multiline, mono }) {
  return (
    <div>
      <HFText size={11} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase' }}>{label}</HFText>
      {multiline
        ? <textarea defaultValue={defaultValue} style={{ ...inputStyle, width: '100%', marginTop: 4, minHeight: 50, resize: 'vertical' }} />
        : <input defaultValue={defaultValue} style={{ ...inputStyle, width: '100%', marginTop: 4, fontFamily: mono ? HF.mono : HF.ui }} />}
    </div>
  );
}

function FieldSelect({ label, options, small }) {
  return (
    <div>
      {label && <HFText size={11} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase' }}>{label}</HFText>}
      <select style={{ ...(small ? inputStyleSm : inputStyle), width: '100%', marginTop: label ? 4 : 0 }}>
        {options.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
}

// ─── Command palette ───
function StudioCommandPalette() {
  const { s, set, runs, specs } = useStudio();
  if (!s.commandOpen) return null;
  const groups = [
    { label: 'Switch view', items: [
      { l: 'Pipeline',       sub: 'cmd+1', go: { activeTab: 'pipeline' } },
      { l: 'Convergence',    sub: 'cmd+2', go: { activeTab: 'convergence' } },
      { l: 'YAML setup',     sub: 'cmd+3', go: { activeActivity: 'setup', activeTab: 'yaml' } },
      { l: 'Schematic',      sub: 'cmd+4', go: { activeActivity: 'schematic', activeTab: 'schematic' } },
      { l: 'Score shaping',  sub: 'cmd+5', go: { activeActivity: 'specs', activeTab: 'shaping' } },
      { l: 'Compare A ↔ B',  sub: 'cmd+6', go: { activeActivity: 'compare', activeTab: 'compare' } },
    ]},
    { label: 'Jump to run', items: runs.slice(0, 4).map(r => ({ l: `${r.id.toUpperCase()} · ${r.name}`, sub: r.time, go: { selectedRunId: r.id, activeTab: 'convergence' } })) },
    { label: 'Jump to spec', items: specs.slice(0, 4).map(sp => ({ l: `${sp.name}  ·  ${sp.goal} ${sp.target_str}`, sub: sp.pass ? 'pass' : 'fail', go: { selectedSpec: sp.name, activeTab: 'shaping' } })) },
    { label: 'Actions', items: [
      { l: '+ New project · wizard', go: { wizardOpen: true } },
      { l: 'Validate YAML' },
      { l: 'Apply setup' },
      { l: 'Export YAML · apply best' },
      { l: s.isRunning ? 'Stop run' : 'Start run', go: { isRunning: !s.isRunning } },
    ]},
  ];

  return (
    <div onClick={() => set({ commandOpen: false })} style={{
      position: 'absolute', inset: 0, background: 'rgba(28,25,23,0.35)',
      display: 'flex', alignItems: 'flex-start', justifyContent: 'center', paddingTop: 80, zIndex: 100,
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        width: 620, maxHeight: 560, background: HF.panel, borderRadius: 10,
        boxShadow: '0 24px 60px rgba(0,0,0,0.30)', overflow: 'hidden',
        display: 'flex', flexDirection: 'column',
      }}>
        <div style={{ padding: 14, borderBottom: `1px solid ${HF.border}`, display: 'flex', alignItems: 'center', gap: 10 }}>
          <HFIcon name="search" size={16} color={HF.textMute} />
          <input autoFocus placeholder="Jump to a view · run · spec · device · action…" style={{
            flex: 1, fontFamily: HF.ui, fontSize: 14, border: 'none', outline: 'none', background: 'transparent', color: HF.text,
          }} />
          <HFMono size={10} color={HF.textDim}>esc</HFMono>
        </div>
        <div style={{ flex: 1, overflow: 'auto', padding: 8 }}>
          {groups.map(g => (
            <div key={g.label} style={{ marginBottom: 6 }}>
              <HFText size={10} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase', letterSpacing: 0.4, padding: '6px 8px', display: 'block' }}>{g.label}</HFText>
              {g.items.map((it, i) => (
                <div key={i} onClick={() => { if (it.go) set(it.go); set({ commandOpen: false }); }} style={{
                  padding: '6px 10px', borderRadius: 6, display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer',
                }}
                  onMouseEnter={(e) => e.currentTarget.style.background = HF.accentSoft}
                  onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}>
                  <HFText size={13}>{it.l}</HFText>
                  <div style={{ flex: 1 }} />
                  {it.sub && <HFMono size={10} color={HF.textDim}>{it.sub}</HFMono>}
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Main HFStudio shell ───
function HFStudioInner() {
  const { s, set } = useStudio();
  const W = 1440, H = 900;
  const centerW = W - 52 - 268 - (s.rightOpen ? 320 : 0);
  const centerH = H - 38 - 36 - (s.bottomOpen ? 184 : 24) - 24;
  return (
    <div style={{ width: W, height: H, background: HF.bg, fontFamily: HF.ui, color: HF.text, display: 'flex', flexDirection: 'column', position: 'relative', overflow: 'hidden' }}>
      <StudioTitleBar />
      <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
        <StudioActivityBar />
        <div style={{ width: 268, borderRight: `1px solid ${HF.border}`, background: HF.panel, display: 'flex', flexDirection: 'column' }}>
          <StudioLeftRail />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <StudioCenter width={centerW} height={centerH} />
        </div>
        {s.rightOpen && (
          <div style={{ width: 320, borderLeft: `1px solid ${HF.border}`, background: HF.panel }}>
            <StudioRightRail />
          </div>
        )}
      </div>
      <StudioStatusBar />
      <StudioWizard />
      <StudioCommandPalette />
    </div>
  );
}

function HFStudio() {
  return (
    <StudioProvider>
      <HFStudioInner />
    </StudioProvider>
  );
}

window.HFStudio = HFStudio;
