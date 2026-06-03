// Studio rails — left (depends on activity), right (always-on), bottom (collapsible).

// ─────────────────────────────────────────────────────────────
// Left rail variants — selected by activeActivity
// ─────────────────────────────────────────────────────────────
function StudioRunHistoryRail() {
  const { s, set, runs } = useStudio();
  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: 0, flex: 1 }}>
      <div style={{ padding: '12px 14px 8px', borderBottom: `1px solid ${HF.border}` }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <HFText size={11} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase', letterSpacing: 0.4 }}>Run history</HFText>
          <HFIcon name="plus" size={12} color={HF.textMute} />
        </div>
        <div style={{ marginTop: 8, padding: '4px 8px', border: `1px solid ${HF.border}`, borderRadius: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
          <HFIcon name="search" size={11} color={HF.textDim} />
          <HFText size={11} color={HF.textDim}>filter · tag · seed · loss…</HFText>
        </div>
        <div style={{ display: 'flex', gap: 4, marginTop: 8, flexWrap: 'wrap' }}>
          <HFBadge tone="ink">all</HFBadge>
          <HFBadge tone="neutral">live</HFBadge>
          <HFBadge tone="neutral">sigmoid</HFBadge>
          <HFBadge tone="neutral">linear</HFBadge>
        </div>
      </div>
      <div style={{ flex: 1, overflow: 'hidden', padding: '8px 10px', display: 'flex', flexDirection: 'column', gap: 6 }}>
        {runs.map((r) => {
          const active = r.id === s.selectedRunId;
          return (
            <div key={r.id} onClick={() => set({ selectedRunId: r.id, activeTab: 'convergence' })} style={{
              padding: 9, cursor: 'pointer',
              border: `1px solid ${active ? HF.accentMid : HF.border}`,
              background: active ? HF.accentSoft : HF.panel,
              borderRadius: 7, display: 'flex', flexDirection: 'column', gap: 4,
              boxShadow: active ? `inset 2px 0 0 ${HF.accent}` : 'none',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <HFMono size={10} color={HF.textMute} weight={500}>{r.id.toUpperCase()}</HFMono>
                {r.status === 'live' && <HFBadge tone="indigo" style={{ fontSize: 9, padding: '0 5px' }}>● live</HFBadge>}
                {r.status === 'done' && <HFBadge tone="success" style={{ fontSize: 9, padding: '0 5px' }}>✓ done</HFBadge>}
                {r.status === 'fail' && <HFBadge tone="error" style={{ fontSize: 9, padding: '0 5px' }}>✗ failed</HFBadge>}
              </div>
              <HFText size={12} weight={500}>{r.name}</HFText>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <HFSpark width={104} height={18} color={r.status === 'fail' ? HF.error : HF.accent} kind={r.spark} fill />
                <div style={{ flex: 1 }} />
                {r.score != null && <HFMono size={11} weight={600}>{r.score.toFixed(3)}</HFMono>}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function StudioFileTreeRail() {
  const { s, set } = useStudio();
  const tree = [
    { l: 'spice', ind: 0, ic: 'folder' },
    { l: 'ota-improved.spice', ind: 1, ic: 'file' },
    { l: 'ota-improved_tb-loopgain.spice', ind: 1, ic: 'file' },
    { l: 'ota-improved_tb-noise.spice', ind: 1, ic: 'file' },
    { l: 'ota-improved_tb-tran.spice', ind: 1, ic: 'file' },
    { l: 'sizing', ind: 0, ic: 'folder' },
    { l: 'project_setup.yaml', ind: 1, ic: 'file', sel: true, dirty: s.yamlDirty },
    { l: 'runs', ind: 0, ic: 'folder' },
    { l: 'checkpoints', ind: 0, ic: 'folder' },
  ];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      <div style={{ padding: '10px 12px', borderBottom: `1px solid ${HF.border}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <HFText size={11} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase', letterSpacing: 0.4 }}>Cascode OTA</HFText>
        <HFButton size="sm" kind="ghost" onClick={() => set({ wizardOpen: true })} style={{ padding: '2px 8px' }}
          icon={<HFIcon name="plus" size={11} />}>New project</HFButton>
      </div>
      <div style={{ flex: 1, padding: '6px 6px', display: 'flex', flexDirection: 'column', gap: 1, overflow: 'hidden' }}>
        {tree.map((r, i) => (
          <div key={i} style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '3px 6px', paddingLeft: 6 + r.ind * 14, borderRadius: 4,
            background: r.sel ? HF.accentSoft : 'transparent', cursor: 'pointer',
          }}
            onClick={() => r.l.endsWith('.yaml') && set({ activeTab: 'yaml' })}>
            <HFIcon name={r.ic} size={12} color={r.sel ? HF.accent : HF.textMute} />
            <HFText size={12} weight={r.sel ? 600 : 400} color={r.sel ? HF.accent : HF.text}>{r.l}</HFText>
            <div style={{ flex: 1 }} />
            {r.dirty && <span style={{ color: HF.warn, fontFamily: HF.mono, fontSize: 10 }}>●</span>}
          </div>
        ))}
      </div>
    </div>
  );
}

function StudioPipelineOutlineRail() {
  const { s, set, specs, testbenches } = useStudio();
  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      <div style={{ padding: '12px 14px 8px', borderBottom: `1px solid ${HF.border}` }}>
        <HFText size={11} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase', letterSpacing: 0.4 }}>Graph outline</HFText>
      </div>
      <div style={{ flex: 1, padding: '8px 10px', overflow: 'hidden', display: 'flex', flexDirection: 'column', gap: 2 }}>
        {[
          { l: 'Inputs', ind: 0, n: 3 },
          { l: 'Netlist', ind: 1 }, { l: 'PDK rules', ind: 1 }, { l: 'PVT corner', ind: 1 },
          { l: 'Variables', ind: 0, n: 1 },
          { l: 'DUT parameters · 13', ind: 1 },
          { l: 'Simulation', ind: 0, n: testbenches.length },
          ...testbenches.map(tb => ({ l: tb.name, ind: 1 })),
          { l: 'Scoring', ind: 0, n: specs.length },
          ...specs.map(sp => ({ l: sp.name, ind: 1, key: sp.name, sel: sp.name === s.selectedSpec })),
          { l: 'Optimizer', ind: 0, n: 1 },
          { l: `${s.algo}`, ind: 1 },
        ].map((r, i) => (
          <div key={i} onClick={() => r.key && set({ selectedSpec: r.key })} style={{
            display: 'flex', justifyContent: 'space-between',
            padding: '3px 6px', paddingLeft: 6 + r.ind * 12, borderRadius: 4,
            background: r.sel ? HF.accentSoft : 'transparent',
            cursor: r.key ? 'pointer' : 'default',
          }}>
            <HFText size={12} color={r.sel ? HF.accent : HF.text} weight={r.sel ? 600 : (r.ind === 0 ? 500 : 400)}>{r.l}</HFText>
            {r.n != null && <HFMono size={10} color={HF.textMute}>{r.n}</HFMono>}
          </div>
        ))}
      </div>
    </div>
  );
}

function StudioSpecListRail() {
  const { s, set, specs } = useStudio();
  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      <div style={{ padding: '12px 14px 8px', borderBottom: `1px solid ${HF.border}`, display: 'flex', justifyContent: 'space-between' }}>
        <HFText size={11} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase', letterSpacing: 0.4 }}>Target specs</HFText>
        <HFIcon name="plus" size={12} color={HF.textMute} />
      </div>
      <div style={{ flex: 1, padding: '8px 10px', overflow: 'hidden', display: 'flex', flexDirection: 'column', gap: 5 }}>
        {specs.map(sp => {
          const sel = sp.name === s.selectedSpec;
          return (
            <div key={sp.name} onClick={() => set({ selectedSpec: sp.name, specTryValue: sp.target, activeTab: 'shaping' })} style={{
              padding: 8, cursor: 'pointer',
              border: `1px solid ${sel ? HF.accentMid : HF.border}`,
              background: sel ? HF.accentSoft : HF.panel,
              borderRadius: 7, display: 'flex', flexDirection: 'column', gap: 3,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ width: 6, height: 6, borderRadius: 999, background: sp.pass ? HF.success : HF.error }} />
                  <HFMono size={11} weight={600}>{sp.name}</HFMono>
                </span>
                <HFMono size={10} color={HF.textMute}>w {sp.weight}</HFMono>
              </div>
              <HFMono size={10} color={HF.textMute}>{sp.goal} {sp.target_str}</HFMono>
              <HFMono size={10} color={sp.pass ? HF.success : HF.error} weight={500}>{sp.value}</HFMono>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function StudioCompareSetupRail() {
  const { s, set, runs } = useStudio();
  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      <div style={{ padding: '12px 14px 8px', borderBottom: `1px solid ${HF.border}` }}>
        <HFText size={11} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase', letterSpacing: 0.4 }}>Compare</HFText>
      </div>
      <div style={{ padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 12 }}>
        {[['Run A', 'compareRunA', HF.accent], ['Run B', 'compareRunB', '#0ea5e9']].map(([label, key, color]) => (
          <div key={key}>
            <HFText size={11} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase', letterSpacing: 0.4 }}>{label}</HFText>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 6 }}>
              {runs.slice(0, 4).map(r => {
                const sel = s[key] === r.id;
                return (
                  <div key={r.id} onClick={() => set({ [key]: r.id })} style={{
                    padding: '5px 8px', cursor: 'pointer',
                    border: `1px solid ${sel ? color : HF.border}`,
                    background: sel ? `${color}14` : HF.panel,
                    borderRadius: 6, display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  }}>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                      <HFMono size={10} color={HF.textMute}>{r.id.toUpperCase()}</HFMono>
                      <HFText size={12} weight={sel ? 600 : 400}>{r.name}</HFText>
                    </span>
                    {r.score != null && <HFMono size={11} color={color} weight={600}>{r.score.toFixed(2)}</HFMono>}
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function StudioSchematicListRail() {
  const { s, set } = useStudio();
  const groups = [
    { name: 'M1M2',    desc: 'Input pair · NMOS' },
    { name: 'M1CM2C',  desc: 'NMOS cascode' },
    { name: 'M3CM4C',  desc: 'PMOS cascode' },
    { name: 'M3M4',    desc: 'PMOS load' },
    { name: 'M5',      desc: 'Tail · NG=3' },
    { name: 'M6',      desc: 'Bias · NMOS' },
  ];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      <div style={{ padding: '12px 14px 8px', borderBottom: `1px solid ${HF.border}` }}>
        <HFText size={11} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase', letterSpacing: 0.4 }}>Device groups</HFText>
      </div>
      <div style={{ flex: 1, padding: '8px 10px', overflow: 'hidden', display: 'flex', flexDirection: 'column', gap: 5 }}>
        {groups.map(g => {
          const sel = s.selectedDevice === g.name;
          return (
            <div key={g.name} onClick={() => set({ selectedDevice: g.name })} style={{
              padding: 8, cursor: 'pointer',
              border: `1px solid ${sel ? HF.accentMid : HF.border}`,
              background: sel ? HF.accentSoft : HF.panel,
              borderRadius: 7,
            }}>
              <HFMono size={11} weight={600}>{g.name}</HFMono>
              <HFMono size={10} color={HF.textMute} style={{ display: 'block' }}>{g.desc}</HFMono>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function StudioLeftRail() {
  const { s } = useStudio();
  const map = {
    runs: <StudioRunHistoryRail />,
    setup: <StudioFileTreeRail />,
    pipeline: <StudioPipelineOutlineRail />,
    schematic: <StudioSchematicListRail />,
    specs: <StudioSpecListRail />,
    compare: <StudioCompareSetupRail />,
  };
  return map[s.activeActivity] || <StudioRunHistoryRail />;
}

// ─────────────────────────────────────────────────────────────
// Right rail — ALWAYS-ON live spec status + run progress + best params
// ─────────────────────────────────────────────────────────────
function StudioRightRail() {
  const { s, set, specs, params, runs } = useStudio();
  const run = runs.find(r => r.id === s.selectedRunId) || runs[0];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: 0, height: '100%' }}>
      {/* Current run header */}
      <div style={{ padding: '12px 14px', borderBottom: `1px solid ${HF.border}`, display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <HFText size={11} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase', letterSpacing: 0.4 }}>
              {run.id.toUpperCase()} · {run.status}
            </HFText>
            <HFText size={13} weight={600}>{run.name}</HFText>
          </div>
          {s.isRunning ? (
            <HFButton size="sm" onClick={() => set({ isRunning: false })} icon={<HFIcon name="stop" size={11} />}>Stop</HFButton>
          ) : (
            <HFButton size="sm" kind="primary" onClick={() => set({ isRunning: true })} icon={<HFIcon name="play" size={11} color="#fff" />}>Run</HFButton>
          )}
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
          <HFText size={11} color={HF.textMute}>iter {run.iter} / {run.budget} · ETA 14 min</HFText>
          <HFMono size={11} weight={600} color={HF.accent}>F = {run.score?.toFixed(3) ?? '—'}</HFMono>
        </div>
        <div style={{ height: 5, background: HF.panelAlt, borderRadius: 999, overflow: 'hidden' }}>
          <div style={{ width: `${(run.iter / run.budget) * 100}%`, height: '100%', background: HF.accent }} />
        </div>
      </div>

      {/* Live specs */}
      <div style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 10 }}>
        <HFSectionLabel action={
          <HFBadge tone={specs.filter(sp => !sp.pass).length > 0 ? 'warn' : 'success'}>
            {specs.filter(sp => !sp.pass).length} failing
          </HFBadge>
        }>Specs · live</HFSectionLabel>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
          {specs.map(sp => (
            <div key={sp.name} onClick={() => set({ selectedSpec: sp.name, specTryValue: sp.target, activeTab: 'shaping' })} style={{ cursor: 'pointer' }}>
              <HFSpecRow name={sp.name} value={sp.value} target={sp.goal === 'exceed' ? `≥ ${sp.target_str}` : sp.goal === 'minimize' ? `≤ ${sp.target_str}` : `= ${sp.target_str}`} pass={sp.pass} progress={sp.progress} />
            </div>
          ))}
        </div>
      </div>

      <div style={{ height: 1, background: HF.border }} />

      {/* Best params */}
      <div style={{ padding: 14, flex: 1, minHeight: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column', gap: 8 }}>
        <HFSectionLabel action={<HFMono size={10} color={HF.textMute}>iter 832</HFMono>}>Best params</HFSectionLabel>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 3, overflow: 'hidden' }}>
          {params.slice(0, 6).map(p => (
            <div key={p.name} style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0', borderBottom: `1px dashed ${HF.border}` }}>
              <HFMono size={11} color={HF.textMute}>{p.name.replace('X_DUT_', '').replace(/_/g, ' ')}</HFMono>
              <HFMono size={11} weight={500}>{p.best}</HFMono>
            </div>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div style={{ padding: '12px 14px', borderTop: `1px solid ${HF.border}`, display: 'flex', flexDirection: 'column', gap: 6 }}>
        <HFButton kind="primary" icon={<HFIcon name="pin" size={12} color="#fff" />}>Pin as candidate</HFButton>
        <HFButton icon={<HFIcon name="file" size={12} />}>Export YAML · apply best</HFButton>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Bottom panel
// ─────────────────────────────────────────────────────────────
function StudioBottomPanel({ height = 180 }) {
  const { s, set, specs } = useStudio();
  const tabs = [
    { id: 'terminal', l: 'Terminal' },
    { id: 'log',      l: 'Optimizer log', badge: '847' },
    { id: 'problems', l: 'Problems',      badge: specs.filter(sp => !sp.pass).length },
    { id: 'diff',     l: 'Diff · r12 ↔ r11' },
  ];

  return (
    <div style={{ height, background: HF.panelInv, color: HF.inkInv, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ display: 'flex', height: 30, padding: '0 12px', borderBottom: `1px solid ${HF.borderInv}`, alignItems: 'stretch' }}>
        {tabs.map(t => (
          <div key={t.id} onClick={() => set({ bottomTab: t.id })} style={{
            display: 'flex', alignItems: 'center', gap: 6, padding: '0 12px',
            borderBottom: s.bottomTab === t.id ? `2px solid ${HF.accent}` : '2px solid transparent',
            cursor: 'pointer',
          }}>
            <HFText size={12} weight={s.bottomTab === t.id ? 600 : 500} color={s.bottomTab === t.id ? '#fafaf9' : '#a8a29e'}>{t.l}</HFText>
            {t.badge && <HFBadge tone="indigo" style={{ fontSize: 9, padding: '0 5px' }}>{t.badge}</HFBadge>}
          </div>
        ))}
        <div style={{ flex: 1 }} />
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#a8a29e', padding: '0 8px' }}>
          <HFText size={11} color="#a8a29e">streaming · 1 ev / 1.4 s</HFText>
          <span onClick={() => set({ bottomOpen: false })} style={{ cursor: 'pointer' }}>
            <HFIcon name="x" size={12} color="#a8a29e" />
          </span>
        </div>
      </div>

      {s.bottomTab === 'log' && (
        <div style={{ flex: 1, padding: '8px 14px', overflow: 'hidden', display: 'flex', flexDirection: 'column', gap: 3 }}>
          {[
            ['845', '0.408', 'ugf=210M  dcgain=45.9dB  pm=51°  inoise=1.01m  idd=32µ  tsettle=12.8µ'],
            ['846', '0.410', 'ugf=212M  dcgain=46.0dB  pm=51°  inoise=0.99m  idd=31µ  tsettle=12.6µ'],
            ['847', '0.412', 'ugf=214M  dcgain=46.2dB  pm=52°  inoise=0.98m  idd=31µ  tsettle=12.4µ', 'new best'],
            [null, null, '↳ pm and idd failing · sensitivity peaks on M5.NG and M1M2.W → fork branch?', 'hint'],
          ].map((row, i) => (
            <div key={i} style={{ display: 'flex', gap: 14, alignItems: 'baseline' }}>
              <HFMono size={11} color="#78716c" style={{ width: 28 }}>{row[0] ?? ''}</HFMono>
              {row[1] ? <HFMono size={11} color={row[3] === 'new best' ? HF.accent : '#fafaf9'} weight={500} style={{ width: 64 }}>F={row[1]}</HFMono> : <span style={{ width: 64 }} />}
              <HFMono size={11} color={row[3] === 'hint' ? '#a8a29e' : '#e7e5e4'}>{row[2]}</HFMono>
              {row[3] && row[3] !== 'hint' && <span style={{ marginLeft: 'auto', color: HF.accent, fontSize: 10, fontFamily: HF.mono }}>← {row[3]}</span>}
            </div>
          ))}
        </div>
      )}

      {s.bottomTab === 'terminal' && (
        <div style={{ flex: 1, padding: '10px 14px', overflow: 'hidden', display: 'flex', flexDirection: 'column', gap: 4, fontFamily: HF.mono, fontSize: 11 }}>
          <div style={{ color: '#a8a29e' }}>$ uv run python -c "import spicexplorer; print(spicexplorer.__version__)"</div>
          <div style={{ color: '#e7e5e4' }}>0.4.2-dev</div>
          <div style={{ color: '#a8a29e' }}>$ spx run --algo {s.algo} --budget {s.budget} --loss {s.specShape}</div>
          <div style={{ color: '#e7e5e4' }}>
            <span style={{ color: HF.success }}>✓</span> loaded project_setup.yaml<br />
            <span style={{ color: HF.success }}>✓</span> 3 testbenches compiled<br />
            <span style={{ color: HF.accent }}>▶</span> optimization started · seed 48 · pid 14237
          </div>
          <div style={{ color: HF.accent }}>$ <span style={{ background: HF.accent, color: '#fff', marginLeft: 2 }}>&nbsp;</span></div>
        </div>
      )}

      {s.bottomTab === 'problems' && (
        <div style={{ flex: 1, padding: '10px 14px', overflow: 'hidden', display: 'flex', flexDirection: 'column', gap: 6 }}>
          {specs.filter(sp => !sp.pass).map(sp => (
            <div key={sp.name} style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
              <HFIcon name="x" size={12} color={HF.error} />
              <HFMono size={11} color="#fafaf9" weight={500}>{sp.name}</HFMono>
              <HFMono size={11} color="#a8a29e">{sp.value} · target {sp.goal === 'exceed' ? '≥' : sp.goal === 'minimize' ? '≤' : '='} {sp.target_str}</HFMono>
              <HFMono size={11} color={HF.error}>P̂ {sp.P_sig.toFixed(2)}</HFMono>
            </div>
          ))}
        </div>
      )}

      {s.bottomTab === 'diff' && (
        <div style={{ flex: 1, padding: '8px 14px', overflow: 'hidden', display: 'flex', flexDirection: 'column', gap: 3, fontFamily: HF.mono, fontSize: 11 }}>
          <div style={{ color: '#a8a29e' }}>diff project_setup · r11 → r12</div>
          <div style={{ color: HF.error }}>-     name: LogBFGSCMAPlus</div>
          <div style={{ color: HF.success }}>+     name: LhsDE</div>
          <div style={{ color: HF.error }}>-         error_type: relative-absolute  # for all specs</div>
          <div style={{ color: HF.success }}>+         error_type: sigmoid           # for all specs</div>
          <div style={{ color: '#a8a29e' }}>~     budget: 2000 (unchanged)</div>
        </div>
      )}
    </div>
  );
}

Object.assign(window, {
  StudioLeftRail, StudioRightRail, StudioBottomPanel,
  StudioRunHistoryRail, StudioFileTreeRail, StudioPipelineOutlineRail,
  StudioSchematicListRail, StudioSpecListRail, StudioCompareSetupRail,
});
