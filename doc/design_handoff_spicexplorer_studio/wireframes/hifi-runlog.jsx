// Hi-fi A — Run Log workspace.

function HFRunLog() {
  const W = 1440, H = 900;

  const runs = [
    { id: 'r12', name: 'sigmoid · DE · seed 48',  status: 'live', score: 0.412, dScore: '+0.09', spark: 'convFast', algo: 'LhsDE',  loss: 'sigmoid', iter: '847/2000' },
    { id: 'r11', name: 'linear · DE · seed 48',   status: 'done', score: 0.183, dScore: null,    spark: 'conv',     algo: 'LhsDE',  loss: 'linear',  iter: '2000/2000' },
    { id: 'r10', name: 'CMA · wider M5 W',         status: 'done', score: 0.318, dScore: '−0.09', spark: 'conv',     algo: 'CMA',    loss: 'sigmoid', iter: '2000/2000' },
    { id: 'r09', name: 'blind LHS baseline',       status: 'done', score: 0.054, dScore: null,    spark: 'flat',     algo: 'LHS',    loss: 'sigmoid', iter: '500/500'  },
    { id: 'r08', name: 'PM=60° hard constraint',   status: 'fail', score: null,  dScore: null,    spark: 'noisy',    algo: 'LhsDE',  loss: 'sigmoid', iter: '120/2000' },
    { id: 'r07', name: 'first cascode trial',      status: 'done', score: 0.091, dScore: null,    spark: 'flat',     algo: 'LBFGSB', loss: 'linear',  iter: '600/600'  },
  ];

  const specs = [
    { name: 'ugf',     value: '214 MHz',    target: '≥ 200 MHz',  pass: true,  progress: 0.92 },
    { name: 'dcgain',  value: '46.2 dB',    target: '≥ 40 dB',    pass: true,  progress: 0.88 },
    { name: 'pm',      value: '52°',        target: '60° ± 10°',  pass: false, progress: 0.42 },
    { name: 'inoise',  value: '0.98 mV/√Hz',target: '≤ 1.2 mV',   pass: true,  progress: 0.78 },
    { name: 'idd',     value: '31 µA',      target: '≤ 25 µA',    pass: false, progress: 0.35 },
    { name: 'tsettle', value: '12.4 µs',    target: '≤ 15 µs',    pass: true,  progress: 0.83 },
  ];

  return (
    <div style={{ width: W, height: H, background: HF.bg, fontFamily: HF.ui, color: HF.text, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Top bar */}
      <div style={{
        height: 56, borderBottom: `1px solid ${HF.border}`, background: HF.panel,
        display: 'flex', alignItems: 'center', padding: '0 20px', gap: 14,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 24, height: 24, borderRadius: 6, background: `linear-gradient(135deg, ${HF.accent}, ${HF.accentHov})`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <HFIcon name="activity" size={14} color="#fff" />
          </div>
          <HFText size={15} weight={600}>SpiceXplorer</HFText>
        </div>
        <HFText size={13} color={HF.textDim}>/</HFText>
        <HFText size={13} weight={500}>OTA Cascode</HFText>
        <HFBadge tone="neutral">ihp-sg13g2</HFBadge>
        <div style={{ flex: 1 }} />
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '5px 10px', border: `1px solid ${HF.border}`, borderRadius: 6, background: HF.panel, width: 280 }}>
          <HFIcon name="search" size={14} color={HF.textDim} />
          <HFText size={12} color={HF.textDim}>Search runs, specs, params…</HFText>
          <div style={{ flex: 1 }} />
          <HFMono size={10} color={HF.textDim}>⌘K</HFMono>
        </div>
        <HFButton kind="default" icon={<HFIcon name="file" size={13} />}>Setup</HFButton>
        <HFButton kind="default" icon={<HFIcon name="diff" size={13} />}>Compare</HFButton>
        <HFButton kind="primary" icon={<HFIcon name="plus" size={13} color="#fff" />}>New run</HFButton>
      </div>

      <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
        {/* === Left rail: runs === */}
        <div style={{
          width: 296, borderRight: `1px solid ${HF.border}`, background: HF.panel,
          display: 'flex', flexDirection: 'column', minHeight: 0,
        }}>
          <div style={{ padding: '14px 16px 10px', display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <HFText size={13} weight={600}>Runs</HFText>
              <HFText size={11} color={HF.textMute}>24 total · 6 shown</HFText>
            </div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              <HFBadge tone="ink">all</HFBadge>
              <HFBadge tone="neutral">live</HFBadge>
              <HFBadge tone="neutral">sigmoid</HFBadge>
              <HFBadge tone="neutral">linear</HFBadge>
              <HFBadge tone="neutral">+3</HFBadge>
            </div>
          </div>
          <div style={{ flex: 1, overflow: 'hidden', padding: '0 10px 10px', display: 'flex', flexDirection: 'column', gap: 6 }}>
            {runs.map((r, i) => {
              const active = i === 0;
              return (
                <div key={r.id} style={{
                  padding: 10,
                  border: `1px solid ${active ? HF.accentMid : HF.border}`,
                  background: active ? HF.accentSoft : HF.panel,
                  borderRadius: 8, display: 'flex', flexDirection: 'column', gap: 6,
                  boxShadow: active ? `inset 2px 0 0 ${HF.accent}` : 'none',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <HFMono size={10} color={HF.textMute} weight={500}>{r.id.toUpperCase()}</HFMono>
                    {r.status === 'live' && <HFBadge tone="indigo"><span style={{ width: 6, height: 6, borderRadius: 999, background: HF.accent, marginRight: 3 }} />live · iter {r.iter.split('/')[0]}</HFBadge>}
                    {r.status === 'done' && <HFBadge tone="success"><HFIcon name="check" size={10} />done</HFBadge>}
                    {r.status === 'fail' && <HFBadge tone="error"><HFIcon name="x" size={10} />failed</HFBadge>}
                  </div>
                  <HFText size={13} weight={500}>{r.name}</HFText>
                  <div style={{ display: 'flex', gap: 4 }}>
                    <HFBadge tone="neutral" style={{ fontSize: 10 }}>{r.algo}</HFBadge>
                    <HFBadge tone="neutral" style={{ fontSize: 10 }}>{r.loss}</HFBadge>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <HFSpark width={120} height={20} color={r.status === 'fail' ? HF.error : HF.accent} kind={r.spark} fill />
                    <div style={{ flex: 1 }} />
                    {r.score != null && <HFMono size={12} weight={600}>{r.score.toFixed(3)}</HFMono>}
                    {r.dScore && (
                      <HFMono size={10} color={r.dScore.startsWith('+') ? HF.success : HF.error}>{r.dScore}</HFMono>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* === Center: current run === */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0, padding: 20, gap: 14 }}>
          {/* Run header */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <HFMono size={11} color={HF.textMute} weight={500}>R12</HFMono>
            <HFText size={18} weight={600}>sigmoid · DE · seed 48</HFText>
            <HFBadge tone="indigo"><span style={{ width: 6, height: 6, borderRadius: 999, background: HF.accent, marginRight: 3 }} />running</HFBadge>
            <HFBadge tone="neutral">LhsDE</HFBadge>
            <HFBadge tone="neutral">sigmoid · log reward</HFBadge>
            <div style={{ flex: 1 }} />
            <HFText size={12} color={HF.textMute}>Started 14:22  ·  ETA 14 min</HFText>
            <HFButton size="sm" icon={<HFIcon name="stop" size={11} />}>Stop</HFButton>
            <HFButton size="sm" icon={<HFIcon name="fork" size={11} />}>Fork</HFButton>
          </div>

          {/* Tabs */}
          <div style={{ display: 'flex', gap: 4, borderBottom: `1px solid ${HF.border}`, marginTop: -4 }}>
            {[
              ['Overview', true],
              ['Convergence', false],
              ['Specs', false],
              ['Params', false],
              ['Score shaping', false],
              ['Diff vs r11', false, <HFBadge tone="neutral" style={{ fontSize: 10, padding: '0 5px' }}>3</HFBadge>],
            ].map(([label, sel, extra], i) => (
              <div key={label} style={{
                padding: '8px 12px',
                borderBottom: sel ? `2px solid ${HF.accent}` : '2px solid transparent',
                marginBottom: -1,
                display: 'inline-flex', alignItems: 'center', gap: 6,
              }}>
                <HFText size={13} weight={sel ? 600 : 500} color={sel ? HF.text : HF.textMute}>{label}</HFText>
                {extra}
              </div>
            ))}
          </div>

          {/* KPI strip */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
            {[
              { label: 'Best score', value: '0.412', sub: '+0.09 vs r11', tone: HF.success, spark: 'convFast' },
              { label: 'Iteration', value: '847', sub: 'of 2 000  ·  42%', tone: HF.text, bar: 0.42 },
              { label: 'Specs passing', value: '4 / 6', sub: 'pm, idd failing', tone: HF.warn },
              { label: 'Sim time / iter', value: '1.41 s', sub: 'p95 1.62 s', tone: HF.text, spark: 'noisy' },
            ].map((k) => (
              <HFCard key={k.label} pad={12}>
                <HFText size={11} color={HF.textMute} weight={500} style={{ textTransform: 'uppercase', letterSpacing: 0.4 }}>{k.label}</HFText>
                <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginTop: 4 }}>
                  <HFText size={22} weight={600} color={k.tone}>{k.value}</HFText>
                  {k.spark && <HFSpark width={64} height={20} color={k.tone} kind={k.spark} />}
                </div>
                <HFText size={11} color={HF.textMute} style={{ marginTop: 2 }}>{k.sub}</HFText>
                {k.bar != null && (
                  <div style={{ marginTop: 6, height: 4, background: HF.panelAlt, borderRadius: 999, overflow: 'hidden' }}>
                    <div style={{ width: `${k.bar * 100}%`, height: '100%', background: HF.accent }} />
                  </div>
                )}
              </HFCard>
            ))}
          </div>

          {/* Charts */}
          <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 12, flex: 1, minHeight: 0 }}>
            <HFCard pad={10} style={{ display: 'flex', flexDirection: 'column' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '2px 4px 8px' }}>
                <HFText size={13} weight={600}>Score convergence</HFText>
                <div style={{ display: 'flex', gap: 8 }}>
                  <HFBadge tone="neutral">log scale</HFBadge>
                  <HFBadge tone="neutral">overlay r11</HFBadge>
                </div>
              </div>
              <div style={{ flex: 1, minHeight: 0 }}>
                <HFChart width={580} height={300}
                  lines={[
                    { kind: 'raw', color: HF.textDim, opacity: 0.6, label: 'r12 · raw',     seed: 0 },
                    { kind: 'conv', color: HF.accent, width: 2,        label: 'r12 · best',    seed: 0 },
                    { kind: 'conv', color: '#0ea5e9', dash: true,       label: 'r11 · best',    seed: 1.5 },
                  ]}
                  xLabel="iteration" yLabel="score F(x)" />
              </div>
            </HFCard>
            <HFCard pad={10} style={{ display: 'flex', flexDirection: 'column' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '2px 4px 8px' }}>
                <HFText size={13} weight={600}>Metric: ugf</HFText>
                <HFBadge tone="success"><HFIcon name="check" size={10} />214 MHz</HFBadge>
              </div>
              <div style={{ flex: 1, minHeight: 0 }}>
                <HFChart width={400} height={300}
                  lines={[
                    { kind: 'conv', color: HF.success, width: 2, label: 'best ugf', seed: 0.2 },
                  ]}
                  target xLabel="iteration" yLabel="ugf (MHz)" />
              </div>
            </HFCard>
          </div>

          {/* Live log strip */}
          <HFCard pad={10} style={{ background: HF.panelAlt, display: 'flex', alignItems: 'center', gap: 12 }}>
            <HFBadge tone="indigo">iter 847</HFBadge>
            <HFMono size={11} color={HF.text}>
              ugf=<span style={{ color: HF.success }}>214M</span>{'  '}
              dcgain=<span style={{ color: HF.success }}>46.2dB</span>{'  '}
              pm=<span style={{ color: HF.error }}>52°</span>{'  '}
              inoise=<span style={{ color: HF.success }}>0.98m</span>{'  '}
              idd=<span style={{ color: HF.error }}>31µA</span>{'  '}
              tsettle=<span style={{ color: HF.success }}>12.4µs</span>{'   '}
              <span style={{ color: HF.textMute }}>·</span>{'  '}
              F=<span style={{ color: HF.accent, fontWeight: 600 }}>0.412 ↑</span>
            </HFMono>
            <div style={{ flex: 1 }} />
            <HFText size={11} color={HF.textMute}>streaming · 1 event / 1.4s</HFText>
          </HFCard>
        </div>

        {/* === Right rail === */}
        <div style={{
          width: 320, borderLeft: `1px solid ${HF.border}`, background: HF.panel,
          padding: 16, display: 'flex', flexDirection: 'column', gap: 14, minHeight: 0, overflow: 'hidden',
        }}>
          <HFSectionLabel action={<HFBadge tone="warn">2 failing</HFBadge>}>Spec status · live</HFSectionLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {specs.map((s) => <HFSpecRow key={s.name} {...s} />)}
          </div>

          <div style={{ height: 1, background: HF.border, margin: '4px -16px' }} />

          <HFSectionLabel action={<HFButton size="sm" kind="ghost" style={{ padding: '2px 6px' }}>copy</HFButton>}>Best params  ·  iter 832</HFSectionLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {[
              ['M1M2 · W / L',   '24.2µ / 0.42µ'],
              ['M1cM2c · W / L', '18.0µ / 0.36µ'],
              ['M3M4 · W / L',   '46.8µ / 0.55µ'],
              ['M3cM4c · W / L', '38.4µ / 0.50µ'],
              ['M5 · W / L · NG','12.0µ / 0.30µ · 3'],
              ['M6 · W / L',     '6.4µ / 0.40µ'],
              ['V_BIAS_1',       '0.62 V'],
              ['V_BIAS_2',       '0.81 V'],
            ].map(([k, v]) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0', borderBottom: `1px dashed ${HF.border}` }}>
                <HFMono size={11} color={HF.textMute}>{k}</HFMono>
                <HFMono size={11} weight={500}>{v}</HFMono>
              </div>
            ))}
          </div>

          <div style={{ flex: 1 }} />

          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <HFButton kind="primary" icon={<HFIcon name="pin" size={12} color="#fff" />}>Pin as candidate</HFButton>
            <HFButton icon={<HFIcon name="file" size={12} />}>Export YAML  ·  apply best</HFButton>
          </div>
        </div>
      </div>
    </div>
  );
}

window.HFRunLog = HFRunLog;
