// Remix — "SpiceXplorer Studio".
// IDE shell (E) + run history as the primary sidebar (A) + pipeline graph as a first-class tab (D),
// plus always-visible live spec status. One workspace, every pattern reachable in one click.

function HFRemix() {
  const W = 1440, H = 900;
  const titleH = 36;
  const tabH = 36;
  const statusH = 24;
  const activityW = 52;
  const leftW = 268;
  const rightW = 320;
  const bottomH = 184;

  // Compact node for pipeline preview
  const PNode = ({ x, y, w = 168, h = 56, kind, title, status, accent = false, selected = false, sub }) => (
    <div style={{
      position: 'absolute', left: x, top: y, width: w, minHeight: h,
      background: HF.panel,
      border: `1px solid ${selected ? HF.accent : accent ? HF.accentMid : HF.border}`,
      borderRadius: 7,
      boxShadow: selected ? `0 0 0 2px ${HF.accentSoft}` : '0 1px 2px rgba(28,25,23,0.04)',
      padding: 8, display: 'flex', flexDirection: 'column', gap: 2,
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
    return (
      <path d={`M ${x1} ${y1} C ${x1 + dx} ${y1} ${x2 - dx} ${y2} ${x2} ${y2}`}
        fill="none" stroke={color} strokeWidth={1.4}
        strokeDasharray={dashed ? '4 3' : 'none'} />
    );
  };

  return (
    <div style={{ width: W, height: H, background: HF.bg, fontFamily: HF.ui, color: HF.text, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* === Title bar === */}
      <div style={{
        height: titleH, background: HF.panel, borderBottom: `1px solid ${HF.border}`,
        display: 'flex', alignItems: 'center', padding: '0 14px', gap: 14,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <div style={{ width: 18, height: 18, borderRadius: 5, background: `linear-gradient(135deg, ${HF.accent}, ${HF.accentHov})`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <HFIcon name="activity" size={11} color="#fff" />
          </div>
          <HFText size={13} weight={600}>SpiceXplorer Studio</HFText>
        </div>
        <HFText size={12} color={HF.textDim}>·</HFText>
        <HFText size={12}>OTA Cascode  ·  ihp-sg13g2</HFText>
        <div style={{ flex: 1 }} />
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          background: HF.panelAlt, border: `1px solid ${HF.border}`,
          borderRadius: 6, padding: '3px 10px', width: 420,
        }}>
          <HFIcon name="search" size={12} color={HF.textDim} />
          <HFText size={12} color={HF.textDim}>Search runs · specs · params · devices · files</HFText>
          <div style={{ flex: 1 }} />
          <HFMono size={10} color={HF.textDim}>⌘K</HFMono>
        </div>
        <div style={{ flex: 1 }} />
        <HFBadge tone="indigo"><span style={{ width: 6, height: 6, borderRadius: 999, background: HF.accent, marginRight: 3 }} />r12 live · iter 847</HFBadge>
        <HFButton size="sm" icon={<HFIcon name="fork" size={11} />}>Fork</HFButton>
        <HFButton kind="primary" size="sm" icon={<HFIcon name="play" size={11} color="#fff" />}>Run</HFButton>
      </div>

      <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
        {/* === Activity bar === */}
        <div style={{
          width: activityW, borderRight: `1px solid ${HF.border}`, background: HF.panel,
          display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '10px 0', gap: 4,
        }}>
          {[
            ['activity', true,  'Runs', 12],
            ['graph',    false, 'Pipeline'],
            ['chip',     false, 'Schematic'],
            ['file',     false, 'Files'],
            ['sliders',  false, 'Specs'],
            ['diff',     false, 'Compare'],
          ].map(([icon, sel, l, badge], i) => (
            <div key={i} style={{
              position: 'relative',
              width: 36, height: 36, borderRadius: 7,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: sel ? HF.accentSoft : 'transparent',
              borderLeft: sel ? `2px solid ${HF.accent}` : '2px solid transparent',
              marginLeft: -2,
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
          ))}
          <div style={{ flex: 1 }} />
          <HFIcon name="git" size={18} color={HF.textMute} />
          <HFIcon name="settings" size={18} color={HF.textMute} />
        </div>

        {/* === Left rail: Run history === */}
        <div style={{ width: leftW, borderRight: `1px solid ${HF.border}`, background: HF.panel, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          <div style={{ padding: '12px 14px 8px', borderBottom: `1px solid ${HF.border}` }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <HFText size={11} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase', letterSpacing: 0.4 }}>Run history</HFText>
              <HFIcon name="plus" size={12} color={HF.textMute} />
            </div>
            <div style={{ marginTop: 8, padding: '4px 8px', border: `1px solid ${HF.border}`, borderRadius: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
              <HFIcon name="search" size={11} color={HF.textDim} />
              <HFText size={11} color={HF.textDim}>filter · tag · seed · loss…</HFText>
            </div>
          </div>
          <div style={{ flex: 1, overflow: 'hidden', padding: '8px 10px', display: 'flex', flexDirection: 'column', gap: 6 }}>
            {[
              { id: 'r12', name: 'sigmoid · DE',     status: 'live', score: 0.412, dScore: '+0.09', spark: 'convFast' },
              { id: 'r11', name: 'linear · DE',      status: 'done', score: 0.183, dScore: null,    spark: 'conv' },
              { id: 'r10', name: 'CMA · wider M5',   status: 'done', score: 0.318, dScore: '−0.09', spark: 'conv' },
              { id: 'r09', name: 'blind LHS',        status: 'done', score: 0.054, dScore: null,    spark: 'flat' },
              { id: 'r08', name: 'PM=60° hard',      status: 'fail', score: null,  dScore: null,    spark: 'noisy' },
              { id: 'r07', name: 'first cascode',    status: 'done', score: 0.091, dScore: null,    spark: 'flat' },
            ].map((r, i) => {
              const active = i === 0;
              return (
                <div key={r.id} style={{
                  padding: 9,
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
                    <HFSpark width={110} height={18} color={r.status === 'fail' ? HF.error : HF.accent} kind={r.spark} fill />
                    <div style={{ flex: 1 }} />
                    {r.score != null && <HFMono size={11} weight={600}>{r.score.toFixed(3)}</HFMono>}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* === Center === */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
          {/* Tab bar */}
          <div style={{
            height: tabH, borderBottom: `1px solid ${HF.border}`, background: HF.panel,
            display: 'flex', alignItems: 'center',
          }}>
            {[
              { l: 'Pipeline',     ic: 'graph',   sel: true },
              { l: 'Convergence',  ic: 'activity'},
              { l: 'YAML setup',   ic: 'file',    dirty: true },
              { l: 'Schematic',    ic: 'chip'    },
              { l: 'Score shaping', ic: 'sliders'},
              { l: 'Diff vs r11',  ic: 'diff',    badge: '3' },
            ].map((t) => (
              <div key={t.l} style={{
                display: 'flex', alignItems: 'center', gap: 6,
                height: '100%', padding: '0 14px',
                borderRight: `1px solid ${HF.border}`,
                borderTop: t.sel ? `2px solid ${HF.accent}` : '2px solid transparent',
                marginTop: t.sel ? -1 : 0,
                background: t.sel ? HF.bg : HF.panel,
              }}>
                <HFIcon name={t.ic} size={12} color={t.sel ? HF.accent : HF.textMute} />
                <HFText size={12} weight={t.sel ? 600 : 500} color={t.sel ? HF.text : HF.textMute}>{t.l}</HFText>
                {t.dirty && <span style={{ color: HF.warn, fontFamily: HF.mono, fontSize: 10 }}>●</span>}
                {t.badge && <HFBadge tone="neutral" style={{ fontSize: 9, padding: '0 5px' }}>{t.badge}</HFBadge>}
              </div>
            ))}
            <div style={{ flex: 1 }} />
            <div style={{ padding: '0 12px', display: 'flex', gap: 6, alignItems: 'center' }}>
              <HFBadge tone="neutral">⌗ snap</HFBadge>
              <HFBadge tone="neutral">⚡ auto-layout</HFBadge>
            </div>
          </div>

          {/* Pipeline canvas */}
          <div style={{
            flex: 1, position: 'relative', overflow: 'hidden',
            backgroundImage: `radial-gradient(${HF.borderDk} 0.7px, transparent 0.7px)`,
            backgroundSize: '16px 16px',
            backgroundColor: HF.bg,
          }}>
            <svg width="100%" height="100%" style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
              {/* inputs -> dut */}
              {wire(190, 76, 232, 196)}
              {wire(190, 156, 232, 206)}
              {wire(190, 236, 232, 216)}
              {/* dut -> testbench */}
              {wire(400, 206, 442, 76)}
              {wire(400, 206, 442, 156)}
              {wire(400, 206, 442, 236)}
              {/* testbench -> specs */}
              {wire(610, 76, 654, 56, HF.accent)}
              {wire(610, 76, 654, 116, HF.accent)}
              {wire(610, 76, 654, 176, HF.accent)}
              {wire(610, 156, 654, 236)}
              {wire(610, 236, 654, 296)}
              {wire(610, 236, 654, 356)}
              {/* specs -> aggregate */}
              {wire(822, 56, 866, 196)}
              {wire(822, 116, 866, 196)}
              {wire(822, 176, 866, 196)}
              {wire(822, 236, 866, 206)}
              {wire(822, 296, 866, 216)}
              {wire(822, 356, 866, 216)}
              {/* aggregate -> optimizer */}
              {wire(1034, 206, 1078, 206)}
              {/* feedback */}
              <path d={`M 1140 240 C 1170 320 660 410 232 250`} fill="none"
                stroke={HF.accent} strokeWidth={1.6} strokeDasharray="5 4" opacity={0.85} />
            </svg>

            {/* Inputs */}
            <PNode x={64} y={50}  kind="input" title="Netlist" status="ok" sub="ota-improved.spice" />
            <PNode x={64} y={130} kind="input" title="PDK rules" status="ok" sub="ihp-sg13g2" />
            <PNode x={64} y={210} kind="input" title="PVT corner" status="ok" sub="tt · 25°C · 1.5 V" />

            {/* DUT vars */}
            <PNode x={232} y={180} kind="variables" title="DUT parameters" status="ok" sub="13 vars · 1 integer" />

            {/* Testbenches */}
            <PNode x={442} y={50}  kind="simulation" title="tb_ac" status="ok" sub="→ ugf · gain · pm" />
            <PNode x={442} y={130} kind="simulation" title="tb_noise" status="ok" sub="→ inoise · idd" />
            <PNode x={442} y={210} kind="simulation" title="tb_tran" status="ok" sub="→ tsettle" />

            {/* Specs */}
            <PNode x={654} y={30}  w={168} kind="spec · sigmoid" title="ugf" status="ok" accent selected sub="≥ 200 MHz" />
            <PNode x={654} y={90}  w={168} kind="spec · sigmoid" title="dcgain" status="ok" sub="≥ 40 dB" />
            <PNode x={654} y={150} w={168} kind="spec · sigmoid" title="pm" status="ok" sub="= 60° ± 10° · failing" />
            <PNode x={654} y={210} w={168} kind="spec · sigmoid" title="inoise" status="ok" sub="≤ 1.2 m" />
            <PNode x={654} y={270} w={168} kind="spec · sigmoid" title="idd" status="ok" sub="≤ 25 µA · failing" />
            <PNode x={654} y={330} w={168} kind="spec · sigmoid" title="tsettle" status="ok" sub="≤ 15 µs" />

            {/* Aggregate */}
            <PNode x={866} y={180} kind="scoring" title="Aggregate F(x)" status="ok" sub="weighted log sum  ·  0.412" />

            {/* Optimizer */}
            <PNode x={1078} y={180} kind="optimizer" title="LhsDE · seed 48" status="live" accent sub="847 / 2000 · 42%" />

            {/* feedback hint */}
            <div style={{ position: 'absolute', left: 580, top: 410 }}>
              <HFBadge tone="indigo" style={{ background: HF.panel }}>
                ↺ optimizer rewrites DUT params each iter
              </HFBadge>
            </div>

            {/* Spec inspector floating */}
            <div style={{
              position: 'absolute', right: 16, bottom: 16, width: 308,
              background: HF.panel, border: `1px solid ${HF.accentMid}`,
              borderRadius: 9, boxShadow: '0 6px 18px rgba(28,25,23,0.10)',
              padding: 12, display: 'flex', flexDirection: 'column', gap: 8,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <div>
                  <HFText size={10} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase', letterSpacing: 0.4 }}>Selected · spec</HFText>
                  <HFText size={13} weight={600}>ugf · unity gain freq</HFText>
                </div>
                <HFIcon name="x" size={12} color={HF.textMute} />
              </div>
              <div style={{ display: 'flex', gap: 5 }}>
                <HFBadge tone="indigo">sigmoid</HFBadge>
                <HFBadge tone="neutral">linear</HFBadge>
              </div>
              <HFPenalty width={282} height={72} />
              <HFMono size={10} color={HF.textMute}>exceed 200 MHz · w=100 · reward log</HFMono>
            </div>
          </div>

          {/* Bottom terminal panel */}
          <div style={{ height: bottomH, borderTop: `1px solid ${HF.border}`, background: HF.panelInv, display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', height: 30, padding: '0 12px', borderBottom: `1px solid ${HF.borderInv}`, alignItems: 'stretch' }}>
              {[
                ['Terminal', false],
                ['Optimizer log', true, '847'],
                ['Problems', false, '2'],
                ['Diff', false],
              ].map(([l, sel, badge]) => (
                <div key={l} style={{
                  display: 'flex', alignItems: 'center', gap: 6, padding: '0 12px',
                  borderBottom: sel ? `2px solid ${HF.accent}` : '2px solid transparent',
                }}>
                  <HFText size={12} weight={sel ? 600 : 500} color={sel ? '#fafaf9' : '#a8a29e'}>{l}</HFText>
                  {badge && <HFBadge tone="indigo" style={{ fontSize: 9, padding: '0 5px' }}>{badge}</HFBadge>}
                </div>
              ))}
              <div style={{ flex: 1 }} />
              <div style={{ display: 'flex', alignItems: 'center', color: '#a8a29e' }}>
                <HFText size={11} color="#a8a29e">streaming · 1 event / 1.4 s</HFText>
              </div>
            </div>
            <div style={{ flex: 1, padding: '8px 14px', overflow: 'hidden', display: 'flex', flexDirection: 'column', gap: 3 }}>
              {[
                ['845', '0.408', 'ugf=210M  dcgain=45.9dB  pm=51°  inoise=1.01m  idd=32µ  tsettle=12.8µ'],
                ['846', '0.410', 'ugf=212M  dcgain=46.0dB  pm=51°  inoise=0.99m  idd=31µ  tsettle=12.6µ'],
                ['847', '0.412', 'ugf=214M  dcgain=46.2dB  pm=52°  inoise=0.98m  idd=31µ  tsettle=12.4µ', 'new best'],
                [null, null, '↳ pm and idd failing  ·  sensitivity peaks on M5.NG and M1M2.W  →  fork branch?', 'hint'],
              ].map((row, i) => (
                <div key={i} style={{ display: 'flex', gap: 14, alignItems: 'baseline' }}>
                  <HFMono size={11} color="#78716c" style={{ width: 28 }}>{row[0] ?? ''}</HFMono>
                  {row[1]
                    ? <HFMono size={11} color={row[3] === 'new best' ? HF.accent : '#fafaf9'} weight={500} style={{ width: 64 }}>F={row[1]}</HFMono>
                    : <span style={{ width: 64 }} />}
                  <HFMono size={11} color={row[3] === 'hint' ? '#a8a29e' : '#e7e5e4'}>{row[2]}</HFMono>
                  {row[3] && row[3] !== 'hint' && (
                    <span style={{ marginLeft: 'auto', color: HF.accent, fontSize: 10, fontFamily: HF.mono }}>← {row[3]}</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* === Right rail: always-on === */}
        <div style={{ width: rightW, borderLeft: `1px solid ${HF.border}`, background: HF.panel, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          {/* Header strip with current run progress */}
          <div style={{ padding: '12px 14px', borderBottom: `1px solid ${HF.border}`, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <HFText size={11} color={HF.textMute} weight={600} style={{ textTransform: 'uppercase', letterSpacing: 0.4 }}>r12 · live</HFText>
                <HFText size={13} weight={600}>sigmoid · DE</HFText>
              </div>
              <HFButton size="sm" icon={<HFIcon name="stop" size={11} />}>Stop</HFButton>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
              <HFText size={11} color={HF.textMute}>iter 847 / 2000 · ETA 14 min</HFText>
              <HFMono size={11} weight={600} color={HF.accent}>F = 0.412</HFMono>
            </div>
            <div style={{ height: 5, background: HF.panelAlt, borderRadius: 999, overflow: 'hidden' }}>
              <div style={{ width: '42%', height: '100%', background: HF.accent }} />
            </div>
          </div>

          {/* Live spec status */}
          <div style={{ padding: '14px', display: 'flex', flexDirection: 'column', gap: 10 }}>
            <HFSectionLabel action={<HFBadge tone="warn">2 failing</HFBadge>}>Specs · live</HFSectionLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
              <HFSpecRow name="ugf"     value="214 MHz"     target="≥ 200 MHz" pass progress={0.92} />
              <HFSpecRow name="dcgain"  value="46.2 dB"     target="≥ 40 dB"   pass progress={0.88} />
              <HFSpecRow name="pm"      value="52°"         target="60° ± 10°" pass={false} progress={0.42} />
              <HFSpecRow name="inoise"  value="0.98 mV/√Hz" target="≤ 1.2 mV"  pass progress={0.78} />
              <HFSpecRow name="idd"     value="31 µA"       target="≤ 25 µA"   pass={false} progress={0.35} />
              <HFSpecRow name="tsettle" value="12.4 µs"     target="≤ 15 µs"   pass progress={0.83} />
            </div>
          </div>

          <div style={{ height: 1, background: HF.border }} />

          {/* Best params */}
          <div style={{ padding: '14px', display: 'flex', flexDirection: 'column', gap: 8, flex: 1, minHeight: 0, overflow: 'hidden' }}>
            <HFSectionLabel action={<HFMono size={10} color={HF.textMute}>iter 832</HFMono>}>Best params</HFSectionLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              {[
                ['M1M2 · W/L',  '24.2µ / 0.42µ'],
                ['M1cM2c · W/L','18.0µ / 0.36µ'],
                ['M3M4 · W/L',  '46.8µ / 0.55µ'],
                ['M5 · W/L · NG','12.0µ / 0.30µ · 3'],
                ['V_BIAS_1',    '0.62 V'],
                ['V_BIAS_2',    '0.81 V'],
              ].map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0', borderBottom: `1px dashed ${HF.border}` }}>
                  <HFMono size={11} color={HF.textMute}>{k}</HFMono>
                  <HFMono size={11} weight={500}>{v}</HFMono>
                </div>
              ))}
            </div>
          </div>

          <div style={{ padding: '12px 14px', borderTop: `1px solid ${HF.border}`, display: 'flex', flexDirection: 'column', gap: 6 }}>
            <HFButton kind="primary" icon={<HFIcon name="pin" size={12} color="#fff" />}>Pin as candidate</HFButton>
            <HFButton icon={<HFIcon name="file" size={12} />}>Export YAML  ·  apply best</HFButton>
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
        <HFMono size={11} color="#fff">ngspice · 1.41 s / iter</HFMono>
        <HFMono size={11} color="#fff">parallel · 4 workers</HFMono>
      </div>
    </div>
  );
}

window.HFRemix = HFRemix;
