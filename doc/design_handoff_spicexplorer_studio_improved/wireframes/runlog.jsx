// Direction A — "Run Log" workspace.
// Pattern: experiment tracker (W&B / Tensorboard / MLflow).
// Engineer's mental model: I tweak something, fire a run, compare to my history.
// Persistent left rail of runs; center is the current run; right rail is live spec status.

function RunLogWireframe() {
  const W = 1440, H = 900;
  const headerH = 56;
  const railW = 248;
  const rightW = 280;

  // Mock runs
  const runs = [
    { id: 'r12', name: 'sigmoid_de · seed 48', status: 'live', score: 0.41, spark: 'convFast', tag: 'sigmoid' },
    { id: 'r11', name: 'linear_de · seed 48', status: 'done', score: 0.18, spark: 'conv', tag: 'linear' },
    { id: 'r10', name: 'CMA · wider M5 range', status: 'done', score: 0.32, spark: 'conv', tag: 'sigmoid' },
    { id: 'r09', name: 'blind_search baseline', status: 'done', score: 0.05, spark: 'flat', tag: 'sigmoid' },
    { id: 'r08', name: 'fix PM target 60°', status: 'fail', score: null, spark: 'noisy', tag: 'sigmoid' },
    { id: 'r07', name: 'first cascode trial', status: 'done', score: 0.09, spark: 'flat', tag: 'linear' },
  ];

  const specs = [
    { name: 'ugf', pass: true, value: '214M', target: '≥ 200M', unit: 'Hz' },
    { name: 'dcgain', pass: true, value: '46.2', target: '≥ 40', unit: 'dB' },
    { name: 'pm', pass: false, value: '52°', target: '60° ±10', unit: '' },
    { name: 'inoise', pass: true, value: '0.98m', target: '≤ 1.2m', unit: 'V/√Hz' },
    { name: 'idd', pass: false, value: '31µ', target: '≤ 25µ', unit: 'A' },
    { name: 'tsettle', pass: true, value: '12.4µ', target: '≤ 15µ', unit: 's' },
  ];

  return (
    <div style={{
      width: W, height: H, background: SK.paper, fontFamily: SK.fontUI,
      color: SK.ink, position: 'relative', overflow: 'hidden',
    }}>
      {/* === Top header === */}
      <div style={{
        height: headerH, borderBottom: `1.5px solid ${SK.ink}`,
        display: 'flex', alignItems: 'center', padding: '0 20px', gap: 16,
      }}>
        <SkTitle size={22}>SpiceXplorer</SkTitle>
        <span style={{ color: SK.mute, fontFamily: SK.fontHand, fontSize: 16 }}>/</span>
        <SkText size={16}>OTA Cascode · ihp-sg13g2</SkText>
        <div style={{ flex: 1 }} />
        <SkPill bg={SK.softer}><SkMono size={10}>project_setup.yaml</SkMono></SkPill>
        <SkBtn ghost>Edit setup</SkBtn>
        <SkBtn accent>＋ New run</SkBtn>
      </div>

      <div style={{ display: 'flex', height: H - headerH }}>
        {/* === Left rail: runs history === */}
        <div style={{
          width: railW, borderRight: `1.5px solid ${SK.ink}`,
          padding: 14, display: 'flex', flexDirection: 'column', gap: 10,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <SkTitle size={15}>Runs</SkTitle>
            <SkText size={11} color={SK.mute}>6 of 24</SkText>
          </div>
          <SkBox dashed style={{ padding: '4px 8px', display: 'flex', alignItems: 'center', gap: 6 }}>
            <SkMono size={10} color={SK.mute}>⌕  filter · tag · seed</SkMono>
          </SkBox>
          {runs.map((r, i) => {
            const active = i === 0;
            return (
              <div key={r.id} style={{
                border: `${active ? 2 : 1.2}px solid ${active ? SK.accent : SK.ink}`,
                background: active ? '#fff' : 'transparent',
                borderRadius: 4, padding: 8,
                display: 'flex', flexDirection: 'column', gap: 4,
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <SkMono size={10} color={SK.mute}>{r.id}</SkMono>
                  <SkPill color={r.status === 'live' ? SK.accent : r.status === 'fail' ? SK.red : SK.green}
                    style={{ fontSize: 10, padding: '0 6px' }}>
                    {r.status === 'live' ? '● live' : r.status === 'fail' ? '✗ fail' : '✓ done'}
                  </SkPill>
                </div>
                <SkText size={12}>{r.name}</SkText>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <SkSparkline width={120} height={16} color={SK.ink} kind={r.spark} />
                  <SkMono size={10}>{r.score == null ? '—' : r.score.toFixed(2)}</SkMono>
                </div>
              </div>
            );
          })}
        </div>

        {/* === Center: current run === */}
        <div style={{ flex: 1, padding: 18, display: 'flex', flexDirection: 'column', gap: 14, minWidth: 0 }}>
          {/* Run summary header */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <SkTitle size={20}>r12 · sigmoid_de · seed 48</SkTitle>
            <SkPill color={SK.accent} bg={SK.accentSoft}>● running</SkPill>
            <SkPill bg={SK.softer}>LhsDE</SkPill>
            <SkPill bg={SK.softer}>sigmoid loss</SkPill>
            <div style={{ flex: 1 }} />
            <SkText size={12} color={SK.mute}>iter 847 / 2000 · 42 min left</SkText>
            <SkBtn ghost>Stop</SkBtn>
            <SkBtn ghost>Fork ↗</SkBtn>
          </div>

          {/* Tab bar */}
          <div style={{ display: 'flex', gap: 18, borderBottom: `1px solid ${SK.soft}`, paddingBottom: 2 }}>
            {['Overview', 'Convergence', 'Specs', 'Params', 'Score shaping', 'Diff vs r11'].map((t, i) => (
              <div key={t} style={{
                paddingBottom: 6,
                borderBottom: i === 0 ? `2px solid ${SK.accent}` : '2px solid transparent',
              }}>
                <SkText size={14} color={i === 0 ? SK.accent : SK.ink}>{t}</SkText>
              </div>
            ))}
          </div>

          {/* Row 1: progress bar */}
          <div style={{
            border: `1.5px solid ${SK.ink}`, borderRadius: 4, padding: 12,
            display: 'flex', flexDirection: 'column', gap: 8,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <SkText size={13}>Optimization progress</SkText>
              <SkMono size={11} color={SK.mute}>847 / 2000  ·  best score 0.412</SkMono>
            </div>
            <div style={{ height: 10, border: `1.2px solid ${SK.ink}`, borderRadius: 2, position: 'relative' }}>
              <div style={{ width: '42%', height: '100%', background: SK.accent }} />
            </div>
          </div>

          {/* Row 2: charts */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, flex: 1, minHeight: 0 }}>
            <SkBox style={{ padding: 8, display: 'flex', flexDirection: 'column' }}>
              <SkText size={13}>Score convergence  <SkText size={11} color={SK.mute}>· best-so-far + raw</SkText></SkText>
              <SkChart width={500} height={260} lines={['ink', 'accent']} target={false} />
            </SkBox>
            <SkBox style={{ padding: 8, display: 'flex', flexDirection: 'column' }}>
              <SkText size={13}>Best-so-far per metric  <SkText size={11} color={SK.mute}>· ugf</SkText></SkText>
              <SkChart width={500} height={260} lines={['blue']} target={true} />
            </SkBox>
          </div>

          {/* Row 3: live log strip */}
          <SkBox soft style={{ padding: 8, display: 'flex', alignItems: 'center', gap: 10, overflow: 'hidden' }}>
            <SkPill color={SK.accent}>iter 847</SkPill>
            <SkMono size={10} color={SK.mute}>ugf=214.2M  dcgain=46.2dB  pm=52°  inoise=0.98m  idd=31µA  tsettle=12.4µs  ·  score=0.412 ↑</SkMono>
          </SkBox>
        </div>

        {/* === Right rail: live spec status === */}
        <div style={{
          width: rightW, borderLeft: `1.5px solid ${SK.ink}`,
          padding: 14, display: 'flex', flexDirection: 'column', gap: 12,
        }}>
          <SkTitle size={15}>Spec status</SkTitle>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            {specs.map((s) => (
              <SkSpecChip key={s.name} {...s} />
            ))}
          </div>

          <SkTitle size={15} style={{ marginTop: 6 }}>Best params</SkTitle>
          <SkBox soft style={{ padding: 8, display: 'flex', flexDirection: 'column', gap: 3 }}>
            {[
              ['M1M2 · W/L', '24.2µ / 0.42µ'],
              ['M1CM2C · W/L', '18.0µ / 0.36µ'],
              ['M3M4 · W/L', '46.8µ / 0.55µ'],
              ['M3CM4C · W/L', '38.4µ / 0.50µ'],
              ['M5 · W/L · NG', '12.0µ / 0.30µ · 3'],
              ['M6 · W/L', '6.4µ / 0.40µ'],
              ['V_BIAS_1', '0.62 V'],
              ['V_BIAS_2', '0.81 V'],
            ].map(([k, v]) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between' }}>
                <SkMono size={10} color={SK.mute}>{k}</SkMono>
                <SkMono size={10}>{v}</SkMono>
              </div>
            ))}
          </SkBox>

          <SkBtn style={{ justifyContent: 'center' }}>Pin as candidate</SkBtn>
          <SkBtn ghost style={{ justifyContent: 'center' }}>Export YAML</SkBtn>
        </div>
      </div>

      {/* === Annotations === */}
      <SkCallout x={20} y={H - 56} w={500} color={SK.accent}>
        ↑ history is first-class · every tweak becomes a comparable run · the "Diff" tab shows what changed
      </SkCallout>
      <SkCallout x={W - 480} y={H - 56} w={460} color={SK.accent}>
        ↑ specs are always visible · no tab-hunting to know if the run is passing
      </SkCallout>
    </div>
  );
}

window.RunLogWireframe = RunLogWireframe;
