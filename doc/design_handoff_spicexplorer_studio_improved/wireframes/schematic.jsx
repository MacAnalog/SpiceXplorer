// Direction C — "Schematic-centric".
// Pattern: Cadence / xschem-inspired. The schematic IS the navigation.
// Click a transistor to set its W/L bounds. Hover to see what specs depend on it.
// Specs and live run status are docked in the surrounding rails.

function SchematicWireframe() {
  const W = 1440, H = 900;
  const headerH = 52;

  return (
    <div style={{ width: W, height: H, background: SK.paper, position: 'relative', overflow: 'hidden' }}>
      {/* header */}
      <div style={{
        height: headerH, borderBottom: `1.5px solid ${SK.ink}`, padding: '0 22px',
        display: 'flex', alignItems: 'center', gap: 14,
      }}>
        <SkTitle size={20}>SpiceXplorer</SkTitle>
        <SkText size={13} color={SK.mute}>· cascode_ota · ihp-sg13g2</SkText>
        {/* breadcrumb mode switcher */}
        <div style={{ display: 'flex', gap: 0, marginLeft: 14, border: `1.2px solid ${SK.ink}`, borderRadius: 4, overflow: 'hidden' }}>
          {['Schematic', 'Specs', 'Run', 'Compare'].map((m, i) => (
            <div key={m} style={{
              padding: '4px 12px',
              background: i === 0 ? SK.ink : 'transparent',
              color: i === 0 ? SK.paper : SK.ink,
              fontFamily: SK.fontHand, fontSize: 13,
              borderRight: i < 3 ? `1px solid ${SK.ink}` : 'none',
            }}>{m}</div>
          ))}
        </div>
        <div style={{ flex: 1 }} />
        <SkPill color={SK.green} bg="#eaf2ec">last sim 1.4 s</SkPill>
        <SkBtn ghost>Save sizing</SkBtn>
        <SkBtn accent>▶ Optimize</SkBtn>
      </div>

      <div style={{ display: 'flex', height: H - headerH }}>
        {/* === Left palette: device groups === */}
        <div style={{
          width: 220, borderRight: `1.5px solid ${SK.ink}`, padding: 12,
          display: 'flex', flexDirection: 'column', gap: 10,
        }}>
          <SkTitle size={14}>Device groups</SkTitle>
          {[
            { name: 'Input pair · M1/M2', n: 'NMOS', sel: false },
            { name: 'NMOS cascode · M1c/M2c', n: 'NMOS', sel: false },
            { name: 'PMOS cascode · M3c/M4c', n: 'PMOS', sel: false },
            { name: 'PMOS load · M3/M4', n: 'PMOS', sel: false },
            { name: 'Tail · M5  (NG=3)', n: 'NMOS', sel: true },
            { name: 'Bias · M6', n: 'NMOS', sel: false },
          ].map((g) => (
            <div key={g.name} style={{
              border: `${g.sel ? 2 : 1.2}px solid ${g.sel ? SK.accent : SK.ink}`,
              borderRadius: 4, padding: 8,
              background: g.sel ? SK.accentSoft : 'transparent',
              display: 'flex', flexDirection: 'column', gap: 2,
            }}>
              <SkMono size={11} style={{ fontWeight: 600 }}>{g.name}</SkMono>
              <SkMono size={10} color={SK.mute}>{g.n}  ·  click to edit bounds</SkMono>
            </div>
          ))}
          <SkText size={11} color={SK.mute}>
            ↑ groups are matched-pair · sizing one sizes both
          </SkText>

          <div style={{ marginTop: 6 }}>
            <SkTitle size={14}>Bias nets</SkTitle>
          </div>
          <SkBox soft style={{ padding: 6 }}>
            <SkMono size={10}>V_BIAS_1  ·  [0, 1.2] V</SkMono><br />
            <SkMono size={10}>V_BIAS_2  ·  [0, 1.2] V</SkMono>
          </SkBox>
        </div>

        {/* === Center: schematic stage === */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
          {/* schematic canvas */}
          <div style={{
            flex: 1, padding: 20, position: 'relative',
            background: `repeating-linear-gradient(0deg, transparent 0 19px, ${SK.softer} 19px 20px), repeating-linear-gradient(90deg, transparent 0 19px, ${SK.softer} 19px 20px)`,
          }}>
            <div style={{ position: 'absolute', top: 14, left: 24, display: 'flex', gap: 8, alignItems: 'center' }}>
              <SkPill bg="#fff">⌖ click any device  →  edit bounds</SkPill>
              <SkPill bg="#fff">hover  →  see which specs depend</SkPill>
            </div>
            <div style={{
              position: 'absolute', top: '50%', left: '50%',
              transform: 'translate(-50%, -50%)',
              background: SK.paper,
              border: `1.5px dashed ${SK.mute}`,
              padding: 12,
              borderRadius: 6,
            }}>
              <SkSchematic width={520} height={340} highlighted="M5" />
            </div>

            {/* Spec → device hint */}
            <div style={{ position: 'absolute', top: 100, right: 30, transform: 'rotate(-2deg)' }}>
              <SkBox accent style={{ padding: 8, background: '#fff', maxWidth: 200 }}>
                <SkText size={12} color={SK.accent}>
                  ugf depends on  <SkMono color={SK.accent}>M1/M2 · W</SkMono>  &amp;  <SkMono color={SK.accent}>M5 · I</SkMono>
                </SkText>
              </SkBox>
            </div>

            {/* annotation arrow */}
            <div style={{ position: 'absolute', bottom: 80, left: 80, transform: 'rotate(3deg)' }}>
              <SkText size={13} color={SK.accent}>highlighted = selected → edit panel on the right</SkText>
            </div>
          </div>

          {/* bottom strip: live mini convergence + status */}
          <div style={{
            borderTop: `1.5px solid ${SK.ink}`, padding: 12,
            display: 'grid', gridTemplateColumns: '1fr 1.4fr', gap: 14,
            background: SK.softer,
          }}>
            <SkBox style={{ padding: 8 }}>
              <SkText size={12}>score · last optimize  ·  best 0.412</SkText>
              <SkChart width={460} height={100} lines={['ink', 'accent']} target={false} noAxis />
            </SkBox>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 6 }}>
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

        {/* === Right: inspector for selected device === */}
        <div style={{ width: 280, borderLeft: `1.5px solid ${SK.ink}`, padding: 14, display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div>
            <SkText size={11} color={SK.mute}>SELECTED</SkText>
            <SkTitle size={16}>Tail current · M5</SkTitle>
            <SkMono size={10} color={SK.mute}>NMOS  ·  multiplier NG=3</SkMono>
          </div>

          <SkBox soft style={{ padding: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <SkText size={12}>Width  W</SkText>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <SkMono size={10}>0.18µ</SkMono>
              <div style={{ flex: 1, height: 6, border: `1.2px solid ${SK.ink}`, borderRadius: 999, position: 'relative' }}>
                <div style={{ position: 'absolute', left: '8%', right: '20%', top: 1.5, height: 1.5, background: SK.accent }} />
                <div style={{ position: 'absolute', left: '8%', top: -3, width: 10, height: 12, background: SK.ink, borderRadius: 2 }} />
                <div style={{ position: 'absolute', left: '80%', top: -3, width: 10, height: 12, background: SK.ink, borderRadius: 2 }} />
              </div>
              <SkMono size={10}>200µ</SkMono>
            </div>
            <SkMono size={10} color={SK.mute}>current best: 12.0 µ</SkMono>

            <SkText size={12} style={{ marginTop: 4 }}>Length  L</SkText>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <SkMono size={10}>0.18µ</SkMono>
              <div style={{ flex: 1, height: 6, border: `1.2px solid ${SK.ink}`, borderRadius: 999, position: 'relative' }}>
                <div style={{ position: 'absolute', left: '4%', right: '60%', top: 1.5, height: 1.5, background: SK.accent }} />
                <div style={{ position: 'absolute', left: '4%', top: -3, width: 10, height: 12, background: SK.ink, borderRadius: 2 }} />
                <div style={{ position: 'absolute', left: '40%', top: -3, width: 10, height: 12, background: SK.ink, borderRadius: 2 }} />
              </div>
              <SkMono size={10}>10µ</SkMono>
            </div>
            <SkMono size={10} color={SK.mute}>current best: 0.30 µ</SkMono>

            <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
              <SkPill bg={SK.softer}>☐ freeze</SkPill>
              <SkPill bg={SK.softer}>☐ log scale</SkPill>
              <SkPill bg={SK.softer}>☑ integer (NG)</SkPill>
            </div>
          </SkBox>

          <SkTitle size={13}>Sensitivity  <SkText size={11} color={SK.mute}>· last run</SkText></SkTitle>
          <SkBox style={{ padding: 8, display: 'flex', flexDirection: 'column', gap: 4 }}>
            {[
              ['idd', 0.92, 'high'],
              ['ugf', 0.55, 'med'],
              ['inoise', 0.41, 'med'],
              ['pm', 0.12, 'low'],
              ['dcgain', 0.07, 'low'],
            ].map(([m, v, t]) => (
              <div key={m} style={{ display: 'grid', gridTemplateColumns: '60px 1fr 38px', alignItems: 'center', gap: 6 }}>
                <SkMono size={10}>{m}</SkMono>
                <div style={{ height: 6, background: SK.softer, borderRadius: 999, overflow: 'hidden', border: `1px solid ${SK.soft}` }}>
                  <div style={{ width: `${v * 100}%`, height: '100%', background: t === 'high' ? SK.accent : t === 'med' ? SK.amber : SK.mute }} />
                </div>
                <SkMono size={10} color={SK.mute}>{(v * 100).toFixed(0)}%</SkMono>
              </div>
            ))}
          </SkBox>

          <SkText size={11} color={SK.mute}>↑ sizing this device most strongly trades off <SkMono color={SK.ink}>idd</SkMono> ↔ <SkMono color={SK.ink}>ugf</SkMono></SkText>
        </div>
      </div>

      {/* annotations */}
      <SkCallout x={240} y={H - 30} w={400}>
        ↑ no YAML to write · the circuit IS the form
      </SkCallout>
    </div>
  );
}

window.SchematicWireframe = SchematicWireframe;
