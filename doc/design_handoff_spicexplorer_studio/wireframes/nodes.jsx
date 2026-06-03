// Direction D — "Pipeline / Node graph".
// Pattern: a DAG you can SEE. Each block is a swap-in stage of the optimization recipe.
// Engineers can re-wire which testbench feeds which spec, swap optimizers, fork branches.
// Best when you want to make data flow obvious rather than hide it in tabs.

function NodesWireframe() {
  const W = 1440, H = 900;
  const headerH = 52;

  // Node layout: roughly left-to-right pipeline. Coordinates inside the canvas area.
  const N = ({ x, y, w = 200, h = 110, title, sub, body, accent = false, status = null, ports = { i: [], o: [] } }) => (
    <div style={{
      position: 'absolute', left: x, top: y, width: w, minHeight: h,
      border: `${accent ? 2 : 1.5}px solid ${accent ? SK.accent : SK.ink}`,
      background: '#fff', borderRadius: 6, padding: 10,
      boxShadow: '3px 3px 0 ' + SK.soft,
      display: 'flex', flexDirection: 'column', gap: 6,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <SkText size={13} style={{ fontWeight: 700 }}>{title}</SkText>
        {status && <SkPill color={status === 'ok' ? SK.green : status === 'live' ? SK.accent : SK.mute} style={{ fontSize: 10 }}>{status === 'ok' ? '✓' : status === 'live' ? '●' : '○'} {status}</SkPill>}
      </div>
      {sub && <SkMono size={10} color={SK.mute}>{sub}</SkMono>}
      {body}
    </div>
  );

  // Wire helper inside SVG
  const wire = (x1, y1, x2, y2, color = SK.ink, dashed = false) => {
    const dx = Math.max(40, Math.abs(x2 - x1) * 0.45);
    return (
      <path
        d={`M ${x1} ${y1} C ${x1 + dx} ${y1} ${x2 - dx} ${y2} ${x2} ${y2}`}
        fill="none" stroke={color} strokeWidth={1.6}
        strokeDasharray={dashed ? '4 4' : 'none'}
        markerEnd="url(#arrow)"
      />
    );
  };

  return (
    <div style={{ width: W, height: H, background: SK.paper, position: 'relative', overflow: 'hidden' }}>
      {/* header */}
      <div style={{
        height: headerH, borderBottom: `1.5px solid ${SK.ink}`, padding: '0 22px',
        display: 'flex', alignItems: 'center', gap: 14,
      }}>
        <SkTitle size={20}>SpiceXplorer · Pipeline</SkTitle>
        <SkText size={13} color={SK.mute}>· cascode_ota.graph</SkText>
        <div style={{ flex: 1 }} />
        <SkPill bg={SK.softer}>⌗ snap to grid</SkPill>
        <SkBtn ghost>＋ Add node</SkBtn>
        <SkBtn ghost>Fork branch ⎇</SkBtn>
        <SkBtn accent>▶ Run pipeline</SkBtn>
      </div>

      <div style={{ display: 'flex', height: H - headerH }}>
        {/* Left palette */}
        <div style={{ width: 200, borderRight: `1.5px solid ${SK.ink}`, padding: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
          <SkTitle size={14}>Node palette</SkTitle>
          {[
            ['Inputs', ['Netlist file', 'PDK / Tech rules', 'PVT corner']],
            ['Variables', ['DUT params', 'Sweep set']],
            ['Simulation', ['Testbench  · AC', 'Testbench · Noise', 'Testbench · Tran']],
            ['Scoring', ['Spec (sigmoid)', 'Spec (linear)', 'Aggregate F(x)']],
            ['Optimizers', ['LhsDE', 'CMA-ES', 'LBFGSB', 'Blind search']],
            ['Outputs', ['Convergence plot', 'Pareto / scatter', 'Best params YAML']],
          ].map(([group, items]) => (
            <div key={group}>
              <SkText size={11} color={SK.mute} style={{ textTransform: 'uppercase', letterSpacing: 0.5 }}>{group}</SkText>
              {items.map((it) => (
                <div key={it} style={{
                  border: `1px dashed ${SK.ink}`, borderRadius: 3,
                  padding: '3px 6px', marginTop: 4,
                }}>
                  <SkMono size={10}>{it}</SkMono>
                </div>
              ))}
            </div>
          ))}
        </div>

        {/* Canvas */}
        <div style={{
          flex: 1, position: 'relative', overflow: 'hidden',
          background: `radial-gradient(${SK.soft} 1px, transparent 1px)`,
          backgroundSize: '20px 20px', backgroundPosition: '0 0',
        }}>
          {/* SVG for wires (under nodes) */}
          <svg width="100%" height="100%" style={{ position: 'absolute', inset: 0 }}>
            <defs>
              <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto" markerUnits="strokeWidth">
                <path d="M0,1 L6,4 L0,7" fill="none" stroke={SK.ink} strokeWidth={1.2} />
              </marker>
            </defs>
            {/* Wires (rough coords below match the absolute-positioned nodes) */}
            {/* netlist + pdk -> dut params */}
            {wire(216, 130, 256, 240)}
            {wire(216, 250, 256, 250)}
            {wire(216, 370, 256, 270)}
            {/* dut params -> testbenches */}
            {wire(456, 250, 500, 130)}
            {wire(456, 260, 500, 250)}
            {wire(456, 270, 500, 370)}
            {/* testbenches -> specs */}
            {wire(700, 130, 744, 130)}
            {wire(700, 250, 744, 250)}
            {wire(700, 370, 744, 370)}
            {wire(700, 250, 744, 370)}
            {wire(700, 130, 744, 250)}
            {/* specs -> aggregate */}
            {wire(944, 130, 988, 250)}
            {wire(944, 250, 988, 250)}
            {wire(944, 370, 988, 250)}
            {/* aggregate -> optimizer */}
            {wire(1188, 250, 1232, 250)}
            {/* optimizer -> outputs (loop back arrow) */}
            {wire(1232, 280, 360, 540, SK.accent, true)}
            {/* optimizer -> outputs */}
            {wire(1332, 350, 1332, 440)}
          </svg>

          {/* Nodes */}
          <N x={36} y={100} w={180} title="Netlist" sub="ota-improved.spice" status="ok"
            body={<SkMono size={10} color={SK.mute}>6 transistors  ·  3 .param</SkMono>} />
          <N x={36} y={220} w={180} title="PDK rules" sub="ihp-sg13g2" status="ok"
            body={<SkMono size={10} color={SK.mute}>min/max W,L per type</SkMono>} />
          <N x={36} y={340} w={180} title="PVT" sub="1 corner" status="ok"
            body={<SkMono size={10} color={SK.mute}>tt · 25°C · 1.5 V</SkMono>} />

          <N x={256} y={210} w={200} title="DUT params" sub="13 variables · 1 integer" status="ok"
            body={<div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
              {['M1M2 W,L', 'M3M4 W,L', 'M1cM2c W,L', 'M3cM4c W,L', 'M5 W,L,NG', 'M6 W,L', 'V_BIAS_1', 'V_BIAS_2'].map((p) => (
                <SkPill key={p} bg={SK.softer} style={{ fontSize: 9, padding: '0 5px' }}>{p}</SkPill>
              ))}
            </div>} />

          <N x={500} y={100} w={200} title="Testbench · tb_ac" sub="AC · CL=50f" status="ok"
            body={<SkMono size={10} color={SK.mute}>→ ugf · gain · pm</SkMono>} />
          <N x={500} y={220} w={200} title="Testbench · tb_noise" sub="Noise + DC op" status="ok"
            body={<SkMono size={10} color={SK.mute}>→ inoise · idd</SkMono>} />
          <N x={500} y={340} w={200} title="Testbench · tb_tran" sub="Step response" status="ok"
            body={<SkMono size={10} color={SK.mute}>→ tsettle</SkMono>} />

          <N x={744} y={100} w={200} title="Spec · ugf" sub="≥ 200M · sigmoid" status="ok" accent
            body={<SkMono size={10} color={SK.accent}>w=100 · tol 10M</SkMono>} />
          <N x={744} y={220} w={200} title="Spec · dcgain, pm" sub="2 specs · sigmoid" status="ok"
            body={<SkMono size={10} color={SK.mute}>≥40dB  ·  =60°±10</SkMono>} />
          <N x={744} y={340} w={200} title="Spec · noise, idd, tsettle" sub="3 specs · sigmoid" status="ok"
            body={<SkMono size={10} color={SK.mute}>minimize bundle</SkMono>} />

          <N x={988} y={210} w={200} title="Aggregate F(x)" sub="weighted sum  ·  log reward" status="ok"
            body={<SkMono size={10} color={SK.mute}>6 specs in</SkMono>} />

          <N x={1232} y={220} w={180} title="Optimizer" sub="LhsDE  ·  seed 48" status="live" accent
            body={
              <div>
                <SkMono size={10} color={SK.accent}>iter 847 / 2000</SkMono>
                <div style={{ height: 5, border: `1.2px solid ${SK.accent}`, borderRadius: 999, marginTop: 4 }}>
                  <div style={{ width: '42%', height: '100%', background: SK.accent }} />
                </div>
              </div>
            } />

          <N x={1232} y={440} w={180} title="Outputs" sub="checkpoints · plots"
            body={<SkMono size={10} color={SK.mute}>JSON · YAML · PNG</SkMono>} />

          {/* loop annotation */}
          <div style={{ position: 'absolute', left: 600, top: 590, transform: 'rotate(-3deg)' }}>
            <SkText size={13} color={SK.accent}>
              ↺ feedback loop · optimizer rewrites DUT params each iteration
            </SkText>
          </div>

          {/* live preview floater (inspector for selected node) */}
          <div style={{
            position: 'absolute', right: 16, bottom: 16, width: 320,
            border: `1.5px solid ${SK.accent}`, background: '#fff',
            borderRadius: 6, padding: 12, boxShadow: '3px 3px 0 ' + SK.soft,
            display: 'flex', flexDirection: 'column', gap: 6,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <SkTitle size={14}>Inspecting: Spec · ugf</SkTitle>
              <SkText size={11} color={SK.mute}>node</SkText>
            </div>
            <SkPenaltyCurve width={296} height={90} />
            <SkMono size={10} color={SK.mute}>goal exceed 200 MHz · sigmoid penalty</SkMono>
            <div style={{ display: 'flex', gap: 6 }}>
              <SkPill bg={SK.accentSoft} color={SK.accent}>● sigmoid</SkPill>
              <SkPill color={SK.mute}>○ linear</SkPill>
              <SkPill bg={SK.softer}>w 100</SkPill>
            </div>
          </div>
        </div>
      </div>

      <SkCallout x={36} y={H - 50} w={400}>
        ↑ swap optimizers · re-route testbenches · fork the spec block to compare loss shapes
      </SkCallout>
    </div>
  );
}

window.NodesWireframe = NodesWireframe;
