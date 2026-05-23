// Direction 1: Observatory
// Light, modern engineering. Geist + JetBrains Mono. Restrained indigo + emerald.
// Linear/Vercel aesthetic applied to circuit optimization.

const obsTheme = {
  bg: "#fafafa",
  panel: "#ffffff",
  fg: "#0a0a0a",
  muted: "#71717a",
  faint: "#a1a1aa",
  border: "#e4e4e7",
  hairline: "#f4f4f5",
  grid: "rgba(0,0,0,0.06)",
  axis: "#a1a1aa",
  primary: "#4f46e5",      // indigo-600
  primarySoft: "#eef2ff",
  secondary: "#0891b2",    // cyan-600
  tertiary: "#ea580c",     // orange-600
  danger: "#dc2626",
  ok: "#059669",           // emerald-600
  okSoft: "#d1fae5",
  warnSoft: "#fef3c7",
  font: "'Geist', 'Inter Tight', -apple-system, BlinkMacSystemFont, sans-serif",
  monoFont: "'JetBrains Mono', 'IBM Plex Mono', ui-monospace, monospace",
  strokeWidth: 1.6,
  labelSize: 10,
  gridDash: "none",
  glow: false,
};

const obsCSS = `
.obs { font-family: ${obsTheme.font}; color: ${obsTheme.fg}; background: ${obsTheme.bg}; --primary: ${obsTheme.primary}; height: 100%; display: flex; flex-direction: column; font-variant-numeric: tabular-nums; }
.obs * { box-sizing: border-box; }
.obs .mono { font-family: ${obsTheme.monoFont}; font-feature-settings: "calt" 0; }
.obs .topbar { display: flex; align-items: center; gap: 14px; padding: 0 16px; height: 44px; border-bottom: 1px solid ${obsTheme.border}; background: ${obsTheme.panel}; }
.obs .brand { display: flex; align-items: baseline; gap: 8px; font-weight: 600; letter-spacing: -0.01em; }
.obs .brand b { font-weight: 600; }
.obs .brand .ver { font-family: ${obsTheme.monoFont}; font-size: 11px; color: ${obsTheme.muted}; }
.obs .tabs { display: flex; gap: 2px; flex: 1; align-items: stretch; }
.obs .tab { padding: 0 12px; display: flex; align-items: center; gap: 6px; font-size: 13px; color: ${obsTheme.muted}; cursor: pointer; border-bottom: 1.5px solid transparent; margin-bottom: -1px; position: relative; }
.obs .tab.active { color: ${obsTheme.fg}; border-bottom-color: ${obsTheme.primary}; font-weight: 500; }
.obs .tab.disabled { opacity: 0.4; cursor: not-allowed; }
.obs .tab .kbd { font-family: ${obsTheme.monoFont}; font-size: 10px; color: ${obsTheme.faint}; padding: 1px 4px; border: 1px solid ${obsTheme.border}; border-radius: 3px; }
.obs .pill { display: inline-flex; align-items: center; gap: 6px; padding: 2px 8px; border-radius: 999px; font-size: 11px; }
.obs .pill.ok { background: ${obsTheme.okSoft}; color: ${obsTheme.ok}; }
.obs .pill.warn { background: ${obsTheme.warnSoft}; color: #b45309; }
.obs .pill.live { background: ${obsTheme.primarySoft}; color: ${obsTheme.primary}; }
.obs .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.obs .dot.pulse { animation: obsPulse 1.4s ease-in-out infinite; }
@keyframes obsPulse { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(1.4); } }
.obs .topright { display: flex; align-items: center; gap: 10px; font-size: 12px; color: ${obsTheme.muted}; }
.obs .btn { font-family: ${obsTheme.font}; font-size: 12px; padding: 5px 10px; border-radius: 5px; border: 1px solid ${obsTheme.border}; background: ${obsTheme.panel}; color: ${obsTheme.fg}; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; transition: background 0.12s; }
.obs .btn:hover { background: ${obsTheme.hairline}; }
.obs .btn.primary { background: ${obsTheme.primary}; color: white; border-color: ${obsTheme.primary}; }
.obs .btn.primary:hover { background: #4338ca; }
.obs .btn.danger { background: ${obsTheme.danger}; color: white; border-color: ${obsTheme.danger}; }
.obs .btn.sm { padding: 3px 8px; font-size: 11px; }
.obs .content { flex: 1; display: flex; min-height: 0; overflow: hidden; }
.obs .leftrail { width: 200px; border-right: 1px solid ${obsTheme.border}; background: ${obsTheme.panel}; padding: 12px; font-size: 12px; flex-shrink: 0; overflow-y: auto; }
.obs .rail-section { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: ${obsTheme.faint}; margin: 12px 0 6px; font-weight: 500; }
.obs .rail-section:first-child { margin-top: 0; }
.obs .rail-item { display: flex; align-items: center; justify-content: space-between; padding: 4px 6px; border-radius: 4px; cursor: pointer; color: ${obsTheme.fg}; }
.obs .rail-item:hover { background: ${obsTheme.hairline}; }
.obs .rail-item .badge { font-family: ${obsTheme.monoFont}; font-size: 10px; color: ${obsTheme.muted}; }
.obs .main { flex: 1; overflow: auto; display: flex; flex-direction: column; min-width: 0; }
.obs .toolbar { display: flex; align-items: center; gap: 8px; padding: 8px 16px; border-bottom: 1px solid ${obsTheme.border}; background: ${obsTheme.panel}; flex-shrink: 0; }
.obs .toolbar .sep { width: 1px; height: 18px; background: ${obsTheme.border}; }
.obs .toolbar .label { font-size: 11px; color: ${obsTheme.muted}; }
.obs .toolbar select, .obs .toolbar input { font-family: ${obsTheme.font}; font-size: 12px; padding: 3px 8px; border: 1px solid ${obsTheme.border}; border-radius: 4px; background: ${obsTheme.panel}; color: ${obsTheme.fg}; }
.obs .toolbar input[type="number"] { width: 72px; }
.obs .panel { background: ${obsTheme.panel}; border: 1px solid ${obsTheme.border}; border-radius: 6px; overflow: hidden; }
.obs .panel-header { display: flex; align-items: center; justify-content: space-between; padding: 7px 12px; border-bottom: 1px solid ${obsTheme.border}; background: ${obsTheme.panel}; }
.obs .panel-title { font-size: 11px; font-weight: 500; color: ${obsTheme.fg}; letter-spacing: 0.01em; }
.obs .panel-title .mute { color: ${obsTheme.muted}; font-weight: 400; }
.obs .panel-body { padding: 10px 12px; }
.obs .panel-body.flush { padding: 0; }
table.obs-tbl { width: 100%; border-collapse: collapse; font-size: 12px; }
.obs table.obs-tbl th { text-align: left; font-weight: 500; color: ${obsTheme.muted}; font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; padding: 6px 10px; border-bottom: 1px solid ${obsTheme.border}; background: ${obsTheme.hairline}; }
.obs table.obs-tbl td { padding: 5px 10px; border-bottom: 1px solid ${obsTheme.hairline}; color: ${obsTheme.fg}; }
.obs table.obs-tbl tr:last-child td { border-bottom: none; }
.obs table.obs-tbl td.mono { font-family: ${obsTheme.monoFont}; font-size: 11px; }
.obs table.obs-tbl tr.footer td { border-top: 1px solid ${obsTheme.border}; background: ${obsTheme.hairline}; font-weight: 500; }
.obs .kv { display: grid; grid-template-columns: 100px 1fr; gap: 4px 12px; font-size: 12px; }
.obs .kv dt { color: ${obsTheme.muted}; }
.obs .kv dd { margin: 0; font-family: ${obsTheme.monoFont}; font-size: 11px; color: ${obsTheme.fg}; }
.obs .yamlpane { background: #1c1c1f; color: #d4d4d8; font-family: ${obsTheme.monoFont}; font-size: 11.5px; line-height: 1.55; padding: 12px 14px; white-space: pre; overflow: auto; height: 100%; }
.obs .yamlpane .yk { color: #c4b5fd; }
.obs .yamlpane .yv { color: #86efac; }
.obs .yamlpane .yn { color: #fda4af; }
.obs .yamlpane .yc { color: #71717a; font-style: italic; }
.obs .gutter { color: #52525b; user-select: none; display: inline-block; width: 22px; text-align: right; padding-right: 8px; }
.obs .chiprow { display: flex; flex-wrap: wrap; gap: 6px; }
.obs .specchip { display: inline-flex; align-items: center; gap: 6px; padding: 4px 8px; border-radius: 4px; border: 1px solid ${obsTheme.border}; background: ${obsTheme.panel}; font-size: 11px; }
.obs .specchip .name { font-family: ${obsTheme.monoFont}; }
.obs .specchip .val { font-family: ${obsTheme.monoFont}; font-weight: 500; }
.obs .specchip.ok { border-color: ${obsTheme.ok}; background: #f0fdf4; }
.obs .specchip.fail { border-color: ${obsTheme.danger}; background: #fef2f2; }
.obs .progress { height: 4px; background: ${obsTheme.hairline}; border-radius: 2px; overflow: hidden; flex: 1; }
.obs .progress .bar { height: 100%; background: ${obsTheme.primary}; transition: width 0.3s; }
.obs .slider-track { position: relative; height: 28px; }
.obs .slider-track .rail { position: absolute; top: 13px; left: 0; right: 0; height: 2px; background: ${obsTheme.border}; border-radius: 1px; }
.obs .slider-track .fill { position: absolute; top: 13px; height: 2px; background: ${obsTheme.primary}; border-radius: 1px; }
.obs .slider-track .thumb { position: absolute; top: 8px; width: 12px; height: 12px; border-radius: 50%; background: white; border: 2px solid ${obsTheme.primary}; transform: translateX(-50%); cursor: pointer; }
.obs .slider-track .marker { position: absolute; top: 8px; width: 1px; height: 12px; background: ${obsTheme.muted}; }
.obs .slider-track .marker .lbl { position: absolute; top: 14px; left: 50%; transform: translateX(-50%); font-family: ${obsTheme.monoFont}; font-size: 9px; color: ${obsTheme.muted}; white-space: nowrap; }
.obs .wizard-stepper { display: flex; align-items: center; gap: 0; padding: 6px 12px; border-bottom: 1px solid ${obsTheme.border}; background: ${obsTheme.panel}; }
.obs .wizard-step { display: flex; align-items: center; gap: 6px; padding: 4px 10px; font-size: 12px; color: ${obsTheme.muted}; cursor: pointer; border-radius: 4px; }
.obs .wizard-step.active { color: ${obsTheme.primary}; background: ${obsTheme.primarySoft}; }
.obs .wizard-step.done { color: ${obsTheme.ok}; }
.obs .wizard-step .num { font-family: ${obsTheme.monoFont}; font-size: 10px; width: 16px; height: 16px; border-radius: 50%; background: ${obsTheme.hairline}; display: inline-flex; align-items: center; justify-content: center; }
.obs .wizard-step.active .num { background: ${obsTheme.primary}; color: white; }
.obs .wizard-step.done .num { background: ${obsTheme.ok}; color: white; }
.obs .wizard-step .arrow { color: ${obsTheme.faint}; margin: 0 2px; }
.obs .modeseg { display: inline-flex; padding: 2px; background: ${obsTheme.hairline}; border-radius: 6px; }
.obs .modeseg .opt { padding: 4px 12px; font-size: 12px; border-radius: 4px; cursor: pointer; color: ${obsTheme.muted}; }
.obs .modeseg .opt.active { background: ${obsTheme.panel}; color: ${obsTheme.fg}; box-shadow: 0 1px 2px rgba(0,0,0,0.06); }
.obs .callout { display: flex; gap: 10px; padding: 10px 12px; background: ${obsTheme.primarySoft}; border-radius: 6px; font-size: 12px; color: ${obsTheme.fg}; border-left: 2px solid ${obsTheme.primary}; }
.obs .callout b { color: ${obsTheme.primary}; }
.obs .input { font-family: ${obsTheme.font}; font-size: 12px; padding: 4px 8px; border: 1px solid ${obsTheme.border}; border-radius: 4px; background: ${obsTheme.panel}; color: ${obsTheme.fg}; width: 100%; }
.obs .input.mono { font-family: ${obsTheme.monoFont}; font-size: 11.5px; }
.obs .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.obs .grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
.obs .stat { padding: 8px 10px; border: 1px solid ${obsTheme.border}; border-radius: 6px; background: ${obsTheme.panel}; }
.obs .stat .l { font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; color: ${obsTheme.muted}; }
.obs .stat .v { font-family: ${obsTheme.monoFont}; font-size: 18px; color: ${obsTheme.fg}; margin-top: 2px; }
.obs .stat .d { font-size: 11px; color: ${obsTheme.muted}; margin-top: 2px; }
.obs .stat .d.up { color: ${obsTheme.ok}; }
.obs .stat .d.dn { color: ${obsTheme.danger}; }
`;

function ObsTopbar({ tab, setTab, running }) {
  return (
    <div className="topbar">
      <div className="brand">
        <svg width="18" height="18" viewBox="0 0 18 18">
          <rect x="2" y="2" width="14" height="14" rx="3" fill={obsTheme.primary} />
          <path d="M5 9 L9 13 L13 5" stroke="white" strokeWidth="1.6" fill="none" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <b>SpiceXplorer</b>
        <span className="ver mono">0.4.2 · NEWCAS demo</span>
      </div>
      <div className="tabs">
        {[
          { id: "setup", label: "Setup", kbd: "1" },
          { id: "score", label: "Score Shaping", kbd: "2" },
          { id: "optimize", label: "Optimize", kbd: "3" },
          { id: "explorer", label: "Explorer", kbd: "4" },
        ].map(t => (
          <div key={t.id} className={`tab ${tab === t.id ? "active" : ""}`} onClick={() => setTab(t.id)}>
            {t.label} <span className="kbd">{t.kbd}</span>
          </div>
        ))}
      </div>
      <div className="topright">
        <span className="pill ok"><span className="dot" /> project applied</span>
        {running && <span className="pill live"><span className="dot pulse" /> replay · iter 147/1000</span>}
        <span className="mono">cascode-ota-sg13g2</span>
      </div>
    </div>
  );
}

function ObsSetupTab() {
  const [mode, setMode] = useState("wizard");
  const [step, setStep] = useState("dut");
  const D = window.SPICE_DATA;
  return (
    <>
      <div className="toolbar">
        <div className="modeseg">
          <div className={`opt ${mode === "load" ? "active" : ""}`} onClick={() => setMode("load")}>Load / Edit YAML</div>
          <div className={`opt ${mode === "wizard" ? "active" : ""}`} onClick={() => setMode("wizard")}>Create Wizard</div>
        </div>
        <span className="sep" />
        {mode === "load" && <>
          <span className="label">demo</span>
          <select defaultValue="cascode">
            <option value="cascode">OTA Cascode (default)</option>
            <option value="folded">OTA Folded Cascode</option>
            <option value="ldo">LDO Regulator</option>
          </select>
          <button className="btn sm">Upload .yaml</button>
        </>}
        <div style={{ flex: 1 }} />
        <button className="btn sm">Validate</button>
        <button className="btn sm primary">Apply →</button>
      </div>
      {mode === "wizard" && (
        <div className="wizard-stepper">
          {[
            { id: "basic", label: "Project", done: true },
            { id: "pdk", label: "PDK rules", done: true },
            { id: "dut", label: "DUT params", done: false },
            { id: "pvt", label: "PVT corners", done: false },
            { id: "tb", label: "Testbenches", done: false },
            { id: "specs", label: "Target specs", done: false },
            { id: "opt", label: "Optimizer", done: false },
          ].map((s, i, arr) => (
            <React.Fragment key={s.id}>
              <div className={`wizard-step ${step === s.id ? "active" : ""} ${s.done ? "done" : ""}`} onClick={() => setStep(s.id)}>
                <span className="num">{s.done ? "✓" : i + 1}</span>{s.label}
              </div>
              {i < arr.length - 1 && <span className="arrow">›</span>}
            </React.Fragment>
          ))}
        </div>
      )}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, padding: 12, flex: 1, minHeight: 0, overflow: "auto" }}>
        {/* LEFT: YAML / Wizard form */}
        {mode === "load" ? (
          <div className="panel" style={{ display: "flex", flexDirection: "column" }}>
            <div className="panel-header">
              <div className="panel-title">project_setup.yaml <span className="mono mute" style={{ fontSize: 10 }}>{D.PROJECT.yamlPath}</span></div>
              <div style={{ display: "flex", gap: 6, alignItems: "center", fontSize: 10, color: obsTheme.muted }}>
                <span className="mono">UTF-8 · 142 lines</span>
                <span className="pill ok"><span className="dot" /> valid</span>
              </div>
            </div>
            <div className="yamlpane">
              {`<span class="gutter">1</span><span class="yc"># Cascode OTA · sg13g2</span>
<span class="gutter">2</span><span class="yk">project</span>:
<span class="gutter">3</span>  <span class="yk">name</span>: <span class="yv">cascode-ota-sg13g2</span>
<span class="gutter">4</span>  <span class="yk">simulator</span>: <span class="yv">ngspice</span>
<span class="gutter">5</span>  <span class="yk">technology</span>: <span class="yv">ihp-sg13g2</span>
<span class="gutter">6</span>  <span class="yk">workspace</span>: <span class="yv">./work</span>
<span class="gutter">7</span>
<span class="gutter">8</span><span class="yk">dut</span>:
<span class="gutter">9</span>  <span class="yk">netlist</span>: <span class="yv">cascode_ota.spice</span>
<span class="gutter">10</span>  <span class="yk">params</span>:
<span class="gutter">11</span>    - { <span class="yk">name</span>: <span class="yv">w_n1</span>, <span class="yk">min</span>: <span class="yn">1e-6</span>, <span class="yk">max</span>: <span class="yn">200e-6</span>, <span class="yk">log</span>: <span class="yn">true</span> }
<span class="gutter">12</span>    - { <span class="yk">name</span>: <span class="yv">l_n1</span>, <span class="yk">min</span>: <span class="yn">130e-9</span>, <span class="yk">max</span>: <span class="yn">2e-6</span> }
<span class="gutter">13</span>    - { <span class="yk">name</span>: <span class="yv">w_p1</span>, <span class="yk">min</span>: <span class="yn">1e-6</span>, <span class="yk">max</span>: <span class="yn">200e-6</span>, <span class="yk">log</span>: <span class="yn">true</span> }
<span class="gutter">14</span>    - { <span class="yk">name</span>: <span class="yv">i_bias</span>, <span class="yk">min</span>: <span class="yn">1e-6</span>, <span class="yk">max</span>: <span class="yn">200e-6</span>, <span class="yk">log</span>: <span class="yn">true</span> }
<span class="gutter">15</span>
<span class="gutter">16</span><span class="yk">pvt</span>:
<span class="gutter">17</span>  - { <span class="yk">temp</span>: <span class="yn">27</span>, <span class="yk">corner</span>: <span class="yv">tt</span>, <span class="yk">vdd</span>: <span class="yn">1.2</span> }
<span class="gutter">18</span>  - { <span class="yk">temp</span>: <span class="yn">-40</span>, <span class="yk">corner</span>: <span class="yv">ff</span>, <span class="yk">vdd</span>: <span class="yn">1.32</span> }
<span class="gutter">19</span>  - { <span class="yk">temp</span>: <span class="yn">125</span>, <span class="yk">corner</span>: <span class="yv">ss</span>, <span class="yk">vdd</span>: <span class="yn">1.08</span> }
<span class="gutter">20</span>
<span class="gutter">21</span><span class="yk">testbenches</span>:
<span class="gutter">22</span>  - <span class="yk">name</span>: <span class="yv">ac_response</span>
<span class="gutter">23</span>    <span class="yk">netlist</span>: <span class="yv">tb_ac.spice</span>
<span class="gutter">24</span>    <span class="yk">measures</span>: [<span class="yv">ugf</span>, <span class="yv">gain</span>, <span class="yv">pm</span>]
<span class="gutter">25</span>
<span class="gutter">26</span><span class="yk">target_specs</span>:
<span class="gutter">27</span>  - { <span class="yk">name</span>: <span class="yv">ugf</span>,    <span class="yk">goal</span>: <span class="yv">">"</span>, <span class="yk">target</span>: <span class="yn">200e6</span>, <span class="yk">tol</span>: <span class="yn">10e6</span>, <span class="yk">range</span>: <span class="yn">50e6</span> }
<span class="gutter">28</span>  - { <span class="yk">name</span>: <span class="yv">gain</span>,   <span class="yk">goal</span>: <span class="yv">">"</span>, <span class="yk">target</span>: <span class="yn">40</span>,    <span class="yk">tol</span>: <span class="yn">2</span>,    <span class="yk">range</span>: <span class="yn">10</span> }
<span class="gutter">29</span>  - { <span class="yk">name</span>: <span class="yv">pm</span>,     <span class="yk">goal</span>: <span class="yv">">"</span>, <span class="yk">target</span>: <span class="yn">60</span>,    <span class="yk">tol</span>: <span class="yn">5</span>,    <span class="yk">range</span>: <span class="yn">15</span> }
<span class="gutter">30</span>  - { <span class="yk">name</span>: <span class="yv">current</span>,<span class="yk">goal</span>: <span class="yv">"<"</span>, <span class="yk">target</span>: <span class="yn">100e-6</span>, <span class="yk">tol</span>: <span class="yn">5e-6</span> }
<span class="gutter">31</span>`}
            </div>
          </div>
        ) : (
          <div className="panel" style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <div className="panel-header">
              <div className="panel-title">step 3 · DUT parameters <span className="mono mute" style={{ fontSize: 10 }}>· 11 params · 1 frozen</span></div>
              <div style={{ display: "flex", gap: 6 }}>
                <button className="btn sm">Upload netlist</button>
                <button className="btn sm">+ Add param</button>
              </div>
            </div>
            <div className="panel-body" style={{ overflow: "auto" }}>
              <div style={{ display: "flex", gap: 6, marginBottom: 10, fontSize: 11, color: obsTheme.muted, alignItems: "center" }}>
                <span className="mono">cascode_ota.spice</span><span>·</span>
                <span>parsed 11 <span className="mono">.param</span> declarations</span>
                <span style={{ flex: 1 }} />
                <span className="pill ok"><span className="dot" /> auto-filled</span>
              </div>
              <table className="obs-tbl">
                <thead>
                  <tr><th>param</th><th>min</th><th>max</th><th>init</th><th>log</th><th>int</th><th>freeze</th></tr>
                </thead>
                <tbody>
                  {D.DUT_PARAMS.map((p, i) => (
                    <tr key={i} style={{ opacity: p.frozen ? 0.55 : 1 }}>
                      <td className="mono">{p.name}</td>
                      <td className="mono">{p.min}</td>
                      <td className="mono">{p.max}</td>
                      <td className="mono">{p.val}</td>
                      <td><span style={{ color: p.log ? obsTheme.primary : obsTheme.faint }}>{p.log ? "●" : "○"}</span></td>
                      <td><span style={{ color: p.int ? obsTheme.primary : obsTheme.faint }}>{p.int ? "●" : "○"}</span></td>
                      <td><span style={{ color: p.frozen ? obsTheme.tertiary : obsTheme.faint }}>{p.frozen ? "● frozen" : "○"}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div style={{ display: "flex", justifyContent: "space-between", marginTop: 12 }}>
                <button className="btn sm">← PDK rules</button>
                <button className="btn sm primary">PVT corners →</button>
              </div>
            </div>
          </div>
        )}
        {/* RIGHT: Summary panels */}
        <div style={{ display: "flex", flexDirection: "column", gap: 10, minHeight: 0 }}>
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">project summary <span className="mute">· live preview</span></div>
              <span className="mono" style={{ fontSize: 10, color: obsTheme.muted }}>cascode-ota-sg13g2</span>
            </div>
            <div className="panel-body">
              <dl className="kv">
                <dt>name</dt><dd>cascode-ota-sg13g2</dd>
                <dt>simulator</dt><dd>ngspice 41</dd>
                <dt>technology</dt><dd>ihp-sg13g2</dd>
                <dt>workspace</dt><dd>./work</dd>
                <dt>params</dt><dd>11 · 4 log · 1 int · 1 frozen</dd>
                <dt>pvt corners</dt><dd>3 · tt/ff/ss · 1.20V ± 10%</dd>
                <dt>testbenches</dt><dd>4 · 3 enabled</dd>
                <dt>specs</dt><dd>6 · weight Σ = 4.6</dd>
              </dl>
            </div>
          </div>
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">target specs <span className="mute">· 6</span></div>
            </div>
            <table className="obs-tbl">
              <thead><tr><th>name</th><th>goal</th><th>target</th><th>tol</th><th>weight</th></tr></thead>
              <tbody>
                {D.SPECS.map((s, i) => (
                  <tr key={i}>
                    <td className="mono">{s.name}</td>
                    <td className="mono">{s.goal}</td>
                    <td className="mono">{window.SPICE_FMT.formatEng(s.target, 3)}{s.unit}</td>
                    <td className="mono">±{window.SPICE_FMT.formatEng(s.tol, 2)}</td>
                    <td className="mono">{s.weight.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">device under test <span className="mute">· schematic</span></div>
              <span className="mono" style={{ fontSize: 10, color: obsTheme.muted }}>cascode_ota.spice</span>
            </div>
            <div style={{ padding: 6 }}>
              <SchematicPanel width={420} height={210} theme={obsTheme} accent={obsTheme.primary} />
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

function ObsScoreTab() {
  const [specIdx, setSpecIdx] = useState(0);
  const D = window.SPICE_DATA;
  const spec = D.SPECS[specIdx];
  const [value, setValue] = useState(spec.current);
  useEffect(() => setValue(spec.current), [specIdx]);
  const lo = spec.target - 3 * spec.range;
  const hi = spec.target + 3 * spec.range;
  const t = (value - lo) / (hi - lo);
  const dir = spec.goal === ">" ? (spec.target - value) : (value - spec.target);
  const sigP = 1 / (1 + Math.exp(-dir / (spec.range * 0.35)));
  const linP = Math.max(0, Math.min(1, dir / spec.range));
  const totalSig = D.SPECS.reduce((acc, s) => acc + s.sigP * s.weight, 0);
  const totalLin = D.SPECS.reduce((acc, s) => acc + s.linP * s.weight, 0);

  return (
    <>
      <div className="toolbar">
        <span className="label">spec</span>
        <select value={specIdx} onChange={e => setSpecIdx(+e.target.value)}>
          {D.SPECS.map((s, i) => <option key={i} value={i}>{s.name} · {s.goal} {window.SPICE_FMT.formatEng(s.target, 3)}{s.unit}</option>)}
        </select>
        <span className="sep" />
        <span className="label">range</span>
        <span className="mono" style={{ fontSize: 11 }}>target ± 3 × {window.SPICE_FMT.formatEng(spec.range, 2)}</span>
        <div style={{ flex: 1 }} />
        <span className="mono" style={{ fontSize: 11, color: obsTheme.muted }}>POST /api/score · 150ms debounce</span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: 12, padding: 12, flex: 1, minHeight: 0, overflow: "auto" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">penalty curve · sigmoid vs linear</div>
              <span className="mono" style={{ fontSize: 10, color: obsTheme.muted }}>{spec.name} · {spec.goal} {window.SPICE_FMT.formatEng(spec.target, 3)}{spec.unit}</span>
            </div>
            <div className="panel-body">
              <PenaltyCurveChart width={620} height={280} theme={obsTheme} spec={spec} currentValue={value} />
              <div style={{ marginTop: 10 }}>
                <div className="slider-track" style={{ position: "relative" }}>
                  <div className="rail" />
                  <div className="fill" style={{ left: 0, width: `${t * 100}%` }} />
                  <div className="marker" style={{ left: `${((spec.target - lo) / (hi - lo)) * 100}%` }}>
                    <span className="lbl">target</span>
                  </div>
                  <div className="thumb" style={{ left: `${t * 100}%` }} />
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6, fontSize: 11, color: obsTheme.muted }}>
                  <span className="mono">{window.SPICE_FMT.formatEng(lo, 2)}{spec.unit}</span>
                  <span className="mono" style={{ color: obsTheme.fg, fontWeight: 500 }}>now {window.SPICE_FMT.formatEng(value, 3)}{spec.unit}</span>
                  <span className="mono">{window.SPICE_FMT.formatEng(hi, 2)}{spec.unit}</span>
                </div>
              </div>
            </div>
          </div>
          <div className="callout">
            <span style={{ fontFamily: obsTheme.monoFont, fontSize: 14, color: obsTheme.primary }}>◆</span>
            <div>
              Under <b>sigmoid shaping</b>, <span className="mono">ugf</span> (P̂={D.SPECS[0].sigP.toFixed(2)}) and <span className="mono">vno_int</span> (P̂={D.SPECS[5].sigP.toFixed(2)}) dominate F(x).
              Under <b>linear shaping</b>, <span className="mono">current</span> (P̂={D.SPECS[3].linP.toFixed(2)}) is washed out — the optimizer chases the wrong constraint.
              This is the paper's core claim.
            </div>
          </div>
        </div>
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title">per-spec breakdown <span className="mute">· F(x) = Σ wᵢ · P̂ᵢ</span></div>
          </div>
          <table className="obs-tbl">
            <thead><tr><th>spec</th><th>current</th><th>sigmoid P̂</th><th>linear P̂</th><th>w</th></tr></thead>
            <tbody>
              {D.SPECS.map((s, i) => (
                <tr key={i} style={i === specIdx ? { background: obsTheme.primarySoft } : {}}>
                  <td className="mono" style={{ fontWeight: i === specIdx ? 600 : 400 }}>{s.name}</td>
                  <td className="mono" style={{ color: s.met ? obsTheme.ok : obsTheme.danger }}>{window.SPICE_FMT.formatEng(s.current, 3)}{s.unit}</td>
                  <td className="mono">
                    <span style={{ display: "inline-block", verticalAlign: "middle", width: 36, height: 4, background: obsTheme.hairline, borderRadius: 2, marginRight: 6, position: "relative" }}>
                      <span style={{ position: "absolute", left: 0, top: 0, height: "100%", width: `${s.sigP * 100}%`, background: obsTheme.primary, borderRadius: 2 }} />
                    </span>
                    {s.sigP.toFixed(2)}
                  </td>
                  <td className="mono">
                    <span style={{ display: "inline-block", verticalAlign: "middle", width: 36, height: 4, background: obsTheme.hairline, borderRadius: 2, marginRight: 6, position: "relative" }}>
                      <span style={{ position: "absolute", left: 0, top: 0, height: "100%", width: `${s.linP * 100}%`, background: obsTheme.secondary, borderRadius: 2 }} />
                    </span>
                    {s.linP.toFixed(2)}
                  </td>
                  <td className="mono">{s.weight.toFixed(1)}</td>
                </tr>
              ))}
              <tr className="footer">
                <td colSpan={2} className="mono">F(x) aggregate</td>
                <td className="mono" style={{ color: obsTheme.primary }}>{totalSig.toFixed(3)}</td>
                <td className="mono" style={{ color: obsTheme.secondary }}>{totalLin.toFixed(3)}</td>
                <td className="mono">Σ {D.SPECS.reduce((a, s) => a + s.weight, 0).toFixed(1)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

function ObsOptimizeTab() {
  const D = window.SPICE_DATA;
  const [metric, setMetric] = useState("ugf");
  const metricMap = { ugf: { traj: D.METRIC_TRAJ.ugf.sigmoid, target: 200e6, unit: "Hz", label: "ugf" },
                      gain: { traj: D.METRIC_TRAJ.gain.sigmoid, target: 40, unit: "dB", label: "gain" },
                      pm: { traj: D.METRIC_TRAJ.pm.sigmoid, target: 60, unit: "°", label: "pm" },
                      current: { traj: D.METRIC_TRAJ.current.sigmoid, target: 100e-6, unit: "A", label: "current" } };
  const m = metricMap[metric];
  return (
    <>
      <div className="toolbar">
        <span className="label">algorithm</span>
        <select defaultValue="LhsDE"><option>LhsDE</option><option>LHSSearch</option><option>LogBFGSCMAPlus</option></select>
        <span className="label">budget</span>
        <input type="number" defaultValue={1000} />
        <span className="label">score</span>
        <div className="modeseg">
          <div className="opt active">sigmoid</div>
          <div className="opt">linear</div>
        </div>
        <span className="sep" />
        <span className="label">demo replay</span>
        <select defaultValue="sigmoid_de">{D.CHECKPOINTS.map(c => <option key={c.id} value={c.id}>{c.label}</option>)}</select>
        <div style={{ flex: 1 }} />
        <button className="btn sm danger"><span className="dot" /> Stop</button>
        <button className="btn sm">+ Save checkpoint</button>
      </div>
      <div style={{ padding: 12, display: "flex", flexDirection: "column", gap: 10, flex: 1, minHeight: 0, overflow: "auto" }}>
        {/* Stat strip */}
        <div className="grid3">
          <div className="stat">
            <div className="l">iteration · budget 1000</div>
            <div className="v">147 / 1000</div>
            <div className="progress" style={{ marginTop: 8 }}><div className="bar" style={{ width: "14.7%" }} /></div>
          </div>
          <div className="stat">
            <div className="l">best F(x) · sigmoid</div>
            <div className="v">0.624 <span style={{ fontSize: 12, color: obsTheme.ok }}>↓</span></div>
            <div className="d up">−5.81 since iter 0</div>
          </div>
          <div className="stat">
            <div className="l">specs met · 6 total</div>
            <div className="v">3 / 6</div>
            <div className="d">gain · current · sr_pos</div>
          </div>
        </div>
        <div className="grid2">
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">F(x) convergence <span className="mute">· raw + best-so-far</span></div>
              <span className="mono" style={{ fontSize: 10, color: obsTheme.primary }}>● best  ● raw</span>
            </div>
            <div className="panel-body">
              <ScoreConvergenceChart width={560} height={240} theme={obsTheme} traj={D.TRAJ_SIGMOID} iterMax={147} animate={false} />
            </div>
          </div>
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">metric · {m.label} best-so-far</div>
              <select value={metric} onChange={e => setMetric(e.target.value)} style={{ fontFamily: obsTheme.monoFont, fontSize: 11 }}>
                <option value="ugf">ugf</option><option value="gain">gain</option><option value="pm">pm</option><option value="current">current</option>
              </select>
            </div>
            <div className="panel-body">
              <MetricConvergenceChart width={560} height={240} theme={obsTheme} traj={m.traj.slice(0, 147)} target={m.target} unit={m.unit} label="" />
            </div>
          </div>
        </div>
        <div className="grid2">
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">live spec status <span className="mute">· @ iter 147</span></div>
              <span className="mono" style={{ fontSize: 10, color: obsTheme.muted }}>3 / 6 met</span>
            </div>
            <div className="panel-body">
              <div className="chiprow">
                {D.SPECS.map((s, i) => (
                  <span key={i} className={`specchip ${s.met ? "ok" : "fail"}`}>
                    <span className="dot" style={{ background: s.met ? obsTheme.ok : obsTheme.danger }} />
                    <span className="name">{s.name}</span>
                    <span className="val">{window.SPICE_FMT.formatEng(s.current, 3)}{s.unit}</span>
                    <span className="mono" style={{ color: obsTheme.muted, fontSize: 10 }}>{s.goal}{window.SPICE_FMT.formatEng(s.target, 2)}{s.unit}</span>
                  </span>
                ))}
              </div>
              <div style={{ marginTop: 10, fontSize: 11, color: obsTheme.muted, fontFamily: obsTheme.monoFont }}>
                <div>[147 · 2026-02-07T10:57:32Z] eval ✓ score=0.624 · best</div>
                <div>[146 · 2026-02-07T10:57:30Z] eval ✓ score=0.681</div>
                <div>[145 · 2026-02-07T10:57:28Z] eval ✓ score=0.692</div>
                <div>[144 · 2026-02-07T10:57:26Z] eval ✗ ngspice diverged · skip</div>
              </div>
            </div>
          </div>
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">best design · iter 147</div>
              <button className="btn sm">Export .spice</button>
            </div>
            <table className="obs-tbl">
              <thead><tr><th>param</th><th>value</th><th>min..max</th><th></th></tr></thead>
              <tbody>
                {D.DUT_PARAMS.slice(0, 6).map((p, i) => (
                  <tr key={i}>
                    <td className="mono">{p.name}</td>
                    <td className="mono">{p.val}</td>
                    <td className="mono" style={{ color: obsTheme.muted }}>{p.min} .. {p.max}</td>
                    <td>
                      <Sparkline data={Array.from({length: 30}, (_, k) => Math.sin(k * 0.4 + i) * 0.3 + 0.5 + (i / 6))} width={60} height={14} color={obsTheme.primary} fill={obsTheme.primarySoft} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </>
  );
}

function ObsExplorerTab() {
  const D = window.SPICE_DATA;
  return (
    <>
      <div className="toolbar">
        <span className="label">run A</span>
        <select defaultValue="sigmoid_de">{D.CHECKPOINTS.map(c => <option key={c.id} value={c.id}>{c.label}</option>)}</select>
        <span className="label">run B</span>
        <select defaultValue="linear_de">{D.CHECKPOINTS.map(c => <option key={c.id} value={c.id}>{c.label}</option>)}</select>
        <button className="btn sm">Load both</button>
        <span className="sep" />
        <span className="mono" style={{ fontSize: 11, color: obsTheme.muted }}>2 runs · 400 evals · 21.4s sim time</span>
        <div style={{ flex: 1 }} />
        <button className="btn sm">Export CSV</button>
        <button className="btn sm primary">Compare report</button>
      </div>
      <div style={{ padding: 12, display: "flex", flexDirection: "column", gap: 10, flex: 1, minHeight: 0, overflow: "auto" }}>
        <div className="grid2">
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">F(x) convergence · A vs B</div>
              <span className="mono" style={{ fontSize: 10 }}><span style={{ color: obsTheme.primary }}>● A sigmoid</span> &nbsp;<span style={{ color: obsTheme.secondary }}>● B linear</span></span>
            </div>
            <div className="panel-body">
              <ScoreConvergenceChart width={560} height={240} theme={obsTheme} traj={D.TRAJ_SIGMOID} overlay={D.TRAJ_LINEAR} showRaw={false} label="" />
            </div>
          </div>
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">ugf · best-so-far</div>
              <span className="mono" style={{ fontSize: 10 }}>target 200 MHz</span>
            </div>
            <div className="panel-body">
              <MetricConvergenceChart width={560} height={240} theme={obsTheme} traj={D.METRIC_TRAJ.ugf.sigmoid} overlay={D.METRIC_TRAJ.ugf.linear} target={200e6} unit="Hz" label="" />
            </div>
          </div>
        </div>
        <div className="grid2">
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">metric scatter · ugf vs current</div>
              <span className="mono" style={{ fontSize: 10 }}>● A feasible &nbsp;● B feasible &nbsp;○ infeasible</span>
            </div>
            <div className="panel-body">
              <ScatterChart width={560} height={240} theme={obsTheme} pointsA={D.SCATTER_A} pointsB={D.SCATTER_B} xLabel="ugf" yLabel="current" xTarget={200e6} yTarget={100e-6} xGoal=">" yGoal="<" />
            </div>
          </div>
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">performance envelope <span className="mute">· best achievable per spec</span></div>
            </div>
            <table className="obs-tbl">
              <thead><tr><th>spec</th><th>target</th><th>run A best</th><th>run B best</th><th>winner</th></tr></thead>
              <tbody>
                {D.ENVELOPE.map((e, i) => {
                  const winA = (e.spec === "current" || e.spec === "vno_int") ? e.bestA < e.bestB : e.bestA > e.bestB;
                  return (
                    <tr key={i}>
                      <td className="mono">{e.spec}</td>
                      <td className="mono" style={{ color: obsTheme.muted }}>{D.SPECS.find(s => s.name === e.spec) && (D.SPECS.find(s => s.name === e.spec).goal + " " + window.SPICE_FMT.formatEng(D.SPECS.find(s => s.name === e.spec).target, 2) + D.SPECS.find(s => s.name === e.spec).unit)}</td>
                      <td className="mono" style={{ color: winA ? obsTheme.primary : obsTheme.fg, fontWeight: winA ? 500 : 400 }}>{e.bestA.toFixed(1)} {e.unit}</td>
                      <td className="mono" style={{ color: !winA ? obsTheme.secondary : obsTheme.fg, fontWeight: !winA ? 500 : 400 }}>{e.bestB.toFixed(1)} {e.unit}</td>
                      <td><span className="pill" style={{ background: winA ? "#eef2ff" : "#ecfeff", color: winA ? obsTheme.primary : obsTheme.secondary }}>{winA ? "A" : "B"}</span></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
        <div className="grid2">
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">ugf distribution · A vs B</div>
              <span className="mono" style={{ fontSize: 10, color: obsTheme.muted }}>n=200 each · binwidth 7 MHz</span>
            </div>
            <div className="panel-body">
              <HistogramChart width={560} height={220} theme={obsTheme} dataA={D.HIST_A_UGF} dataB={D.HIST_B_UGF} target={200e6} label="" />
            </div>
          </div>
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">spec summary · pass / fail</div>
            </div>
            <table className="obs-tbl">
              <thead><tr><th>spec</th><th>goal</th><th>A</th><th>B</th></tr></thead>
              <tbody>
                {D.SPECS.map((s, i) => (
                  <tr key={i}>
                    <td className="mono">{s.name}</td>
                    <td className="mono" style={{ color: obsTheme.muted }}>{s.goal} {window.SPICE_FMT.formatEng(s.target, 2)}{s.unit}</td>
                    <td><span className="pill ok"><span className="dot" />pass</span></td>
                    <td><span className="pill" style={{ background: i % 2 ? "#fef2f2" : obsTheme.okSoft, color: i % 2 ? obsTheme.danger : obsTheme.ok }}><span className="dot" />{i % 2 ? "fail" : "pass"}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </>
  );
}

function ObservatoryApp({ initialTab = "setup" }) {
  const [tab, setTab] = useState(initialTab);
  const D = window.SPICE_DATA;
  return (
    <div className="obs">
      <style>{obsCSS}</style>
      <ObsTopbar tab={tab} setTab={setTab} running={tab === "optimize" || tab === "score"} />
      <div className="content">
        <div className="leftrail">
          <div className="rail-section">project</div>
          <div className="rail-item"><span>cascode-ota-sg13g2</span><span className="badge mono">v3</span></div>
          <div className="rail-item" style={{ color: obsTheme.muted }}><span>folded-cascode</span><span className="badge mono">draft</span></div>
          <div className="rail-item" style={{ color: obsTheme.muted }}><span>ldo-1v2</span><span className="badge mono">v1</span></div>
          <div className="rail-section">checkpoints</div>
          {D.CHECKPOINTS.map((c, i) => (
            <div key={i} className="rail-item" style={{ display: "block", padding: "5px 6px" }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ fontSize: 11 }}>{c.label}</span>
                <span className="badge mono">{c.bestScore.toFixed(2)}</span>
              </div>
              <div className="mono" style={{ fontSize: 9, color: obsTheme.faint, marginTop: 2 }}>{c.iters} iters · {c.when}</div>
            </div>
          ))}
          <div className="rail-section">resources</div>
          <div className="rail-item"><span style={{ color: obsTheme.muted, fontFamily: obsTheme.monoFont, fontSize: 11 }}>ngspice ▸ 41</span></div>
          <div className="rail-item"><span style={{ color: obsTheme.muted, fontFamily: obsTheme.monoFont, fontSize: 11 }}>workers</span><span className="badge mono">8/8</span></div>
          <div className="rail-item"><span style={{ color: obsTheme.muted, fontFamily: obsTheme.monoFont, fontSize: 11 }}>cache hit</span><span className="badge mono">62%</span></div>
        </div>
        <div className="main">
          {tab === "setup" && <ObsSetupTab />}
          {tab === "score" && <ObsScoreTab />}
          {tab === "optimize" && <ObsOptimizeTab />}
          {tab === "explorer" && <ObsExplorerTab />}
        </div>
      </div>
    </div>
  );
}

window.ObservatoryApp = ObservatoryApp;
