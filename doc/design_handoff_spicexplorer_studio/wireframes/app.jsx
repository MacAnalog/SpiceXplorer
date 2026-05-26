// Entry point: assembles the 5 wireframe directions onto a design canvas.

function SectionIntro() {
  return (
    <div style={{
      maxWidth: 1100, margin: '20px auto 0', padding: '6px 18px',
      fontFamily: SK.fontUI, color: SK.ink,
    }}>
      <div style={{ fontFamily: SK.fontHand, fontSize: 28, fontWeight: 700 }}>
        SpiceXplorer · UX directions
      </div>
      <div style={{ fontFamily: SK.fontHand, fontSize: 16, color: SK.mute, marginTop: 4 }}>
        Two sections · low-fi sketches up top · hi-fi A · D · E + a remix below.
      </div>
      <div style={{ fontFamily: SK.fontHand, fontSize: 14, color: SK.ink, marginTop: 10, lineHeight: 1.45 }}>
        Click any artboard to open it fullscreen. Drag tiles to rank or rearrange.
      </div>
    </div>
  );
}

function App() {
  const { DesignCanvas, DCSection, DCArtboard } = window;
  return (
    <>
      <SectionIntro />
      <DesignCanvas defaultZoom={0.7} background={SK.paper}>
        <DCSection id="explore" title="Five directions">
          <DCArtboard id="A" label="A · Run Log workspace  (experiment-tracker pattern)" width={1440} height={900}>
            <RunLogWireframe />
          </DCArtboard>
          <DCArtboard id="B" label="B · Notebook cells  (Jupyter / Observable pattern)" width={1440} height={900}>
            <NotebookWireframe />
          </DCArtboard>
          <DCArtboard id="C" label="C · Schematic-centric  (Cadence / xschem pattern)" width={1440} height={900}>
            <SchematicWireframe />
          </DCArtboard>
          <DCArtboard id="D" label="D · Pipeline / node graph  (ComfyUI / DAG pattern)" width={1440} height={900}>
            <NodesWireframe />
          </DCArtboard>
          <DCArtboard id="E" label="E · IDE three-pane  (VS Code pattern)" width={1440} height={900}>
            <IDEWireframe />
          </DCArtboard>
        </DCSection>

        <DCSection id="hifi" title="Higher fidelity"
          subtitle="A, D, E refined to a shippable look — same indigo/zinc palette as the existing UI. The fourth artboard is a remix combining the strongest parts of all three.">
          <DCArtboard id="HA" label="A · Run Log workspace  (hi-fi)" width={1440} height={900}>
            <HFRunLog />
          </DCArtboard>
          <DCArtboard id="HD" label="D · Pipeline / node graph  (hi-fi)" width={1440} height={900}>
            <HFNodes />
          </DCArtboard>
          <DCArtboard id="HE" label="E · IDE three-pane  (hi-fi)" width={1440} height={900}>
            <HFIDE />
          </DCArtboard>
          <DCArtboard id="REMIX" label="★ Studio (static remix · v1)" width={1440} height={900}>
            <HFRemix />
          </DCArtboard>
          <DCArtboard id="STUDIO" label="★ Studio · interactive — click around: tabs, activity bar, run history, score-shaping slider, compare A/B, ⌘K palette, + New project wizard" width={1440} height={900}>
            <HFStudio />
          </DCArtboard>
        </DCSection>
      </DesignCanvas>
    </>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
