// Studio context — shared state across the interactive remix.
// Loaded BEFORE views/rails/shell so they can read window.StudioCtx at render time.

const StudioCtx = React.createContext(null);

const STUDIO_INITIAL = {
  // Navigation
  activeActivity: 'runs',         // runs | setup | pipeline | schematic | specs | compare
  activeTab: 'pipeline',          // pipeline | convergence | yaml | schematic | shaping | compare

  // Selection
  selectedRunId: 'r12',
  selectedSpec: 'ugf',
  selectedDevice: 'M5',

  // Score shaping
  specShape: 'sigmoid',           // sigmoid | linear
  specTryValue: 187,              // try-value slider (MHz for ugf)

  // Optimize
  algo: 'LhsDE',
  budget: 2000,
  isRunning: true,
  replaySource: null,             // null | 'sigmoid_de' | 'linear_de' | 'blind_search'

  // Compare
  compareRunA: 'r12',
  compareRunB: 'r11',
  scatterX: 'idd',
  scatterY: 'ugf',

  // Panels
  rightOpen: true,
  bottomOpen: true,
  bottomTab: 'log',               // terminal | log | problems | diff

  // Overlays
  commandOpen: false,
  wizardOpen: false,
  wizardStep: 0,
  yamlDirty: true,
};

// Shared mock data
const STUDIO_RUNS = [
  { id: 'r12', name: 'sigmoid · DE · seed 48',  status: 'live', score: 0.412, dScore: '+0.09', spark: 'convFast', algo: 'LhsDE',  loss: 'sigmoid', iter: 847,  budget: 2000, time: '2 min ago' },
  { id: 'r11', name: 'linear · DE · seed 48',   status: 'done', score: 0.183, dScore: null,    spark: 'conv',     algo: 'LhsDE',  loss: 'linear',  iter: 2000, budget: 2000, time: '32 min ago' },
  { id: 'r10', name: 'CMA · wider M5 W',        status: 'done', score: 0.318, dScore: '−0.09', spark: 'conv',     algo: 'CMA',    loss: 'sigmoid', iter: 2000, budget: 2000, time: '1 h ago' },
  { id: 'r09', name: 'blind LHS baseline',      status: 'done', score: 0.054, dScore: null,    spark: 'flat',     algo: 'LHS',    loss: 'sigmoid', iter: 500,  budget: 500,  time: '3 h ago' },
  { id: 'r08', name: 'PM=60° hard constraint',  status: 'fail', score: null,  dScore: null,    spark: 'noisy',    algo: 'LhsDE',  loss: 'sigmoid', iter: 120,  budget: 2000, time: '6 h ago' },
  { id: 'r07', name: 'first cascode trial',     status: 'done', score: 0.091, dScore: null,    spark: 'flat',     algo: 'LBFGSB', loss: 'linear',  iter: 600,  budget: 600,  time: '1 d ago' },
];

const STUDIO_SPECS = [
  { name: 'ugf',     desc: 'Unity gain frequency',  testbench: 'tb_ac',    goal: 'exceed',   target: 200e6, target_str: '200 MHz', value: '214 MHz',     value_num: 214e6, unit: 'Hz',  pass: true,  progress: 0.92, weight: 100, shape: 'sigmoid', P_lin: 0.07, P_sig: 0.04, range: 100e6 },
  { name: 'dcgain',  desc: 'DC gain',               testbench: 'tb_ac',    goal: 'exceed',   target: 40,    target_str: '40 dB',   value: '46.2 dB',     value_num: 46.2,  unit: 'dB',  pass: true,  progress: 0.88, weight: 100, shape: 'sigmoid', P_lin: 0.06, P_sig: 0.02, range: 100 },
  { name: 'pm',      desc: 'Phase margin',          testbench: 'tb_ac',    goal: 'exact',    target: 60,    target_str: '60°',     value: '52°',         value_num: 52,    unit: '°',   pass: false, progress: 0.42, weight: 100, shape: 'sigmoid', P_lin: 0.13, P_sig: 0.74, range: 45 },
  { name: 'inoise',  desc: 'Input-referred noise', testbench: 'tb_noise', goal: 'minimize', target: 1.2e-3, target_str: '1.2 mV/√Hz', value: '0.98 mV/√Hz', value_num: 0.98e-3,unit: 'V/√Hz', pass: true,  progress: 0.78, weight: 100, shape: 'sigmoid', P_lin: 0.10, P_sig: 0.05, range: 1e-3 },
  { name: 'idd',     desc: 'Supply current',        testbench: 'tb_noise', goal: 'minimize', target: 25e-6, target_str: '25 µA',   value: '31 µA',       value_num: 31e-6, unit: 'A',   pass: false, progress: 0.35, weight: 100, shape: 'sigmoid', P_lin: 0.24, P_sig: 0.81, range: 100e-6 },
  { name: 'tsettle', desc: 'Settling time',         testbench: 'tb_tran',  goal: 'minimize', target: 15e-6, target_str: '15 µs',   value: '12.4 µs',     value_num: 12.4e-6,unit: 's',   pass: true,  progress: 0.83, weight: 10,  shape: 'sigmoid', P_lin: 0.05, P_sig: 0.02, range: 1e-6 },
];

const STUDIO_PARAMS = [
  { name: 'X_DUT_M1M2_W',    min: 'min_nfet_w', max: 'max_nfet_w', best: '24.2µ',   bounds: '0.18µ … 200µ' },
  { name: 'X_DUT_M1M2_L',    min: 'min_nfet_l', max: 'max_nfet_l', best: '0.42µ',   bounds: '0.18µ … 10µ'  },
  { name: 'X_DUT_M1CM2C_W',  min: 'min_nfet_w', max: 'max_nfet_w', best: '18.0µ',   bounds: '0.18µ … 200µ' },
  { name: 'X_DUT_M1CM2C_L',  min: 'min_nfet_l', max: 'max_nfet_l', best: '0.36µ',   bounds: '0.18µ … 10µ'  },
  { name: 'X_DUT_M3M4_W',    min: 'min_pfet_w', max: 'max_pfet_w', best: '46.8µ',   bounds: '0.18µ … 200µ' },
  { name: 'X_DUT_M3M4_L',    min: 'min_pfet_l', max: 'max_pfet_l', best: '0.55µ',   bounds: '0.18µ … 10µ'  },
  { name: 'X_DUT_M3CM4C_W',  min: 'min_pfet_w', max: 'max_pfet_w', best: '38.4µ',   bounds: '0.18µ … 200µ' },
  { name: 'X_DUT_M3CM4C_L',  min: 'min_pfet_l', max: 'max_pfet_l', best: '0.50µ',   bounds: '0.18µ … 10µ'  },
  { name: 'X_DUT_M5_W',      min: 'min_nfet_w', max: 'max_nfet_w', best: '12.0µ',   bounds: '0.18µ … 200µ' },
  { name: 'X_DUT_M5_L',      min: 'min_nfet_l', max: 'max_nfet_l', best: '0.30µ',   bounds: '0.18µ … 10µ'  },
  { name: 'X_DUT_M5_NG',     min: '1',          max: '5',          best: '3',       bounds: '1 … 5', isInt: true },
  { name: 'X_DUT_M6_W',      min: 'min_nfet_w', max: 'max_nfet_w', best: '6.4µ',    bounds: '0.18µ … 200µ' },
  { name: 'X_DUT_M6_L',      min: 'min_nfet_l', max: 'max_nfet_l', best: '0.40µ',   bounds: '0.18µ … 10µ'  },
  { name: 'X_DUT_V_BIAS_1',  min: '0',          max: '1.2',        best: '0.62 V',  bounds: '0 … 1.2 V' },
  { name: 'X_DUT_V_BIAS_2',  min: '0',          max: '1.2',        best: '0.81 V',  bounds: '0 … 1.2 V' },
];

const STUDIO_TESTBENCHES = [
  { name: 'tb_ac',    netlist: 'spice/ota-improved_tb-loopgain.spice', enabled: true, desc: 'AC analysis for UGF, gain, PM', specs: ['ugf', 'dcgain', 'pm'] },
  { name: 'tb_noise', netlist: 'spice/ota-improved_tb-noise.spice',    enabled: true, desc: 'Noise + DC op for inoise, idd', specs: ['inoise', 'idd'] },
  { name: 'tb_tran',  netlist: 'spice/ota-improved_tb-tran.spice',     enabled: true, desc: 'Transient for settling time',   specs: ['tsettle'] },
];

const STUDIO_ALGOS = [
  { id: 'LhsDE',          label: 'LhsDE',           desc: 'Latin-hypercube seed + DE' },
  { id: 'LogBFGSCMAPlus', label: 'LogBFGSCMAPlus',  desc: 'CMA-ES + LBFGS hybrid' },
  { id: 'TinyCMA',        label: 'TinyCMA',         desc: 'Lightweight CMA' },
  { id: 'LHSSearch',      label: 'LHSSearch',       desc: 'Blind LHS baseline' },
  { id: 'ScrHaltonSearch',label: 'ScrHaltonSearch', desc: 'Scrambled Halton blind' },
  { id: 'LBFGSB',         label: 'LBFGSB',          desc: 'Gradient-free L-BFGS-B' },
];

const STUDIO_DEMOS = [
  { id: 'cascode_default', label: 'OTA Cascode (default)', path: 'examples/OTA/cascode/ihp-sg13g2/sizing/project_setup.yaml' },
  { id: 'cascode_minimal', label: 'OTA Cascode · minimal', path: 'examples/OTA/cascode/minimal.yaml' },
];

const STUDIO_REPLAYS = [
  { id: 'sigmoid_de',    label: 'sigmoid · DE',      file: 'CASCODE-OTA_LhsDE_…_sigmoid-loss.csv' },
  { id: 'linear_de',     label: 'linear · DE',       file: 'CASCODE-OTA_LhsDE_…_relAbs-loss.csv' },
  { id: 'blind_search',  label: 'blind LHS search',  file: 'CASCODE-OTA_LHSSearch_…_blind-search.csv' },
];

// Hook
function useStudio() {
  const ctx = React.useContext(StudioCtx);
  if (!ctx) throw new Error('useStudio must be used inside <StudioProvider>');
  return ctx;
}

// Provider
function StudioProvider({ children }) {
  const [state, setState] = React.useState(STUDIO_INITIAL);
  const set = React.useCallback((patch) => {
    setState((prev) => (typeof patch === 'function' ? patch(prev) : { ...prev, ...patch }));
  }, []);

  // Convenience action: switch activity AND update active tab to that activity's default
  const setActivity = React.useCallback((activity) => {
    const defaultTab = {
      runs: 'convergence',
      setup: 'yaml',
      pipeline: 'pipeline',
      schematic: 'schematic',
      specs: 'shaping',
      compare: 'compare',
    }[activity] || 'pipeline';
    set({ activeActivity: activity, activeTab: defaultTab });
  }, [set]);

  const value = { s: state, set, setActivity,
    runs: STUDIO_RUNS, specs: STUDIO_SPECS, params: STUDIO_PARAMS,
    testbenches: STUDIO_TESTBENCHES, algos: STUDIO_ALGOS, demos: STUDIO_DEMOS, replays: STUDIO_REPLAYS,
  };
  return <StudioCtx.Provider value={value}>{children}</StudioCtx.Provider>;
}

Object.assign(window, {
  StudioCtx, StudioProvider, useStudio,
  STUDIO_INITIAL, STUDIO_RUNS, STUDIO_SPECS, STUDIO_PARAMS,
  STUDIO_TESTBENCHES, STUDIO_ALGOS, STUDIO_DEMOS, STUDIO_REPLAYS,
});
