// Mock data for SpiceXplorer UI design
// Realistic cascode OTA optimization run data

// === Project / DUT ===
const PROJECT = {
  name: "cascode-ota-sg13g2",
  description: "Cascode OTA for low-noise sensor frontend",
  simulator: "ngspice 41",
  technology: "IHP SG13G2",
  workspace: "/work/spicexplorer/examples/OTA/cascode",
  yamlPath: "examples/OTA/cascode/ihp-sg13g2/sizing/project_setup.yaml",
  status: "applied",
};

const DUT_PARAMS = [
  { name: "w_n1",  min: "1.0µ",  max: "200µ", val: "62.4µ", log: true,  int: false, frozen: false },
  { name: "l_n1",  min: "130n",  max: "2.0µ", val: "180n",  log: false, int: false, frozen: false },
  { name: "w_n2",  min: "1.0µ",  max: "200µ", val: "48.1µ", log: true,  int: false, frozen: false },
  { name: "l_n2",  min: "130n",  max: "2.0µ", val: "320n",  log: false, int: false, frozen: false },
  { name: "w_p1",  min: "1.0µ",  max: "200µ", val: "112µ",  log: true,  int: false, frozen: false },
  { name: "l_p1",  min: "130n",  max: "2.0µ", val: "240n",  log: false, int: false, frozen: false },
  { name: "w_p2",  min: "1.0µ",  max: "200µ", val: "96.0µ", log: true,  int: false, frozen: false },
  { name: "l_p2",  min: "130n",  max: "2.0µ", val: "400n",  log: false, int: false, frozen: false },
  { name: "i_bias",min: "1µ",    max: "200µ", val: "42µ",   log: true,  int: false, frozen: false },
  { name: "m_mult",min: "1",     max: "8",    val: "2",     log: false, int: true,  frozen: false },
  { name: "c_load",min: "100f",  max: "10p",  val: "2.0p",  log: true,  int: false, frozen: true },
];

const TESTBENCHES = [
  { name: "ac_response",   netlist: "tb_ac.spice",     params: 3, enabled: true },
  { name: "dc_operating",  netlist: "tb_dc.spice",     params: 1, enabled: true },
  { name: "transient_step",netlist: "tb_tran.spice",   params: 2, enabled: true },
  { name: "noise_input",   netlist: "tb_noise.spice",  params: 1, enabled: false },
];

const PVT = [
  { temp: -40, corner: "ff", supply: 1.32 },
  { temp:  27, corner: "tt", supply: 1.20 },
  { temp: 125, corner: "ss", supply: 1.08 },
];

// === Specs (the headline metrics) ===
const SPECS = [
  { name: "ugf",     tb: "ac_response",   goal: ">", target: 200e6, unit: "Hz",   tol: 10e6,  range: 50e6,  weight: 1.0, current: 187.4e6, met: false, sigP: 0.34, linP: 0.06 },
  { name: "gain",    tb: "ac_response",   goal: ">", target: 40,    unit: "dB",   tol: 2,     range: 10,    weight: 1.0, current: 43.8,    met: true,  sigP: 0.04, linP: 0.00 },
  { name: "pm",      tb: "ac_response",   goal: ">", target: 60,    unit: "°",    tol: 5,     range: 15,    weight: 0.8, current: 58.2,    met: false, sigP: 0.12, linP: 0.12 },
  { name: "current", tb: "dc_operating",  goal: "<", target: 100e-6,unit: "A",    tol: 5e-6,  range: 30e-6, weight: 0.6, current: 84e-6,   met: true,  sigP: 0.02, linP: 0.00 },
  { name: "sr_pos",  tb: "transient_step",goal: ">", target: 50e6,  unit: "V/s",  tol: 5e6,   range: 20e6,  weight: 0.7, current: 61.2e6,  met: true,  sigP: 0.03, linP: 0.00 },
  { name: "vno_int", tb: "noise_input",   goal: "<", target: 12e-6, unit: "Vrms", tol: 1e-6,  range: 4e-6,  weight: 0.5, current: 14.1e-6, met: false, sigP: 0.18, linP: 0.21 },
];

// === Optimization trajectories ===
// Iteration-by-iteration data — used for convergence charts.
function makeTrajectory(seed, totalIters, finalScore, shape) {
  // deterministic-ish PRNG
  let s = seed;
  const rnd = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
  const raw = [];
  const best = [];
  let bestSoFar = Infinity;
  for (let i = 0; i < totalIters; i++) {
    const t = i / totalIters;
    let base;
    if (shape === "sigmoid") {
      base = 6 * Math.exp(-i / 35) + finalScore;
    } else if (shape === "linear") {
      base = 4 * Math.exp(-i / 60) + finalScore + 0.5;
    } else {
      base = 5.5 - 1.6 * t + 0.4 * Math.sin(i * 0.18);
    }
    const noise = (rnd() - 0.45) * (2.5 * Math.exp(-i / 40) + 0.4);
    const v = Math.max(0.05, base + noise);
    raw.push(v);
    if (v < bestSoFar) bestSoFar = v;
    best.push(bestSoFar);
  }
  return { raw, best };
}

const TRAJ_SIGMOID = makeTrajectory(7, 200, 0.42, "sigmoid");
const TRAJ_LINEAR  = makeTrajectory(13, 200, 1.18, "linear");
const TRAJ_BLIND   = makeTrajectory(29, 200, 2.4,  "blind");

// Per-metric convergence (best-so-far)
function metricTrajectory(targetVal, startScale, finalGap, totalIters, seed) {
  let s = seed;
  const rnd = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
  const out = [];
  let bestErr = Infinity;
  for (let i = 0; i < totalIters; i++) {
    const t = i / totalIters;
    const candidate = targetVal * (1 + (startScale * Math.exp(-i/40) + finalGap) * (rnd() - 0.4) * 2);
    const err = Math.abs(candidate - targetVal);
    if (err < bestErr) bestErr = err;
    // best-so-far value approaches target from one side
    const sign = seed % 2 === 0 ? 1 : -1;
    const approach = targetVal + sign * bestErr * (1 - 0.3 * t);
    out.push(approach);
  }
  return out;
}

const METRIC_TRAJ = {
  ugf:     { sigmoid: metricTrajectory(200e6, 0.6, 0.05, 200, 4),  linear: metricTrajectory(200e6, 0.6, 0.18, 200, 6) },
  gain:    { sigmoid: metricTrajectory(40,    0.4, 0.02, 200, 8),  linear: metricTrajectory(40,    0.4, 0.04, 200, 10) },
  pm:      { sigmoid: metricTrajectory(60,    0.3, 0.03, 200, 12), linear: metricTrajectory(60,    0.3, 0.08, 200, 14) },
  current: { sigmoid: metricTrajectory(100e-6,0.5, 0.10, 200, 16), linear: metricTrajectory(100e-6,0.5, 0.05, 200, 18) },
};

// === Scatter data (UGF vs current) ===
function makeScatter(seed, n, ugfMean, currentMean) {
  let s = seed;
  const rnd = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
  const pts = [];
  for (let i = 0; i < n; i++) {
    const u = ugfMean * (0.4 + 1.8 * rnd());
    const c = currentMean * (0.3 + 2.0 * rnd());
    const feasible = u >= 200e6 && c <= 100e-6;
    pts.push({ x: u, y: c, feasible });
  }
  return pts;
}
const SCATTER_A = makeScatter(101, 200, 220e6, 95e-6);
const SCATTER_B = makeScatter(202, 200, 160e6, 110e-6);

// === Histogram ===
function makeHist(seed, n, mean, sd) {
  let s = seed;
  const rnd = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
  const out = [];
  for (let i = 0; i < n; i++) {
    // Box-Muller
    const u1 = rnd() || 0.001, u2 = rnd();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    out.push(mean + z * sd);
  }
  return out;
}
const HIST_A_UGF = makeHist(33, 200, 215e6, 35e6);
const HIST_B_UGF = makeHist(77, 200, 165e6, 45e6);

// === Penalty curves ===
// For ugf spec, sweeping value across target ± 3*range
function penaltyCurves(spec) {
  const lo = spec.target - 3 * spec.range;
  const hi = spec.target + 3 * spec.range;
  const n = 120;
  const xs = [];
  const sig = [];
  const lin = [];
  for (let i = 0; i < n; i++) {
    const x = lo + (hi - lo) * (i / (n - 1));
    const dir = spec.goal === ">" ? (spec.target - x) : (x - spec.target);
    // sigmoid: smooth, bounded by 1, sharp drop near target
    const sigVal = 1 / (1 + Math.exp(-dir / (spec.range * 0.35)));
    // linear: clamped relative absolute error
    const linVal = Math.max(0, Math.min(1.0, dir / spec.range));
    xs.push(x);
    sig.push(sigVal);
    lin.push(linVal);
  }
  return { xs, sig, lin };
}

// === Checkpoints (Explorer) ===
const CHECKPOINTS = [
  { id: "sigmoid_de",   label: "LhsDE · sigmoid",      iters: 200, bestScore: 0.42, when: "2026-02-07 10:54" },
  { id: "linear_de",    label: "LhsDE · relAbs",       iters: 200, bestScore: 1.18, when: "2026-02-07 14:53" },
  { id: "blind_search", label: "LHSSearch · blind",    iters: 200, bestScore: 2.40, when: "2026-02-07 10:53" },
  { id: "cma_plus",     label: "LogBFGSCMAPlus · sig", iters:  84, bestScore: 0.51, when: "2026-02-08 09:12" },
];

// Performance envelope (best achievable per spec across all evaluated points)
const ENVELOPE = [
  { spec: "ugf",     unit: "MHz",   bestA: 248.1, bestB: 196.4 },
  { spec: "gain",    unit: "dB",    bestA: 47.2,  bestB: 41.8  },
  { spec: "pm",      unit: "°",     bestA: 72.1,  bestB: 64.0  },
  { spec: "current", unit: "µA",    bestA: 64.0,  bestB: 91.2  },
  { spec: "sr_pos",  unit: "MV/s",  bestA: 84.3,  bestB: 52.6  },
  { spec: "vno_int", unit: "µVrms", bestA: 9.4,   bestB: 13.8  },
];

// Live run state (Optimize tab)
const RUN_STATE = {
  algorithm: "LhsDE",
  budget: 1000,
  iter: 147,
  isRunning: true,
  isReplay: true,
  source: "sigmoid_de",
  scoreFn: "sigmoid",
  bestScore: 0.62,
  bestParams: DUT_PARAMS.slice(0, 8),
};

// Expose to window for cross-script access
window.SPICE_DATA = {
  PROJECT, DUT_PARAMS, TESTBENCHES, PVT, SPECS,
  TRAJ_SIGMOID, TRAJ_LINEAR, TRAJ_BLIND, METRIC_TRAJ,
  SCATTER_A, SCATTER_B, HIST_A_UGF, HIST_B_UGF,
  penaltyCurves, CHECKPOINTS, ENVELOPE, RUN_STATE,
};

// === Formatting helpers ===
function formatEng(v, digits = 3) {
  if (v === 0) return "0";
  const abs = Math.abs(v);
  const units = [
    { e: 1e9,  s: "G" }, { e: 1e6,  s: "M" }, { e: 1e3,  s: "k" },
    { e: 1,    s: ""  }, { e: 1e-3, s: "m" }, { e: 1e-6, s: "µ" },
    { e: 1e-9, s: "n" }, { e: 1e-12,s: "p" }, { e: 1e-15,s: "f" },
  ];
  for (const u of units) {
    if (abs >= u.e * 0.999) {
      const scaled = v / u.e;
      const intDigits = Math.floor(Math.log10(Math.abs(scaled))) + 1;
      const decimals = Math.max(0, digits - intDigits);
      let str = scaled.toFixed(decimals);
      if (str.includes(".")) str = str.replace(/0+$/, "").replace(/\.$/, "");
      return str + u.s;
    }
  }
  return v.toExponential(digits - 1);
}

function formatVal(v, unit) {
  if (unit === "dB" || unit === "°") return v.toFixed(1) + unit;
  return formatEng(v) + unit;
}

window.SPICE_FMT = { formatEng, formatVal };
