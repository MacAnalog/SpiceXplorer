// API response types — mirror FastAPI response schemas

export interface DutParam {
  name: string;
  min_val: number | null;
  max_val: number | null;
  /** Explicit operating-point value (matches the backend sensitivity nominal). */
  val?: number | null;
  init?: number | null;
  is_integer: boolean;
  log_scale: boolean;
  freeze: boolean;
}

export interface TbParam {
  name: string;
  val: string | null;
  description: string | null;
}

export interface Testbench {
  name: string;
  netlist: string;
  enable: boolean;
  description: string | null;
  params: TbParam[];
}

// PVT corner system (Phase 1) — the simulator-driving config (`pvt:` block).

export interface ModelInclude {
  lib_file: string;
  section: string;
}

export interface SupplyOverride {
  node: string;
  value: number;
}

export interface PVTCornerDef {
  name: string;
  enabled: boolean;
  temp: number;
  supplies: SupplyOverride[];
  model_includes: ModelInclude[];
  params: Record<string, number>;
}

export interface PVTConfig {
  active_corner: string;
  model_lib_root: string | null;
  corners: PVTCornerDef[];
}

export interface TargetSpec {
  name: string;
  testbench: string;
  goal: "exceed" | "minimize" | "exact";
  target: number;
  tolerance: number | null;
  range: number | null;
  weight: number;
  error_type: string;
  reward_type: string;
  enable: boolean;
  description: string | null;
}

export interface ParamSensitivity {
  name: string;
  device: string;
  kind: string; // W | L | NG | bias | other
  nominal: number;
  delta: number;
  perturbed_value: number;
  baseline_metric: number | null;
  perturbed_metric: number | null;
  /** Absolute slope d(metric)/d(param). */
  sensitivity: number | null;
  /** Dimensionless elasticity (dmetric/metric)/(dparam/param). */
  elasticity: number | null;
  ok: boolean;
  note: string | null;
}

export interface SensitivityResponse {
  ok: boolean;
  spec: string;
  testbench: string | null;
  goal: string | null;
  target: number | null;
  baseline_value: number | null;
  baseline_score: number | null;
  rel_delta: number;
  n_sims: number;
  params: ParamSensitivity[];
  error: string | null;
  elapsed_ms: number | null;
}

export interface ProjectSummary {
  name: string;
  description: string;
  simulator: string;
  ws_root: string;
  netlist: string;
  /** Optional pointer to the design's .sch (relative to `ws_root`). */
  schematic?: string | null;
  tech: { name: string; constraints: Record<string, number> };
  /** PVT corner system (Phase 1). `null` when the project has no `pvt:` block. */
  pvt: PVTConfig | null;
  dut_params: DutParam[];
  testbenches: Testbench[];
  optimizer: {
    type: string;
    name: string;
    budget: number;
    random_seed: number | null;
  };
  target_specs: TargetSpec[];
}

export interface LoadProjectResponse {
  ok: boolean;
  summary: ProjectSummary;
  yaml_path: string;
}

export interface ValidateResponse {
  ok: boolean;
  errors: string[];
}

// Score shaping

export interface SpecScore {
  linear: number | null;
  sigmoid: number | null;
  value: number | null;
  target: number;
  tolerance: number;
  goal: string;
  passes: boolean | null;
  weight: number;
}

export interface ScoreCurve {
  values: number[];
  linear: number[];
  sigmoid: number[];
  target: number;
  tolerance: number;
  goal: string;
}

export interface ScoreResponse {
  per_spec: Record<string, SpecScore>;
  aggregate: { linear: number; sigmoid: number };
  curve: ScoreCurve | null;
}

// Optimization run

export interface RunStartResponse {
  run_id: string;
  replay: boolean;
  resumed?: boolean;
  /** Row count of the replayed checkpoint, so progress shows iter/length. */
  n_iters?: number | null;
}

/** Emitted when a live run autosaves a (cumulative) checkpoint. */
export interface CheckpointEvent {
  id: string | null;
  index: number;
  iter: number;
}

export interface SSEEvent {
  iter?: number;
  score?: number | null;
  best_score?: number | null;
  metrics?: Record<string, number | null>;
  best_params?: Record<string, number | null>;
  /** Snapshot of the BEST trial's metrics (live runs) — distinct from `metrics`, the
   *  latest trial. The right-rail / Pipeline key their pass-fail off this. */
  best_metrics?: Record<string, number | null>;
  checkpoint?: CheckpointEvent;
  done?: boolean;
  error?: string;
  heartbeat?: boolean;
}

// Checkpoints

export interface CheckpointMeta {
  id: string;
  label: string;
  path: string;
  type: "csv" | "json";
  score_fn: string;
  n_iters?: number | null;
  /** "preset" (config-defined, read-only) or "autosave" (deletable). */
  source?: "preset" | "autosave";
}

export interface CheckpointData {
  id: string;
  label: string;
  type: "csv" | "json";
  score_fn: string;
  scores: (number | null)[];
  best_scores: (number | null)[];
  iterations: number[];
  per_metric: Record<string, (number | null)[]>;
  params: Record<string, (number | null)[]>;
  n_iters: number;
}

export interface EnvelopeEntry {
  metric: string;
  best_ever: number;
  target: number | null;
  goal: string;
  passes: boolean | null;
}

export interface ScatterPoint {
  x: number;
  y: number;
  feasible: boolean;
  score: number | null;
  iter: number;
}

// Sanity check

export interface SanityTestbenchResult {
  name: string;
  ok: boolean;
  error: string | null;
  elapsed_ms: number | null;
  log_path: string | null;
  log_tail: string | null;
  log_size_bytes: number | null;
}

export interface SanityTrialResult {
  ok: boolean;
  score: number | null;
  metrics: Record<string, number | null>;
  error: string | null;
  elapsed_ms: number | null;
  log_files: Record<string, string>;
  log_tails: Record<string, string>;
}

export interface SanityCheckResponse {
  ok: boolean;
  testbenches: SanityTestbenchResult[];
  trial: SanityTrialResult | null;
  error: string | null;
  elapsed_ms_total: number | null;
  elapsed_ms_load: number | null;
  elapsed_ms_optimizer_init: number | null;
  ngspice_path: string | null;
  /** Whether the IHP PDK device models resolved (false on a PDK-less machine). */
  pdk_ok: boolean | null;
  /** Human-readable PDK verdict for the diagnostics panel. */
  pdk_detail: string | null;
  /** PVT corner the trial ran at (null when the project has no `pvt:` block). */
  active_corner: string | null;
}

// Manual single simulation — evaluate ONE chosen design point (POST /api/simulate/once)

export interface SimulateSpecMetric {
  curr_val: number | null;
  score: number | null;
}

export interface SimulateOnceResponse {
  ok: boolean;
  score: number | null;
  /** Per-spec measured value + per-spec score, keyed by spec name. */
  metrics: Record<string, SimulateSpecMetric>;
  /** The engineering-real param vector actually injected. */
  params_used: Record<string, number>;
  /** PVT corner the point was evaluated at (null if the project has no `pvt:`). */
  active_corner: string | null;
  /** Non-fatal advisories (out-of-range value, unknown param, unknown corner, …). */
  warnings: string[];
  log_files: Record<string, string>;
  log_tails: Record<string, string>;
  error: string | null;
  elapsed_ms: number | null;
  elapsed_ms_load: number | null;
  pdk_ok: boolean | null;
  pdk_detail: string | null;
}

// Environment probe — simulator + PDK availability (GET /api/env)

export interface EnvInfo {
  ngspice_path: string | null;
  ngspice_ok: boolean;
  pdk_root: string | null;
  pdk_ok: boolean;
  pdk_detail: string;
  tech: string;
  /** True only when both ngspice and the PDK are present; gates live optimization. */
  live_runs_enabled: boolean;
}

// Config

export interface AppConfig {
  default_yaml: string;
  preset_checkpoints: CheckpointMeta[];
  schematic_svg_path: string;
}

// Wizard

export interface NetlistParam {
  name: string;
  default_val: string;
}

export interface NetlistParseResponse {
  ok: boolean;
  filename: string;
  params: NetlistParam[];
}

export interface GenerateProjectResponse {
  ok: boolean;
  yaml: string;
  errors: string[];
  saved_path: string | null;
}

export interface ParseProjectResponse {
  ok: boolean;
  form: WizardForm;
}

// Wizard form shapes — mirrors yaml_generator.py expectations
export interface WizardProjectInfo {
  name: string;
  description: string;
  simulator: string;
  save_sim: boolean;
  parallel_sim: boolean;
  ws_root: string;
  netlist: string;
  outdir: string;
}

export interface ConstraintRow { key: string; value: string }

export interface WizardTech {
  name: string;
  constraints: ConstraintRow[];
}

// New PVT corner system (Phase 1) — wizard editing shapes. Corners are emitted
// with inline `model_includes` (no process_bundles) to keep the wizard flat.
export interface WizardModelInclude {
  lib_file: string;
  section: string;
}

export interface WizardPVTCorner {
  name: string;
  temp: string;
  supply_node: string;
  supply_value: string;
  /** Rails 2..N, carried losslessly through the wizard (the form edits only the primary
   *  rail; multi-rail corners are authored/edited in the raw YAML). */
  extra_supplies?: { node: string; value: string }[];
  enabled: boolean;
  includes: WizardModelInclude[];
}

export interface WizardPVTConfig {
  active_corner: string;
  corners: WizardPVTCorner[];
  /** Optional root prepended to each corner include's lib_file at sim time. */
  model_lib_root?: string;
}

export interface WizardDutParam {
  name: string;
  default_val?: string; // from netlist, displayed only
  min_val: string;
  max_val: string;
  init?: string;
  is_integer: boolean;
  log_scale: boolean;
  freeze: boolean;
  source?: "netlist" | "manual";
}

export interface WizardTbParam {
  name: string;
  val: string;
  description?: string;
  source?: "netlist" | "manual";
}

export interface WizardTestbench {
  name: string;
  netlist: string;
  enable: boolean;
  description: string;
  params: WizardTbParam[];
}

export interface WizardTargetSpec {
  name: string;
  testbench: string;
  sim_type: "ac" | "dc" | "op" | "tran" | "noise" | "noise_spectrum";
  goal: "exceed" | "minimize" | "exact";
  target: string;
  range: string;
  tolerance: string;
  weight: string;
  log_scale: boolean;
  error_type: string;
  reward_type: string;
  enable: boolean;
  description: string;
}

export interface OptimizerKwargRow { key: string; value: string }

export interface WizardOptimizer {
  type: string;
  name: string;
  budget: number | string;
  random_seed: number | string;
  lin_min: string;
  lin_max: string;
  log_min: string;
  log_max: string;
  optimizer_kwargs: OptimizerKwargRow[];
}

export interface WizardForm {
  project: WizardProjectInfo;
  tech: WizardTech;
  /** PVT corner system (Phase 1) — the simulator-driving config. */
  pvt: WizardPVTConfig;
  dut_params: WizardDutParam[];
  testbenches: WizardTestbench[];
  target_specs: WizardTargetSpec[];
  optimizer: WizardOptimizer;
}
