// API response types — mirror FastAPI response schemas

export interface DutParam {
  name: string;
  min_val: number | null;
  max_val: number | null;
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

export interface PVTCorner {
  temp: number;
  corner: string;
  supply: number;
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

export interface ProjectSummary {
  name: string;
  description: string;
  simulator: string;
  ws_root: string;
  netlist: string;
  /** Optional pointer to the design's .sch (relative to `ws_root`). */
  schematic?: string | null;
  tech: { name: string; constraints: Record<string, number> };
  pvt_corners: PVTCorner[];
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
}

export interface SSEEvent {
  iter?: number;
  score?: number | null;
  best_score?: number | null;
  metrics?: Record<string, number | null>;
  best_params?: Record<string, number | null>;
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

export interface WizardPVT {
  name: string;
  temp: string | number;
  corner: string;
  supply: string | number;
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
  pvt_corners: WizardPVT[];
  dut_params: WizardDutParam[];
  testbenches: WizardTestbench[];
  target_specs: WizardTargetSpec[];
  optimizer: WizardOptimizer;
}
