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

// Config

export interface DemoConfig {
  default_yaml: string;
  demo_checkpoints: CheckpointMeta[];
  schematic_svg_path: string;
}
