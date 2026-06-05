// Typed API client — all calls go to the FastAPI backend on :8000
import type {
  AppConfig,
  LoadProjectResponse,
  ValidateResponse,
  ScoreResponse,
  RunStartResponse,
  CheckpointData,
  CheckpointMeta,
  EnvInfo,
  EnvelopeEntry,
  ScatterPoint,
  SanityCheckResponse,
  SimulateOnceResponse,
  NetlistParseResponse,
  GenerateProjectResponse,
  ParseProjectResponse,
  WizardForm,
  SensitivityResponse,
} from "@/types/api";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    let msg = `API error ${res.status}`;
    try {
      const body = await res.json();
      msg = body.detail ?? body.error ?? msg;
    } catch {}
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  // Config
  config: () => req<AppConfig>("/api/config"),

  // Environment — ngspice + IHP PDK probe for graceful degradation (no live runs without PDK)
  env: () => req<EnvInfo>("/api/env"),

  // Project
  loadProject: (yaml_path: string) =>
    req<LoadProjectResponse>("/api/project/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ yaml_path }),
    }),

  // Apply edited/uploaded YAML that has no on-disk path (returns yaml_path: "").
  loadProjectContent: (yaml_content: string) =>
    req<LoadProjectResponse>("/api/project/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ yaml_content }),
    }),

  validateYaml: (yaml_content: string) =>
    req<ValidateResponse>("/api/project/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ yaml_content }),
    }),

  // Score shaping
  computeScore: (
    yaml_path: string,
    metric_values: Record<string, number>,
    selected_spec?: string,
    n_curve_points = 200,
  ) =>
    req<ScoreResponse>("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ yaml_path, metric_values, selected_spec, n_curve_points }),
    }),

  // Optimization run
  startRun: (body: {
    yaml_path?: string;
    replay?: boolean;
    checkpoint_id?: string;
    budget?: number;
    /** Ephemeral live-run overrides (ignored for replay). */
    algorithm?: string;
    seed?: number;
    /** PVT corner to optimize against (must match a corner in the project's `pvt:`). */
    active_corner?: string;
    /** Autosave a cumulative checkpoint every N trials (live only). */
    autosave_every?: number;
    /** Resume a live run from a saved checkpoint (load + keep_history). */
    resume_checkpoint_id?: string;
  }) =>
    req<RunStartResponse>("/api/optimize/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  stopRun: (run_id: string) =>
    req<{ ok: boolean }>(`/api/optimize/stop/${run_id}`, { method: "POST" }),

  streamUrl: (run_id: string) => `${BASE}/api/optimize/stream/${run_id}`,

  // Checkpoints
  listCheckpoints: () =>
    req<{ checkpoints: CheckpointMeta[] }>("/api/checkpoint").then(
      (r) => r.checkpoints,
    ),

  loadCheckpoint: (id: string, limit = 0) =>
    req<CheckpointData>(`/api/checkpoint/${id}?limit=${limit}`),

  deleteCheckpoint: (id: string) =>
    req<{ ok: boolean; deleted: string[] }>(`/api/checkpoint/${encodeURIComponent(id)}`, {
      method: "DELETE",
    }),

  envelope: (id: string, yaml_path?: string) =>
    req<{ envelope: EnvelopeEntry[] }>(
      `/api/checkpoint/${id}/envelope${yaml_path ? `?yaml_path=${encodeURIComponent(yaml_path)}` : ""}`,
    ).then((r) => r.envelope),

  scatter: (id: string, metric_x: string, metric_y: string, yaml_path?: string) =>
    req<{ metric_x: string; metric_y: string; points: ScatterPoint[] }>(
      `/api/checkpoint/${id}/scatter?metric_x=${encodeURIComponent(metric_x)}&metric_y=${encodeURIComponent(metric_y)}${yaml_path ? `&yaml_path=${encodeURIComponent(yaml_path)}` : ""}`,
    ),

  schematicUrl: () => `${BASE}/api/schematic`,

  sanityCheck: (yaml_path: string) =>
    req<SanityCheckResponse>("/api/sanity-check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ yaml_path }),
    }),

  // Manual single simulation — evaluate ONE chosen design point (live SPICE — needs PDK).
  // Mode B: pass `params` (engineering-real). Mode A: pass `checkpoint_id` (+ optional
  // `point`; omitted → best). `active_corner` optionally overrides the PVT corner.
  simulateOnce: (body: {
    yaml_path: string;
    params?: Record<string, number>;
    checkpoint_id?: string;
    point?: number;
    active_corner?: string;
  }) =>
    req<SimulateOnceResponse>("/api/simulate/once", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  // Finite-difference sensitivity of one spec to DUT params (live SPICE — needs PDK).
  // `params` scopes the sweep (e.g. one device's W/L); `at` overrides the baseline
  // operating point (absolute SI) so the inspector's sliders define the design point.
  specSensitivity: (
    spec: string,
    opts: {
      yaml_path?: string;
      params?: string[];
      at?: Record<string, number>;
      rel_delta?: number;
    } = {},
  ) => {
    const q = new URLSearchParams();
    if (opts.yaml_path) q.set("yaml_path", opts.yaml_path);
    if (opts.params?.length) q.set("params", opts.params.join(","));
    if (opts.at && Object.keys(opts.at).length)
      q.set("at", Object.entries(opts.at).map(([k, v]) => `${k}:${v}`).join(","));
    if (opts.rel_delta != null) q.set("rel_delta", String(opts.rel_delta));
    const qs = q.toString();
    return req<SensitivityResponse>(
      `/api/spec/${encodeURIComponent(spec)}/sensitivity${qs ? `?${qs}` : ""}`,
    );
  },

  // Wizard
  parseNetlist: async (file: File): Promise<NetlistParseResponse> => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(`${BASE}/api/netlist/parse`, { method: "POST", body: fd });
    if (!res.ok) {
      let msg = `API error ${res.status}`;
      try { const b = await res.json(); msg = b.detail ?? msg; } catch {}
      throw new Error(msg);
    }
    return res.json();
  },

  generateProject: (form: WizardForm, save_path?: string) =>
    req<GenerateProjectResponse>("/api/project/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ form, save_path: save_path ?? null }),
    }),

  parseProjectToForm: (args: { yaml_path?: string; yaml_content?: string }) =>
    req<ParseProjectResponse>("/api/project/parse-to-form", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(args),
    }),

  // Xschem viewer
  xschemFile: (path: string) =>
    req<{ path: string; content: string }>(
      `/api/xschem/file?path=${encodeURIComponent(path)}`,
    ),

  xschemResolve: (ref: string, base?: string) => {
    const q = `ref=${encodeURIComponent(ref)}${base ? `&base=${encodeURIComponent(base)}` : ""}`;
    return req<{ path: string; content: string; resolved_from: string }>(
      `/api/xschem/resolve?${q}`,
    );
  },

  xschemProject: (yaml_path: string) =>
    req<{ xschem_dir: string | null; files: { path: string; name: string }[] }>(
      `/api/xschem/project?yaml_path=${encodeURIComponent(yaml_path)}`,
    ),
};
