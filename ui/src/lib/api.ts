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
  SpecLibraryResponse,
  GenerateProjectResponse,
  ParseProjectResponse,
  WizardForm,
  SensitivityResponse,
  ProjectMeta,
  ProjectDetail,
  ProjectRun,
  ExampleMeta,
} from "@/types/api";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// SSE must NOT go through the Next.js rewrite proxy: that proxy buffers
// `text/event-stream` responses (ignoring the backend's `X-Accel-Buffering: no`),
// so per-trial events arrive in one delayed burst instead of live — making a
// running optimization look frozen. Regular fetches stay same-origin (proxied) for
// portability, but the EventSource connects DIRECTLY to the backend origin.
//
//  • Explicit NEXT_PUBLIC_API_URL set → use it (already a direct backend origin).
//  • Same-origin mode (empty BASE) → derive the backend origin from the current
//    page host + the backend port, so it works wherever the browser runs
//    (localhost, a LAN IP, an SSH-forwarded host). CORS allows any localhost:<port>.
const BACKEND_PORT = process.env.NEXT_PUBLIC_BACKEND_PORT ?? "8000";

function streamBase(): string {
  if (BASE) return BASE;
  if (typeof window === "undefined") return "";
  return `${window.location.protocol}//${window.location.hostname}:${BACKEND_PORT}`;
}

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

  // Apply edited/uploaded YAML. Pass yaml_path when the content was edited from a
  // loaded file so the backend anchors relative ws_root/netlist resolution to the
  // original directory (the applied YAML is persisted to a temp file).
  loadProjectContent: (yaml_content: string, yaml_path?: string) =>
    req<LoadProjectResponse>("/api/project/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ yaml_content, yaml_path }),
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
    opts: {
      selectedSpec?: string;
      nCurvePoints?: number;
      /** Ephemeral per-spec edits (what-if); never written to YAML. */
      specOverrides?: Record<string, Record<string, string | number | boolean>>;
    } = {},
  ) =>
    req<ScoreResponse>("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        yaml_path,
        metric_values,
        selected_spec: opts.selectedSpec,
        n_curve_points: opts.nCurvePoints ?? 200,
        spec_overrides: opts.specOverrides,
      }),
    }),

  // Optimization run
  startRun: (body: {
    yaml_path?: string;
    /** Owning project — the run is isolated under its runs/ dir (report.md P3). */
    project_id?: string;
    label?: string;
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

  streamUrl: (run_id: string) => `${streamBase()}/api/optimize/stream/${run_id}`,

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

  // URL for the downloadable run-report zip (checkpoint + YAML + summary.md).
  reportUrl: (checkpointId: string, yamlPath?: string) => {
    const q = new URLSearchParams();
    if (yamlPath) q.set("yaml_path", yamlPath);
    const qs = q.toString();
    return `${BASE}/api/checkpoint/${encodeURIComponent(checkpointId)}/report${qs ? `?${qs}` : ""}`;
  },

  sanityCheck: (yaml_path: string, active_corner?: string) =>
    req<SanityCheckResponse>("/api/sanity-check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ yaml_path, active_corner }),
    }),

  // Manual single simulation — evaluate ONE chosen design point (live SPICE — needs PDK).
  // Mode B: pass `params` (engineering-real). Mode A: pass `checkpoint_id` (+ optional
  // `point`; omitted → best). `active_corner` optionally overrides the PVT corner.
  simulateOnce: (body: {
    yaml_path: string;
    // Values may be eng-strings ("250u") or numbers; parsed server-side.
    params?: Record<string, string | number>;
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

  // Shipped analog-spec templates for the wizard's one-click "Spec library".
  specLibrary: () => req<SpecLibraryResponse>("/api/spec-library"),

  // Projects (report.md P3) — the registry IS WORK_ROOT/projects/.
  listProjects: () => req<{ projects: ProjectMeta[] }>("/api/projects"),
  listExamples: () => req<{ examples: ExampleMeta[] }>("/api/examples"),
  createProject: (name: string, yaml_content?: string) =>
    req<{ id: string }>("/api/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, yaml_content }),
    }),
  fromExample: (example_key: string, name?: string) =>
    req<{ id: string }>("/api/projects/from-example", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ example_key, name }),
    }),
  getProject: (id: string) => req<ProjectDetail>(`/api/projects/${encodeURIComponent(id)}`),
  getProjectRuns: (id: string) =>
    req<{ runs: ProjectRun[] }>(`/api/projects/${encodeURIComponent(id)}/runs`),

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
