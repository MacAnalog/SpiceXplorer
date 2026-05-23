// Typed API client — all calls go to the FastAPI backend on :8000
import type {
  DemoConfig,
  LoadProjectResponse,
  ValidateResponse,
  ScoreResponse,
  RunStartResponse,
  CheckpointData,
  CheckpointMeta,
  EnvelopeEntry,
  ScatterPoint,
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
  config: () => req<DemoConfig>("/api/config"),

  // Project
  loadProject: (yaml_path: string) =>
    req<LoadProjectResponse>("/api/project/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ yaml_path }),
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

  envelope: (id: string, yaml_path?: string) =>
    req<{ envelope: EnvelopeEntry[] }>(
      `/api/checkpoint/${id}/envelope${yaml_path ? `?yaml_path=${encodeURIComponent(yaml_path)}` : ""}`,
    ).then((r) => r.envelope),

  scatter: (id: string, metric_x: string, metric_y: string, yaml_path?: string) =>
    req<{ metric_x: string; metric_y: string; points: ScatterPoint[] }>(
      `/api/checkpoint/${id}/scatter?metric_x=${encodeURIComponent(metric_x)}&metric_y=${encodeURIComponent(metric_y)}${yaml_path ? `&yaml_path=${encodeURIComponent(yaml_path)}` : ""}`,
    ),

  schematicUrl: () => `${BASE}/api/schematic`,
};
