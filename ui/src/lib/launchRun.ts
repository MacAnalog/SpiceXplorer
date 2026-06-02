// Shared "start a live optimization run" flow, used by both the title-bar Run
// popover and the Optimize toolbar so the launch logic (and the algorithm/seed
// overrides) live in exactly one place. Reads the current stores via getState()
// — no hook, so it can be called from event handlers anywhere.
import { api } from "@/lib/api";
import { useProjectStore } from "@/stores/projectStore";
import { useRunStore } from "@/stores/runStore";
import { useUIStore } from "@/stores/uiStore";

export interface LaunchResult {
  ok: boolean;
  error?: string;
}

/**
 * Start a live SPICE run using the shared run config (algorithm/budget/seed).
 * Validates that a project is applied and the PDK is present, sends the
 * overrides to the backend, and opens the SSE stream via runStore. Returns a
 * result instead of throwing so callers can surface the error inline.
 */
export async function launchLiveRun(): Promise<LaunchResult> {
  const { yamlPath, isApplied } = useProjectStore.getState();
  const { env, runConfig } = useUIStore.getState();
  const { startRun } = useRunStore.getState();

  if (!isApplied) {
    return { ok: false, error: "Apply a project on Setup before starting a live run." };
  }
  if (env && !env.live_runs_enabled) {
    return {
      ok: false,
      error: "PDK missing — live runs are disabled. Use Replay on the Optimize view.",
    };
  }

  try {
    const res = await api.startRun({
      yaml_path: yamlPath || undefined,
      budget: runConfig.budget,
      algorithm: runConfig.algorithm,
      seed: runConfig.seed ?? undefined,
    });
    startRun(res.run_id, res.replay, runConfig.budget, {
      kind: "live",
      label: `Live · ${runConfig.algorithm}`,
    });
    return { ok: true };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : "Failed to start run" };
  }
}
