import { create } from "zustand";
import type { AppConfig, EnvInfo } from "@/types/api";

/**
 * Ephemeral live-run overrides shared by the title-bar Run popover and the
 * Optimize toolbar. Sent to POST /api/optimize/start and applied in-memory by
 * the backend (optimizer_runner._apply_overrides) — the YAML on disk is never
 * rewritten. `seed: null` means "let the optimizer pick" (no override sent).
 */
export interface RunConfig {
  algorithm: string;
  budget: number;
  seed: number | null;
  /** Autosave a cumulative checkpoint every N trials (null = end-only). */
  autosaveEvery: number | null;
}

export const DEFAULT_RUN_CONFIG: RunConfig = {
  algorithm: "LhsDE",
  budget: 200,
  seed: null,
  autosaveEvery: null,
};

/** Selectable Nevergrad algorithms (canonical list for the Run config UI). */
export const RUN_ALGORITHMS = ["LhsDE", "LHSSearch", "LogBFGSCMAPlus"] as const;

/**
 * Studio UI store — owns cross-view *navigation/selection* and shared
 * session data (app config, environment probe). Route (the URL) owns "which
 * view"; this store owns selection that deep-links *into* a view plus overlay
 * flags. Phase 1 wires appConfig + env; the selection/overlay fields are the
 * seed for the run-history, score-shaping deep-link, and ⌘K work in later phases.
 */
interface UIStore {
  // Shared session data (hydrated once by the (studio) layout)
  appConfig: AppConfig | null;
  env: EnvInfo | null;

  // Cross-view selection (deep-link targets — consumed in later phases)
  selectedSpec: string | null;
  selectedRunId: string | null;

  // Panels (always-on shell surfaces)
  rightOpen: boolean;
  bottomOpen: boolean;
  bottomTab: "log";

  // Overlays
  commandOpen: boolean;
  wizardOpen: boolean;

  // Shared live-run overrides (Run popover ⇄ Optimize toolbar)
  runConfig: RunConfig;

  setAppConfig: (cfg: AppConfig | null) => void;
  setEnv: (env: EnvInfo | null) => void;
  setSelectedSpec: (name: string | null) => void;
  /** Deep-link: focus a run in the history/convergence surfaces. */
  openRun: (id: string) => void;
  toggleRight: () => void;
  toggleBottom: () => void;
  setBottomTab: (tab: "log") => void;
  openCommand: () => void;
  closeCommand: () => void;
  openWizard: () => void;
  closeWizard: () => void;
  /** Merge a partial into the shared run config (algorithm/budget/seed). */
  setRunConfig: (patch: Partial<RunConfig>) => void;
}

export const useUIStore = create<UIStore>((set) => ({
  appConfig: null,
  env: null,
  selectedSpec: null,
  selectedRunId: null,
  rightOpen: true,
  bottomOpen: false,
  bottomTab: "log",
  commandOpen: false,
  wizardOpen: false,
  runConfig: DEFAULT_RUN_CONFIG,

  setAppConfig: (appConfig) => set({ appConfig }),
  setEnv: (env) => set({ env }),
  setSelectedSpec: (selectedSpec) => set({ selectedSpec }),
  openRun: (id) => set({ selectedRunId: id }),
  toggleRight: () => set((s) => ({ rightOpen: !s.rightOpen })),
  toggleBottom: () => set((s) => ({ bottomOpen: !s.bottomOpen })),
  setBottomTab: (bottomTab) => set({ bottomTab }),
  openCommand: () => set({ commandOpen: true }),
  closeCommand: () => set({ commandOpen: false }),
  openWizard: () => set({ wizardOpen: true, commandOpen: false }),
  closeWizard: () => set({ wizardOpen: false }),
  setRunConfig: (patch) => set((s) => ({ runConfig: { ...s.runConfig, ...patch } })),
}));
