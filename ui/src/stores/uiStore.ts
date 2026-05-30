import { create } from "zustand";
import type { AppConfig, EnvInfo } from "@/types/api";

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
  compareRunA: string | null;
  compareRunB: string | null;

  // Panels (always-on shell surfaces)
  rightOpen: boolean;
  bottomOpen: boolean;
  bottomTab: "log";

  // Overlays (⌘K palette, etc. — UI lands in a later phase)
  commandOpen: boolean;

  setAppConfig: (cfg: AppConfig | null) => void;
  setEnv: (env: EnvInfo | null) => void;
  setSelectedSpec: (name: string | null) => void;
  setSelectedRunId: (id: string | null) => void;
  /** Deep-link: focus a run in the history/convergence surfaces. */
  openRun: (id: string) => void;
  setCompare: (a: string | null, b: string | null) => void;
  toggleRight: () => void;
  toggleBottom: () => void;
  setBottomTab: (tab: "log") => void;
  openCommand: () => void;
  closeCommand: () => void;
}

export const useUIStore = create<UIStore>((set) => ({
  appConfig: null,
  env: null,
  selectedSpec: null,
  selectedRunId: null,
  compareRunA: null,
  compareRunB: null,
  rightOpen: true,
  bottomOpen: false,
  bottomTab: "log",
  commandOpen: false,

  setAppConfig: (appConfig) => set({ appConfig }),
  setEnv: (env) => set({ env }),
  setSelectedSpec: (selectedSpec) => set({ selectedSpec }),
  setSelectedRunId: (selectedRunId) => set({ selectedRunId }),
  openRun: (id) => set({ selectedRunId: id }),
  setCompare: (compareRunA, compareRunB) => set({ compareRunA, compareRunB }),
  toggleRight: () => set((s) => ({ rightOpen: !s.rightOpen })),
  toggleBottom: () => set((s) => ({ bottomOpen: !s.bottomOpen })),
  setBottomTab: (bottomTab) => set({ bottomTab }),
  openCommand: () => set({ commandOpen: true }),
  closeCommand: () => set({ commandOpen: false }),
}));
