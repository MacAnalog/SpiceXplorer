import { create } from "zustand";
import type { SSEEvent } from "@/types/api";

interface RunStore {
  runId: string | null;
  isRunning: boolean;
  isReplay: boolean;
  budget: number;
  events: SSEEvent[];
  bestMetrics: Record<string, number>;
  bestParams: Record<string, number>;
  currentIter: number;

  startRun: (id: string, replay: boolean, budget: number) => void;
  pushEvent: (e: SSEEvent) => void;
  stopRun: () => void;
  reset: () => void;
}

export const useRunStore = create<RunStore>((set) => ({
  runId: null,
  isRunning: false,
  isReplay: false,
  budget: 0,
  events: [],
  bestMetrics: {},
  bestParams: {},
  currentIter: 0,

  startRun: (id, replay, budget) =>
    set({ runId: id, isRunning: true, isReplay: replay, budget, events: [], bestMetrics: {}, bestParams: {}, currentIter: 0 }),

  pushEvent: (e) =>
    set((state) => {
      const events = [...state.events, e];
      const bestMetrics = e.metrics
        ? { ...state.bestMetrics, ...(Object.fromEntries(
            Object.entries(e.metrics).filter(([, v]) => v !== null)
          ) as Record<string, number>) }
        : state.bestMetrics;
      const bestParams = e.best_params
        ? { ...state.bestParams, ...(Object.fromEntries(
            Object.entries(e.best_params).filter(([, v]) => v !== null)
          ) as Record<string, number>) }
        : state.bestParams;
      return { events, bestMetrics, bestParams, currentIter: e.iter ?? state.currentIter };
    }),

  stopRun: () => set({ isRunning: false }),

  reset: () =>
    set({ runId: null, isRunning: false, isReplay: false, budget: 0, events: [], bestMetrics: {}, bestParams: {}, currentIter: 0 }),
}));
