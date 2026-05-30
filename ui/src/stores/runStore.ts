import { create } from "zustand";
import { api } from "@/lib/api";
import type { SSEEvent } from "@/types/api";

/**
 * The live SSE connection is held in a module-level variable, NOT in React
 * state — an EventSource isn't serializable, and keeping it out of the
 * component tree is exactly what lets a run keep streaming across view (route)
 * changes. This handler used to live in OptimizeTab; hoisting it here means the
 * RightRail, BottomPanel, and StatusBar update no matter which view is mounted.
 */
let es: EventSource | null = null;

function closeStream() {
  if (es) {
    es.close();
    es = null;
  }
}

interface RunStore {
  runId: string | null;
  isRunning: boolean;
  isReplay: boolean;
  budget: number;
  events: SSEEvent[];
  bestMetrics: Record<string, number>;
  bestParams: Record<string, number>;
  currentIter: number;

  /** Begin a run: reset state and open the SSE stream for `id`. */
  startRun: (id: string, replay: boolean, budget: number) => void;
  pushEvent: (e: SSEEvent) => void;
  /** Stream ended on its own (done/error) — close locally, no backend call. */
  finishRun: () => void;
  /** User pressed Stop — best-effort tell the backend, then close. */
  stopRun: () => void;
  reset: () => void;
}

const INITIAL = {
  runId: null,
  isRunning: false,
  isReplay: false,
  budget: 0,
  events: [] as SSEEvent[],
  bestMetrics: {} as Record<string, number>,
  bestParams: {} as Record<string, number>,
  currentIter: 0,
};

export const useRunStore = create<RunStore>((set, get) => ({
  ...INITIAL,

  startRun: (id, replay, budget) => {
    closeStream();
    set({
      runId: id,
      isRunning: true,
      isReplay: replay,
      budget,
      events: [],
      bestMetrics: {},
      bestParams: {},
      currentIter: 0,
    });

    es = new EventSource(api.streamUrl(id));
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data) as SSEEvent;
        if (data.done) {
          get().finishRun();
          return;
        }
        if (data.heartbeat) return;
        get().pushEvent(data);
      } catch {
        /* ignore malformed keepalive lines */
      }
    };
    es.onerror = () => {
      get().finishRun();
    };
  },

  pushEvent: (e) =>
    set((state) => {
      const events = [...state.events, e];
      const bestMetrics = e.metrics
        ? {
            ...state.bestMetrics,
            ...(Object.fromEntries(
              Object.entries(e.metrics).filter(([, v]) => v !== null),
            ) as Record<string, number>),
          }
        : state.bestMetrics;
      const bestParams = e.best_params
        ? {
            ...state.bestParams,
            ...(Object.fromEntries(
              Object.entries(e.best_params).filter(([, v]) => v !== null),
            ) as Record<string, number>),
          }
        : state.bestParams;
      return { events, bestMetrics, bestParams, currentIter: e.iter ?? state.currentIter };
    }),

  finishRun: () => {
    closeStream();
    set({ isRunning: false });
  },

  stopRun: () => {
    const id = get().runId;
    if (id) api.stopRun(id).catch(() => {});
    closeStream();
    set({ isRunning: false });
  },

  reset: () => {
    closeStream();
    set({ ...INITIAL });
  },
}));
