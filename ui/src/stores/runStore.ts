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

/** Metadata describing a finished run (Phase 3 run history). */
export interface RunRecord {
  id: string;
  label: string;
  kind: "live" | "replay";
  /** Present for replays so the run can be re-triggered from history. */
  checkpointId?: string;
  budget: number;
  finalIter: number;
  bestScore: number | null;
  /** Down-sampled best-so-far curve for the history sparkline. */
  sparkline: number[];
  ts: number;
}

/** Mutable per-run context captured at start, consumed when the run finishes. */
interface ActiveMeta {
  kind: "live" | "replay";
  label: string;
  checkpointId?: string;
}

const HISTORY_KEY = "sx_run_history";
const HISTORY_CAP = 25;
const SPARK_POINTS = 40;

function loadHistory(): RunRecord[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(HISTORY_KEY);
    return raw ? (JSON.parse(raw) as RunRecord[]) : [];
  } catch {
    return [];
  }
}

function saveHistory(history: RunRecord[]) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  } catch {
    /* quota / serialization errors are non-fatal for the demo */
  }
}

/** Down-sample a best-so-far series to at most SPARK_POINTS evenly spaced values. */
function downsample(values: number[]): number[] {
  const clean = values.filter((v) => Number.isFinite(v));
  if (clean.length <= SPARK_POINTS) return clean;
  const step = (clean.length - 1) / (SPARK_POINTS - 1);
  return Array.from({ length: SPARK_POINTS }, (_, i) => clean[Math.round(i * step)]);
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
  history: RunRecord[];

  /** Begin a run: reset state, capture meta, open the SSE stream for `id`. */
  startRun: (id: string, replay: boolean, budget: number, meta?: Partial<ActiveMeta>) => void;
  pushEvent: (e: SSEEvent) => void;
  /** Stream ended on its own (done/error) — record to history, close locally. */
  finishRun: () => void;
  /** User pressed Stop — best-effort tell the backend, record, then close. */
  stopRun: () => void;
  /** Re-trigger a replay run from a history record (no-op for live records). */
  rerun: (rec: RunRecord) => Promise<void>;
  clearHistory: () => void;
  reset: () => void;
}

let activeMeta: ActiveMeta = { kind: "replay", label: "Run" };

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
  history: loadHistory(),

  startRun: (id, replay, budget, meta) => {
    closeStream();
    activeMeta = {
      kind: meta?.kind ?? (replay ? "replay" : "live"),
      label: meta?.label ?? (replay ? "Replay" : "Live run"),
      checkpointId: meta?.checkpointId,
    };
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
    const { runId, events, budget, currentIter, history } = get();
    set({ isRunning: false });
    if (!runId || events.length === 0) return;

    // Build a metadata-only history record (no full event arrays persisted).
    const bestSeries = events.map((e) => e.best_score ?? e.score).filter((v): v is number =>
      v != null && Number.isFinite(v),
    );
    const bestScore = bestSeries.length ? bestSeries[bestSeries.length - 1] : null;
    const record: RunRecord = {
      id: runId,
      label: activeMeta.label,
      kind: activeMeta.kind,
      checkpointId: activeMeta.checkpointId,
      budget,
      finalIter: currentIter,
      bestScore,
      sparkline: downsample(bestSeries),
      ts: Date.now(),
    };
    // De-dupe by id (a re-run keeps one entry), newest first, capped.
    const next = [record, ...history.filter((r) => r.id !== runId)].slice(0, HISTORY_CAP);
    set({ history: next });
    saveHistory(next);
  },

  stopRun: () => {
    const id = get().runId;
    if (id) api.stopRun(id).catch(() => {});
    get().finishRun();
  },

  rerun: async (rec) => {
    if (rec.kind !== "replay" || !rec.checkpointId) return;
    try {
      const res = await api.startRun({ replay: true, checkpoint_id: rec.checkpointId });
      get().startRun(res.run_id, true, rec.budget, {
        kind: "replay",
        label: rec.label,
        checkpointId: rec.checkpointId,
      });
    } catch {
      /* surfaced elsewhere; history click is best-effort */
    }
  },

  clearHistory: () => {
    set({ history: [] });
    saveHistory([]);
  },

  reset: () => {
    closeStream();
    set({ ...INITIAL });
  },
}));
