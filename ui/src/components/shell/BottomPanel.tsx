"use client";
import { X } from "lucide-react";
import { useRunStore } from "@/stores/runStore";
import { useUIStore } from "@/stores/uiStore";
import { cn } from "@/lib/utils";
import { formatEng } from "@/lib/utils";

/**
 * Collapsible bottom panel — the optimizer log, rendered from runStore.events.
 * Because the SSE stream lives in the store, the log keeps filling during a run
 * even while the user is on a different view. Toggled from the StatusBar.
 */
export function BottomPanel() {
  const bottomOpen = useUIStore((s) => s.bottomOpen);
  const bottomTab = useUIStore((s) => s.bottomTab);
  const toggleBottom = useUIStore((s) => s.toggleBottom);
  const setBottomTab = useUIStore((s) => s.setBottomTab);
  const events = useRunStore((s) => s.events);

  if (!bottomOpen) return null;

  // Show the most recent lines (cap for perf on long replays).
  const lines = events.slice(-500);

  return (
    <div className="flex h-48 shrink-0 flex-col border-t border-border bg-panel">
      <div className="flex items-center justify-between border-b border-border px-2">
        <div className="flex items-stretch">
          <button
            type="button"
            onClick={() => setBottomTab("log")}
            className={cn(
              "border-b-[1.5px] px-3 py-1.5 text-[12px] transition",
              bottomTab === "log"
                ? "border-primary font-medium text-primary"
                : "border-transparent text-muted hover:text-fg",
            )}
          >
            Optimizer log
          </button>
        </div>
        <button
          type="button"
          onClick={toggleBottom}
          aria-label="Close bottom panel"
          title="Close"
          className="rounded p-1 text-faint hover:bg-hairline hover:text-fg"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-2 font-mono text-[11px] leading-relaxed text-muted">
        {lines.length === 0 ? (
          <div className="px-1 py-2 text-faint">
            No run yet — start a run or replay a checkpoint to stream the log.
          </div>
        ) : (
          lines.map((e, i) => {
            const iter = e.iter != null ? `#${e.iter}` : "·";
            const score = e.score != null ? formatEng(e.score) : "—";
            const best = e.best_score != null ? formatEng(e.best_score) : "—";
            return (
              <div key={i} className="whitespace-pre">
                <span className="text-faint">{iter.padStart(5)}</span>{"  "}
                score=<span className="text-fg">{score}</span>{"  "}
                best=<span className="text-primary">{best}</span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
