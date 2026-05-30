"use client";
import { useState } from "react";
import { RefreshCw, X, Trash2 } from "lucide-react";
import { useProjectStore } from "@/stores/projectStore";
import { useExplorerStore } from "@/stores/explorerStore";
import { useRunStore } from "@/stores/runStore";
import { useUIStore } from "@/stores/uiStore";
import { api } from "@/lib/api";
import { cn, formatEng } from "@/lib/utils";
import { Sparkline } from "@/components/ui/sparkline";

/**
 * Left rail (Phase 1): project summary + the checkpoint list (salvaged from the
 * pre-Studio LeftRail). In later phases this becomes per-activity content
 * (run history, file tree, spec list, …); for now it's a single always-on panel
 * so the existing checkpoint-management flow is preserved.
 */
function RailHeading({
  children,
  right,
}: {
  children: React.ReactNode;
  right?: React.ReactNode;
}) {
  return (
    <div className="mb-1.5 mt-3 flex items-center justify-between text-[10px] font-medium uppercase tracking-[0.08em] text-faint first:mt-0">
      <span>{children}</span>
      {right}
    </div>
  );
}

export function StudioLeftRail() {
  const { summary, isApplied } = useProjectStore();
  const { availableCheckpoints, setAvailableCheckpoints } = useExplorerStore();
  const { history, rerun, clearHistory, runId, isRunning } = useRunStore();
  const { selectedRunId, openRun } = useUIStore();
  const [refreshing, setRefreshing] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const activeName = summary?.name ?? "no project";

  const refreshCheckpoints = async () => {
    setRefreshing(true);
    setError(null);
    try {
      setAvailableCheckpoints(await api.listCheckpoints());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Refresh failed");
    } finally {
      setRefreshing(false);
    }
  };

  const handleDelete = async (id: string, label: string) => {
    if (!window.confirm(`Delete checkpoint "${label}"? This removes the file on disk.`)) return;
    setPendingDelete(id);
    setError(null);
    try {
      await api.deleteCheckpoint(id);
      setAvailableCheckpoints(await api.listCheckpoints());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed");
    } finally {
      setPendingDelete(null);
    }
  };

  return (
    <aside className="flex w-[200px] shrink-0 flex-col border-r border-border bg-panel text-xs">
      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        <RailHeading>Project</RailHeading>
        <div className="flex items-center justify-between rounded px-1.5 py-1 text-fg">
          <span className="truncate">{activeName}</span>
          <span className="font-mono text-[10px] text-muted">
            {isApplied ? "active" : "draft"}
          </span>
        </div>

        <RailHeading
          right={
            history.length > 0 ? (
              <button
                type="button"
                onClick={clearHistory}
                aria-label="Clear run history"
                title="Clear run history"
                className="rounded p-0.5 text-muted normal-case tracking-normal hover:bg-hairline hover:text-fg"
              >
                <Trash2 className="h-3 w-3" />
              </button>
            ) : undefined
          }
        >
          Runs
        </RailHeading>
        {history.length === 0 ? (
          <div className="px-1.5 py-1 text-[11px] text-faint">
            No runs yet. Replay a checkpoint on Optimize.
          </div>
        ) : (
          history.map((r) => {
            const active = selectedRunId === r.id || (isRunning && runId === r.id);
            return (
              <button
                key={r.id}
                type="button"
                onClick={() => {
                  openRun(r.id);
                  if (r.kind === "replay") void rerun(r);
                }}
                title={r.kind === "replay" ? "Click to replay this run" : "Live run (view-only)"}
                className={cn(
                  "group flex w-full items-center gap-2 rounded px-1.5 py-1 text-left transition",
                  active ? "bg-primary-soft" : "hover:bg-hairline",
                )}
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-1">
                    <span className="truncate text-[11px]">{r.label}</span>
                    <span className="font-mono text-[10px] text-muted">
                      {r.bestScore != null ? formatEng(r.bestScore) : "—"}
                    </span>
                  </div>
                  <div className="mt-0.5 font-mono text-[9px] text-faint">
                    {r.kind} · {r.finalIter} it
                  </div>
                </div>
                <Sparkline values={r.sparkline} width={56} height={18} />
              </button>
            );
          })
        )}

        <RailHeading
          right={
            <button
              type="button"
              onClick={refreshCheckpoints}
              disabled={refreshing}
              aria-label="Refresh checkpoint list"
              title="Re-scan the checkpoints directory"
              className="rounded p-0.5 text-muted normal-case tracking-normal hover:bg-hairline hover:text-fg disabled:opacity-50"
            >
              <RefreshCw className={cn("h-3 w-3", refreshing && "animate-spin")} />
            </button>
          }
        >
          Checkpoints
        </RailHeading>

        {availableCheckpoints.length === 0 ? (
          <div className="px-1.5 py-1 text-[11px] text-faint">No checkpoints found.</div>
        ) : (
          availableCheckpoints.slice(0, 12).map((c) => {
            const deletable = c.source === "autosave";
            const isDeleting = pendingDelete === c.id;
            return (
              <div
                key={c.id}
                className="group flex items-start gap-1 rounded px-1.5 py-1 hover:bg-hairline"
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-1">
                    <span className="truncate text-[11px]">{c.label}</span>
                    {c.n_iters != null && (
                      <span className="font-mono text-[10px] text-muted">{c.n_iters}</span>
                    )}
                  </div>
                  <div className="mt-0.5 font-mono text-[9px] text-faint">
                    {c.source === "preset" ? "preset · " : ""}
                    {c.score_fn || c.type}
                  </div>
                </div>
                {deletable && (
                  <button
                    type="button"
                    onClick={() => handleDelete(c.id, c.label)}
                    disabled={isDeleting}
                    aria-label={`Delete checkpoint ${c.label}`}
                    title="Delete this autosaved checkpoint"
                    className="rounded p-0.5 text-faint opacity-0 transition-opacity hover:bg-danger-soft hover:text-danger group-hover:opacity-100 disabled:opacity-50"
                  >
                    <X className="h-3 w-3" />
                  </button>
                )}
              </div>
            );
          })
        )}
        {error && <div className="mt-1 px-1.5 py-1 text-[10px] text-danger">{error}</div>}
        <div className="mt-1 px-1.5 text-[10px] text-faint">
          Add new ones via Optimize → Save checkpoint. Presets are read-only.
        </div>
      </div>

      <div className="shrink-0 border-t border-border p-3 font-mono text-[10px] text-muted">
        0.4.2 · NEWCAS demo
      </div>
    </aside>
  );
}
