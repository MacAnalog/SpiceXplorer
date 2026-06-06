"use client";
import { useEffect, useState } from "react";
import type { Route } from "next";
import { useRouter } from "next/navigation";
import { RefreshCw, X, Trash2, Play } from "lucide-react";
import { useExplorerStore } from "@/stores/explorerStore";
import { useRunStore } from "@/stores/runStore";
import { useUIStore } from "@/stores/uiStore";
import { useProjectStore } from "@/stores/projectStore";
import { api } from "@/lib/api";
import { resumeLiveRun } from "@/lib/launchRun";
import { cn, formatEng } from "@/lib/utils";
import { Sparkline } from "@/components/ui/sparkline";
import { RailHeading, RailHint } from "./parts";
import type { ProjectRun } from "@/types/api";

const RUN_STATUS_CLS: Record<string, string> = {
  running: "bg-primary-soft text-primary",
  done: "bg-ok-soft text-ok",
  stopped: "bg-hairline text-muted",
  error: "bg-danger-soft text-danger",
};

/**
 * Run-centric rail (Optimize, Explore): run history with score sparklines plus
 * the on-disk checkpoint list with refresh/delete. Salvaged unchanged from the
 * pre-split single rail — this is the most relevant context next to a run.
 */
export function RunsRail() {
  const router = useRouter();
  const { availableCheckpoints, setAvailableCheckpoints } = useExplorerStore();
  const { history, rerun, clearHistory, runId, isRunning } = useRunStore();
  const { selectedRunId, openRun } = useUIStore();
  const env = useUIStore((s) => s.env);
  const isApplied = useProjectStore((s) => s.isApplied);
  const projectId = useProjectStore((s) => s.projectId);
  const [refreshing, setRefreshing] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [projectRuns, setProjectRuns] = useState<ProjectRun[]>([]);

  // Per-project run history (report.md P3) — server-persisted, replacing the
  // localStorage list when a registered project is active. Refetched on project
  // switch and when a run finishes (isRunning flips false).
  useEffect(() => {
    if (!projectId) {
      setProjectRuns([]);
      return;
    }
    let alive = true;
    api.getProjectRuns(projectId)
      .then((r) => { if (alive) setProjectRuns(r.runs); })
      .catch(() => { if (alive) setProjectRuns([]); });
    return () => { alive = false; };
  }, [projectId, isRunning]);

  // Resume needs an applied project, the PDK, and no run in flight.
  const canResume = isApplied && !(env != null && !env.live_runs_enabled) && !isRunning;

  const handleResume = async (id: string) => {
    setError(null);
    const res = await resumeLiveRun(id);
    if (res.ok) router.push("/optimize" as Route);
    else setError(res.error ?? "Resume failed");
  };

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
    <>
      <RailHeading
        right={
          !projectId && history.length > 0 ? (
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
        {projectId ? "Project runs" : "Runs"}
      </RailHeading>

      {/* Per-project server runs (report.md P3): honest status pills (the startup
          reconciler flips crashed 'running' → 'error'), best score, algo·budget. */}
      {projectId ? (
        projectRuns.length === 0 ? (
          <RailHint>No runs yet for this project.</RailHint>
        ) : (
          projectRuns.map((r) => (
            <div key={r.run_id} className="flex w-full items-center gap-2 rounded px-1.5 py-1">
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between gap-1">
                  <span className="truncate text-[11px]">
                    {r.label ?? r.algorithm ?? r.run_id.slice(0, 8)}
                  </span>
                  <span className="font-mono text-[10px] text-muted">
                    {r.best_score != null ? formatEng(r.best_score) : "—"}
                  </span>
                </div>
                <div className="mt-0.5 flex items-center gap-1.5 font-mono text-[9px] text-faint">
                  <span className={cn("rounded px-1", RUN_STATUS_CLS[r.status] ?? "bg-hairline text-muted")}>
                    {r.status}
                  </span>
                  <span className="truncate">
                    {r.algorithm ?? r.kind}{r.budget ? ` · ${r.budget} it` : ""}
                  </span>
                </div>
              </div>
            </div>
          ))
        )
      ) : history.length === 0 ? (
        <RailHint>No runs yet. Replay a checkpoint on Optimize.</RailHint>
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
        <RailHint>No checkpoints found.</RailHint>
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
                  onClick={() => handleResume(c.id)}
                  disabled={!canResume}
                  aria-label={`Resume from checkpoint ${c.label}`}
                  title={
                    canResume
                      ? "Resume an optimization from this checkpoint"
                      : isRunning
                        ? "A run is already active"
                        : "Apply a project (and the PDK) to resume"
                  }
                  className="rounded p-0.5 text-faint opacity-0 transition-opacity hover:bg-primary-soft hover:text-primary group-hover:opacity-100 disabled:opacity-30"
                >
                  <Play className="h-3 w-3" />
                </button>
              )}
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
    </>
  );
}
