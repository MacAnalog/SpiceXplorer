"use client";
import { useState } from "react";
import { RefreshCw, X } from "lucide-react";
import { useProjectStore } from "@/stores/projectStore";
import { useExplorerStore } from "@/stores/explorerStore";
import { useRunStore } from "@/stores/runStore";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { TAB_META, type TabId } from "./Topbar";

function RailSection({
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

interface Props {
  activeTab: TabId;
  onTabChange: (id: TabId) => void;
}

export function LeftRail({ activeTab, onTabChange }: Props) {
  const { summary, isApplied } = useProjectStore();
  const { availableCheckpoints, setAvailableCheckpoints } = useExplorerStore();
  const { isRunning, isReplay, currentIter, budget } = useRunStore();
  const [refreshing, setRefreshing] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const activeName = summary?.name ?? "no project";
  const healthMeta = TAB_META.health;
  const HealthIcon = healthMeta.icon;
  const healthActive = activeTab === "health";

  const refreshCheckpoints = async () => {
    setRefreshing(true);
    setError(null);
    try {
      const list = await api.listCheckpoints();
      setAvailableCheckpoints(list);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Refresh failed");
    } finally {
      setRefreshing(false);
    }
  };

  const handleDelete = async (id: string, label: string) => {
    if (!window.confirm(`Delete checkpoint "${label}"? This removes the file on disk.`)) {
      return;
    }
    setPendingDelete(id);
    setError(null);
    try {
      await api.deleteCheckpoint(id);
      const list = await api.listCheckpoints();
      setAvailableCheckpoints(list);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed");
    } finally {
      setPendingDelete(null);
    }
  };

  return (
    <aside className="flex w-[200px] shrink-0 flex-col border-r border-border bg-panel text-xs">
      {/* Scrollable upper region: project + checkpoints */}
      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        <RailSection>Project</RailSection>
        <div className="flex items-center justify-between rounded px-1.5 py-1 text-fg">
          <span className="truncate">{activeName}</span>
          <span className="font-mono text-[10px] text-muted">
            {isApplied ? "active" : "draft"}
          </span>
        </div>

        <RailSection
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
        </RailSection>
        {availableCheckpoints.length === 0 ? (
          <div className="px-1.5 py-1 text-[11px] text-faint">
            No checkpoints found.
          </div>
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
                      <span className="font-mono text-[10px] text-muted">
                        {c.n_iters}
                      </span>
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
                    className="opacity-0 transition-opacity group-hover:opacity-100 rounded p-0.5 text-faint hover:bg-danger-soft hover:text-danger disabled:opacity-50"
                  >
                    <X className="h-3 w-3" />
                  </button>
                )}
              </div>
            );
          })
        )}
        {error && (
          <div className="mt-1 px-1.5 py-1 text-[10px] text-danger">{error}</div>
        )}
        <div className="mt-1 px-1.5 text-[10px] text-faint">
          Add new ones via Optimize → Save checkpoint. Presets are read-only.
        </div>
      </div>

      {/* Pinned bottom region: Health link + status badges + version */}
      <div className="shrink-0 border-t border-border p-3 space-y-2">
        <button
          type="button"
          onClick={() => onTabChange("health")}
          aria-current={healthActive ? "page" : undefined}
          className={cn(
            "flex w-full items-center gap-1.5 rounded px-1.5 py-1 text-[12px] transition",
            healthActive
              ? "bg-primary-soft font-medium text-primary"
              : "text-fg hover:bg-hairline",
          )}
        >
          <HealthIcon className="h-3.5 w-3.5" aria-hidden />
          {healthMeta.label}
        </button>

        <div className="flex flex-col items-start gap-1">
          {isApplied && (
            <Badge variant="ok" dot>
              project applied
            </Badge>
          )}
          {isRunning && (
            <Badge variant="live" dot pulse>
              {isReplay ? "replay" : "live"}
              {budget > 0 && (
                <span className="ml-1 font-mono">
                  · {currentIter}/{budget}
                </span>
              )}
            </Badge>
          )}
        </div>

        <div className="font-mono text-[10px] text-muted">
          0.4.2 · NEWCAS demo
        </div>
      </div>
    </aside>
  );
}
