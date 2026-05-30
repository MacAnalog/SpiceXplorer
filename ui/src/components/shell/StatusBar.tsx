"use client";
import { usePathname } from "next/navigation";
import { useProjectStore } from "@/stores/projectStore";
import { useRunStore } from "@/stores/runStore";
import { useUIStore } from "@/stores/uiStore";
import { cn } from "@/lib/utils";
import { viewForPath } from "./nav";

function Dot({ className }: { className: string }) {
  return <span className={cn("inline-block h-1.5 w-1.5 rounded-full", className)} aria-hidden />;
}

/**
 * Bottom status strip — always-on summary of project, active run, and the
 * simulator/PDK environment. The PDK pill is the first-class surfacing of the
 * graceful-degradation state (see /api/env): on a PDK-less machine it reads
 * "PDK missing — replay only" so the limitation is explicit, not a silent error.
 */
export function StatusBar() {
  const pathname = usePathname();
  const view = viewForPath(pathname);
  const { summary, isApplied } = useProjectStore();
  const { isRunning, isReplay, currentIter, budget } = useRunStore();
  const env = useUIStore((s) => s.env);

  // Environment pill
  let envDot = "bg-faint";
  let envText = "sim: checking…";
  if (env) {
    if (!env.ngspice_ok) {
      envDot = "bg-danger";
      envText = "ngspice missing";
    } else if (env.pdk_ok) {
      envDot = "bg-ok";
      envText = "sim ready";
    } else {
      envDot = "bg-tertiary";
      envText = "PDK missing — replay only";
    }
  }

  return (
    <footer className="flex h-6 shrink-0 items-center gap-3 border-t border-border bg-panel px-3 text-[11px] text-muted">
      <span className="font-medium text-fg">{view?.label ?? "Studio"}</span>

      <span className="text-faint">·</span>
      <span className="truncate">
        {summary?.name ?? "no project"}
        <span className="ml-1 font-mono text-faint">{isApplied ? "active" : "draft"}</span>
      </span>

      {isRunning && (
        <>
          <span className="text-faint">·</span>
          <span className="flex items-center gap-1.5 text-fg">
            <Dot className="bg-primary obs-pulse" />
            {isReplay ? "replay" : "live"}
            {budget > 0 && (
              <span className="font-mono text-muted">
                {currentIter}/{budget}
              </span>
            )}
          </span>
        </>
      )}

      <div className="flex-1" />

      <span
        className="flex items-center gap-1.5"
        title={env?.pdk_detail ?? "Probing simulator + PDK availability…"}
      >
        <Dot className={envDot} />
        {envText}
      </span>
    </footer>
  );
}
