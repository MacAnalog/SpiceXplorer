"use client";
import type { Route } from "next";
import { usePathname, useRouter } from "next/navigation";
import { useProjectStore } from "@/stores/projectStore";
import { cn } from "@/lib/utils";
import { PRIMARY_VIEWS, viewForPath } from "./nav";

/**
 * Horizontal tab strip over the center view. Drives navigation by route; the
 * active tab is derived from the pathname. Tabs that need an applied project
 * render disabled until then (matches the pre-Studio Topbar behavior).
 */
export function TabStrip() {
  const pathname = usePathname();
  const router = useRouter();
  const { isApplied } = useProjectStore();
  const active = viewForPath(pathname);

  return (
    <nav
      role="tablist"
      aria-label="Views"
      className="flex h-10 shrink-0 items-stretch gap-0.5 border-b border-border bg-panel px-2"
    >
      {PRIMARY_VIEWS.map((view) => {
        const Icon = view.icon;
        const isActive = active?.id === view.id;
        const disabled = !!view.requiresProject && !isApplied;
        return (
          <button
            key={view.id}
            type="button"
            role="tab"
            aria-selected={isActive}
            aria-label={view.label}
            title={disabled ? `${view.label} — apply a project first` : view.label}
            disabled={disabled}
            onClick={() => !disabled && router.push(view.path as Route)}
            className={cn(
              "relative -mb-px flex items-center gap-1.5 border-b-[1.5px] px-3 text-[13px] transition",
              isActive
                ? "border-primary font-medium text-primary"
                : "border-transparent text-muted hover:text-fg",
              disabled && "cursor-not-allowed opacity-40 hover:text-muted",
            )}
          >
            <Icon className="h-4 w-4" aria-hidden />
            {view.label}
          </button>
        );
      })}
    </nav>
  );
}
