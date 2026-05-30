"use client";
import type { Route } from "next";
import { usePathname, useRouter } from "next/navigation";
import { useProjectStore } from "@/stores/projectStore";
import { cn } from "@/lib/utils";
import { PRIMARY_VIEWS, SETTINGS_VIEW, viewForPath, type StudioView } from "./nav";

/**
 * Vertical icon rail (VS Code "activity bar"). Primary views at the top, the
 * Health/diagnostics gear pinned at the bottom. Selecting an icon routes to the
 * view; the active route is highlighted. Disabled icons mirror the project
 * gating (scoring/optimize need an applied project).
 */
export function ActivityBar() {
  const pathname = usePathname();
  const router = useRouter();
  const { isApplied } = useProjectStore();
  const active = viewForPath(pathname);

  const Item = ({ view }: { view: StudioView }) => {
    const Icon = view.icon;
    const isActive = active?.id === view.id;
    const disabled = !!view.requiresProject && !isApplied;
    return (
      <button
        type="button"
        title={disabled ? `${view.label} — apply a project first` : view.label}
        aria-label={view.label}
        aria-current={isActive ? "page" : undefined}
        disabled={disabled}
        onClick={() => !disabled && router.push(view.path as Route)}
        className={cn(
          "relative flex h-11 w-12 items-center justify-center transition",
          isActive ? "text-primary" : "text-faint hover:text-fg",
          disabled && "cursor-not-allowed opacity-30 hover:text-faint",
        )}
      >
        {/* active accent bar */}
        <span
          className={cn(
            "absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded-r",
            isActive ? "bg-primary" : "bg-transparent",
          )}
          aria-hidden
        />
        <Icon className="h-[18px] w-[18px]" aria-hidden />
      </button>
    );
  };

  return (
    <nav
      aria-label="Activity bar"
      className="flex w-12 shrink-0 flex-col items-center justify-between border-r border-border bg-panel py-1.5"
    >
      <div className="flex flex-col items-center">
        {PRIMARY_VIEWS.map((v) => (
          <Item key={v.id} view={v} />
        ))}
      </div>
      <div className="flex flex-col items-center">
        <Item view={SETTINGS_VIEW} />
      </div>
    </nav>
  );
}
