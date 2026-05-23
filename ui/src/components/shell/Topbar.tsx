"use client";
import {
  Wrench,
  Zap,
  Compass,
  SlidersHorizontal,
  Activity,
  type LucideIcon,
} from "lucide-react";
import { useProjectStore } from "@/stores/projectStore";
import { cn } from "@/lib/utils";

// Health lives on the left rail now, but the type still includes it so
// page.tsx can route to it via the rail click.
export type TabId = "setup" | "optimize" | "explorer" | "shaping" | "health";

/** Tab metadata for both the top tab bar and the left rail's "Health" entry. */
export const TAB_META: Record<TabId, { label: string; icon: LucideIcon }> = {
  setup: { label: "Setup", icon: Wrench },
  optimize: { label: "Optimize", icon: Zap },
  explorer: { label: "Explorer", icon: Compass },
  shaping: { label: "Score Shaping", icon: SlidersHorizontal },
  health: { label: "Health", icon: Activity },
};

const TOPBAR_TABS: TabId[] = ["setup", "optimize", "explorer", "shaping"];

interface Props {
  activeTab: TabId;
  onTabChange: (id: TabId) => void;
}

export function Topbar({ activeTab, onTabChange }: Props) {
  const { isApplied } = useProjectStore();

  return (
    <header className="flex h-11 shrink-0 items-center gap-3.5 border-b border-border bg-panel px-4">
      {/* Brand */}
      <div className="flex items-baseline gap-2">
        <svg width="18" height="18" viewBox="0 0 18 18" className="self-center">
          <rect x="2" y="2" width="14" height="14" rx="3" fill="#4f46e5" />
          <path
            d="M5 9 L9 13 L13 5"
            stroke="white"
            strokeWidth="1.6"
            fill="none"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <span className="text-sm font-semibold tracking-[-0.01em] text-fg">
          SpiceXplorer
        </span>
      </div>

      {/* Tabs — icons + short labels */}
      <nav
        role="tablist"
        aria-label="Main navigation"
        className="-mb-px flex flex-1 items-stretch gap-0.5"
      >
        {TOPBAR_TABS.map((id) => {
          const meta = TAB_META[id];
          const disabled =
            !isApplied && id !== "setup" && id !== "explorer";
          const active = activeTab === id;
          const Icon = meta.icon;
          return (
            <button
              key={id}
              id={`${id}-tab`}
              role="tab"
              aria-label={meta.label}
              title={meta.label}
              aria-selected={active}
              aria-controls={`${id}-panel`}
              onClick={() => !disabled && onTabChange(id)}
              disabled={disabled}
              className={cn(
                "relative flex items-center gap-1.5 border-b-[1.5px] px-3 text-[13px] transition",
                active
                  ? "border-primary font-medium text-primary"
                  : "border-transparent text-muted hover:text-fg",
                disabled && "cursor-not-allowed opacity-40 hover:text-muted",
              )}
            >
              <Icon className="h-4 w-4" aria-hidden />
              {meta.label}
            </button>
          );
        })}
      </nav>
    </header>
  );
}
