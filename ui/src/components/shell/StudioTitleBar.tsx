"use client";
import { Command } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Top title bar: brand + (placeholder) ⌘K command trigger. The run/PDK status
 * lives in the StatusBar; the command palette and Run popover are wired in
 * later phases — the ⌘K affordance is shown disabled so the layout is final.
 */
export function StudioTitleBar() {
  return (
    <header className="flex h-11 shrink-0 items-center gap-3 border-b border-border bg-panel px-3">
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
        <span className="text-[11px] font-medium text-faint">Studio</span>
      </div>

      <div className="flex-1" />

      <button
        type="button"
        disabled
        title="Command palette (coming soon)"
        className={cn(
          "flex items-center gap-1.5 rounded-md border border-border bg-hairline px-2 py-1 text-[11px] text-faint",
          "cursor-not-allowed",
        )}
      >
        <Command className="h-3 w-3" aria-hidden />
        <span className="font-mono">K</span>
      </button>
    </header>
  );
}
