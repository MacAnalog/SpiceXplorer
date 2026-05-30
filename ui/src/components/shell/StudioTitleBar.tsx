"use client";
import { Command } from "lucide-react";
import { useUIStore } from "@/stores/uiStore";

/**
 * Top title bar: brand + ⌘K command-palette trigger. The run/PDK status lives in
 * the StatusBar. The ⌘K button opens the same palette the global ⌘K shortcut does
 * (see CommandPalette).
 */
export function StudioTitleBar() {
  const openCommand = useUIStore((s) => s.openCommand);
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
        onClick={openCommand}
        title="Command palette (⌘K)"
        aria-label="Open command palette"
        className="flex items-center gap-1.5 rounded-md border border-border bg-hairline px-2 py-1 text-[11px] text-muted transition hover:border-primary/40 hover:text-fg"
      >
        <Command className="h-3 w-3" aria-hidden />
        <span className="font-mono">K</span>
      </button>
    </header>
  );
}
