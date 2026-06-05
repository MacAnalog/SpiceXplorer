"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Route } from "next";
import { useRouter, usePathname } from "next/navigation";
import { useUIStore } from "@/stores/uiStore";
import { useRunStore } from "@/stores/runStore";
import { useProjectStore } from "@/stores/projectStore";
import { PRIMARY_VIEWS, SETTINGS_VIEW } from "@/components/shell/nav";
import { cn } from "@/lib/utils";

interface Command {
  id: string;
  group: string;
  label: string;
  hint?: string;
  /** Disabled commands render greyed and are skipped by keyboard/enter. */
  disabled?: boolean;
  run: () => void;
}

/**
 * ⌘K command palette + global keyboard map. Self-contained: it installs the
 * window keydown listener itself (⌘K toggles, ⌘1..6 switch view, Esc closes),
 * and composes already-built state — nav views, run history, target specs — into
 * one searchable list. Lives in StudioShell so it's available on every view.
 */
export function CommandPalette() {
  const router = useRouter();
  const pathname = usePathname();
  const { commandOpen, openCommand, closeCommand, setSelectedSpec, openRun, openWizard } =
    useUIStore();
  const { summary, isApplied } = useProjectStore();
  const { history, isRunning, stopRun, rerun } = useRunStore();

  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // Stable navigation helper used by both the command list and the key map.
  const go = useCallback((path: string) => router.push(path as Route), [router]);

  // Build the full command list from current app state.
  const commands = useMemo<Command[]>(() => {
    const cmds: Command[] = [];

    // Switch view
    for (const v of [...PRIMARY_VIEWS, SETTINGS_VIEW]) {
      const locked = !!v.requiresProject && !isApplied;
      cmds.push({
        id: `view:${v.id}`,
        group: "Switch view",
        label: v.label,
        hint: `⌘${v.shortcut}`,
        disabled: locked,
        run: () => go(v.path),
      });
    }

    // Jump to spec (deep-link into Score Shaping)
    if (isApplied && summary) {
      for (const spec of summary.target_specs) {
        cmds.push({
          id: `spec:${spec.name}`,
          group: "Jump to spec",
          label: spec.name,
          // A disabled spec isn't shown in Score Shaping, so deep-linking it
          // would silently no-op — mark it non-actionable.
          hint: spec.enable ? "score shaping" : "disabled",
          disabled: !spec.enable,
          run: () => {
            setSelectedSpec(spec.name);
            go("/scoring");
          },
        });
      }
    }

    // Jump to run (focus a past run). Replay records can be re-loaded (rerun
    // re-opens the SSE stream and repopulates the charts); live records have no
    // reloadable event data, so they only get the rail highlight.
    for (const r of history) {
      const replayable = r.kind === "replay" && !!r.checkpointId;
      cmds.push({
        id: `run:${r.id}`,
        group: "Jump to run",
        label: r.label,
        hint: replayable
          ? r.bestScore != null
            ? `replay · best ${r.bestScore.toExponential(2)}`
            : "replay"
          : "live (view-only)",
        run: () => {
          openRun(r.id);
          go("/optimize");
          if (replayable) void rerun(r);
        },
      });
    }

    // Actions
    cmds.push({
      id: "action:new-project",
      group: "Actions",
      label: "New project…",
      hint: "wizard",
      run: () => openWizard(),
    });
    cmds.push({
      id: "action:stop",
      group: "Actions",
      label: "Stop run",
      hint: isRunning ? undefined : "no active run",
      disabled: !isRunning,
      run: () => stopRun(),
    });

    return cmds;
  }, [go, summary, isApplied, history, isRunning, setSelectedSpec, openRun, openWizard, stopRun, rerun]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return commands;
    return commands.filter(
      (c) => c.label.toLowerCase().includes(q) || c.group.toLowerCase().includes(q),
    );
  }, [commands, query]);

  // Global key map — installed once.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === "k") {
        e.preventDefault();
        if (commandOpen) closeCommand();
        else openCommand();
        return;
      }
      if (commandOpen && e.key === "Escape") {
        e.preventDefault();
        closeCommand();
        return;
      }
      // ⌘1..7 switch view — but not while typing in an input/editor.
      if (mod && /^[1-7]$/.test(e.key)) {
        const target = e.target as HTMLElement | null;
        const tag = target?.tagName?.toLowerCase();
        if (tag === "input" || tag === "textarea" || target?.isContentEditable) return;
        const view = [...PRIMARY_VIEWS, SETTINGS_VIEW].find((v) => v.shortcut === e.key);
        if (view && !(view.requiresProject && !isApplied)) {
          e.preventDefault();
          go(view.path);
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // go/router are stable enough for this listener; re-bind on gating change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [commandOpen, isApplied, openCommand, closeCommand]);

  // Reset query/cursor + focus on open.
  useEffect(() => {
    if (commandOpen) {
      setQuery("");
      setCursor(0);
      // focus after paint
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [commandOpen]);

  // Keep cursor in range as the filtered list changes.
  useEffect(() => {
    setCursor((c) => Math.min(c, Math.max(0, filtered.length - 1)));
  }, [filtered.length]);

  if (!commandOpen) return null;

  const runAt = (idx: number) => {
    const cmd = filtered[idx];
    if (!cmd || cmd.disabled) return;
    closeCommand();
    cmd.run();
  };

  const onInputKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setCursor((c) => Math.min(c + 1, filtered.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setCursor((c) => Math.max(c - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      runAt(cursor);
    }
  };

  // Group the filtered commands while preserving a flat index for the cursor.
  let flatIndex = -1;
  const groups = [...new Set(filtered.map((c) => c.group))];

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/30 pt-[12vh]"
      onClick={closeCommand}
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
    >
      <div
        className="w-[620px] max-w-[92vw] overflow-hidden rounded-lg border border-border bg-panel shadow-soft"
        onClick={(e) => e.stopPropagation()}
      >
        <input
          ref={inputRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={onInputKey}
          placeholder="Type a command or search… (views, specs, runs)"
          aria-label="Command search"
          className="w-full border-b border-border bg-transparent px-4 py-3 text-sm text-fg outline-none placeholder:text-faint"
        />
        <div className="max-h-[50vh] overflow-y-auto py-1">
          {filtered.length === 0 && (
            <div className="px-4 py-6 text-center text-[13px] text-faint">
              No matching commands.
            </div>
          )}
          {groups.map((group) => (
            <div key={group}>
              <div className="px-4 pb-1 pt-2 text-[10px] font-medium uppercase tracking-wider text-faint">
                {group}
              </div>
              {filtered
                .map((c, i) => ({ c, i }))
                .filter(({ c }) => c.group === group)
                .map(({ c }) => {
                  flatIndex += 1;
                  const idx = flatIndex;
                  const active = idx === cursor;
                  return (
                    <button
                      key={c.id}
                      type="button"
                      disabled={c.disabled}
                      onMouseEnter={() => setCursor(idx)}
                      onClick={() => runAt(idx)}
                      className={cn(
                        "flex w-full items-center justify-between gap-3 px-4 py-1.5 text-left text-[13px] transition",
                        active ? "bg-primary-soft text-primary" : "text-fg",
                        c.disabled && "cursor-not-allowed opacity-40",
                      )}
                    >
                      <span className="truncate">{c.label}</span>
                      {c.hint && (
                        <span className="shrink-0 font-mono text-[10px] text-muted">{c.hint}</span>
                      )}
                    </button>
                  );
                })}
            </div>
          ))}
        </div>
        <div className="flex items-center gap-3 border-t border-border px-4 py-1.5 text-[10px] text-faint">
          <span><span className="font-mono">↑↓</span> navigate</span>
          <span><span className="font-mono">↵</span> run</span>
          <span><span className="font-mono">esc</span> close</span>
          <span className="ml-auto font-mono">{pathname}</span>
        </div>
      </div>
    </div>
  );
}
