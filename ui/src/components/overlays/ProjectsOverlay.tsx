"use client";
import { useEffect, useRef, useState } from "react";
import { FolderOpen, Plus, Beaker, Loader2 } from "lucide-react";
import { useUIStore } from "@/stores/uiStore";
import { useProjectStore } from "@/stores/projectStore";
import { api } from "@/lib/api";
import { cn, formatEng } from "@/lib/utils";
import type { ExampleMeta } from "@/types/api";

/**
 * ⌘P Projects overlay (report.md P3) — the project registry surface. A project IS a
 * directory under WORK_ROOT/projects. Lists existing projects (click to switch), loads
 * a demo AS a new project (copy-on-load), and creates a new (example-structured)
 * project. NOT a new ActivityBar view, so the nav 1..8 shortcut invariant is preserved.
 */
export function ProjectsOverlay() {
  const projectsOpen = useUIStore((s) => s.projectsOpen);
  const closeProjects = useUIStore((s) => s.closeProjects);
  const openWizard = useUIStore((s) => s.openWizard);
  const { projects, projectId, refreshProjects, switchProject } = useProjectStore();

  const [query, setQuery] = useState("");
  const [examples, setExamples] = useState<ExampleMeta[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [newName, setNewName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!projectsOpen) return;
    setQuery("");
    setNewName("");
    setError(null);
    void refreshProjects();
    api.listExamples().then((r) => setExamples(r.examples)).catch(() => setExamples([]));
    requestAnimationFrame(() => inputRef.current?.focus());
  }, [projectsOpen, refreshProjects]);

  if (!projectsOpen) return null;

  const q = query.trim().toLowerCase();
  const filteredProjects = projects.filter((p) => p.name.toLowerCase().includes(q));
  const filteredExamples = examples.filter((e) => e.name.toLowerCase().includes(q));

  const doSwitch = async (id: string) => {
    setBusy(id);
    setError(null);
    const ok = await switchProject(id);
    setBusy(null);
    if (ok) closeProjects();
    else setError("Failed to load project.");
  };

  const doLoadExample = async (e: ExampleMeta) => {
    setBusy(e.key);
    setError(null);
    try {
      const { id } = await api.fromExample(e.key, e.name);
      await refreshProjects();
      const ok = await switchProject(id);
      if (ok) closeProjects();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load example.");
    } finally {
      setBusy(null);
    }
  };

  const doCreate = async () => {
    const name = newName.trim();
    if (!name) return;
    setBusy("__new__");
    setError(null);
    try {
      const { id } = await api.createProject(name);
      await refreshProjects();
      const ok = await switchProject(id);
      if (ok) closeProjects();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create project.");
    } finally {
      setBusy(null);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/30 pt-[10vh]"
      onClick={closeProjects}
      role="dialog"
      aria-modal="true"
      aria-label="Projects"
    >
      <div
        className="flex max-h-[78vh] w-[640px] max-w-[92vw] flex-col overflow-hidden rounded-lg border border-border bg-panel shadow-soft"
        onClick={(e) => e.stopPropagation()}
      >
        <input
          ref={inputRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search projects and examples…"
          aria-label="Search projects"
          className="w-full border-b border-border bg-transparent px-4 py-3 text-sm text-fg outline-none placeholder:text-faint"
        />

        <div className="min-h-0 flex-1 overflow-y-auto py-1">
          {/* Projects (the registry) */}
          <div className="px-4 pb-1 pt-2 text-[10px] font-medium uppercase tracking-wider text-faint">
            Projects ({projects.length})
          </div>
          {filteredProjects.length === 0 && (
            <div className="px-4 py-2 text-[12px] text-faint">No projects yet — create one or load a demo below.</div>
          )}
          {filteredProjects.map((p) => (
            <button
              key={p.id}
              type="button"
              onClick={() => doSwitch(p.id)}
              className={cn(
                "flex w-full items-center justify-between gap-3 px-4 py-1.5 text-left text-[13px] transition",
                p.id === projectId ? "bg-primary-soft text-primary" : "text-fg hover:bg-hairline",
              )}
            >
              <span className="flex min-w-0 items-center gap-2">
                <FolderOpen className="h-3.5 w-3.5 shrink-0 text-faint" />
                <span className="truncate">{p.name}</span>
                <span className="shrink-0 rounded bg-hairline px-1.5 text-[10px] text-muted">{p.source}</span>
              </span>
              <span className="flex shrink-0 items-center gap-2 font-mono text-[10px] text-muted">
                {busy === p.id && <Loader2 className="h-3 w-3 animate-spin" />}
                {p.run_count} run{p.run_count === 1 ? "" : "s"}
                {p.best_score != null && <> · best {formatEng(p.best_score)}</>}
              </span>
            </button>
          ))}

          {/* Examples (load demo as project) */}
          <div className="px-4 pb-1 pt-3 text-[10px] font-medium uppercase tracking-wider text-faint">
            Load a demo as a project
          </div>
          {filteredExamples.map((e) => (
            <button
              key={e.key}
              type="button"
              onClick={() => doLoadExample(e)}
              className="flex w-full items-center justify-between gap-3 px-4 py-1.5 text-left text-[13px] text-fg transition hover:bg-hairline"
            >
              <span className="flex min-w-0 items-center gap-2">
                <Beaker className="h-3.5 w-3.5 shrink-0 text-faint" />
                <span className="truncate">{e.name}</span>
              </span>
              <span className="shrink-0 font-mono text-[10px] text-muted">
                {busy === e.key ? <Loader2 className="h-3 w-3 animate-spin" /> : "copy →"}
              </span>
            </button>
          ))}
        </div>

        {error && (
          <div className="border-t border-border bg-danger-soft px-4 py-1.5 text-[11px] text-danger">{error}</div>
        )}

        {/* Footer: new project (= a new directory) + wizard */}
        <div className="flex items-center gap-2 border-t border-border px-3 py-2">
          <input
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") void doCreate(); }}
            placeholder="New project name…"
            aria-label="New project name"
            className="min-w-0 flex-1 rounded border border-border bg-transparent px-2 py-1 text-[12px] text-fg outline-none placeholder:text-faint"
          />
          <button
            type="button"
            onClick={doCreate}
            disabled={!newName.trim() || busy === "__new__"}
            className="inline-flex shrink-0 items-center gap-1 rounded border border-border px-2 py-1 text-[12px] text-fg hover:bg-hairline disabled:opacity-40"
          >
            <Plus className="h-3 w-3" /> Create
          </button>
          <button
            type="button"
            onClick={() => { closeProjects(); openWizard(); }}
            className="shrink-0 rounded border border-border px-2 py-1 text-[12px] text-muted hover:bg-hairline hover:text-fg"
            title="Author a project YAML in the wizard"
          >
            Wizard…
          </button>
        </div>
      </div>
    </div>
  );
}
