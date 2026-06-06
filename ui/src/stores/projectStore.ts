import { create } from "zustand";
import { api } from "@/lib/api";
import type { ProjectMeta, ProjectSummary } from "@/types/api";

interface ProjectStore {
  yaml: string;
  yamlPath: string;
  summary: ProjectSummary | null;
  validationErrors: string[];
  isValid: boolean;
  isApplied: boolean;
  // The active registered project (report.md P3). null = an ad-hoc/edited YAML
  // (Setup load/upload) that isn't tied to a project — its runs land unscoped.
  projectId: string | null;
  projectName: string | null;
  projects: ProjectMeta[];

  setYaml: (yaml: string) => void;
  setYamlPath: (path: string) => void;
  setSummary: (summary: ProjectSummary | null) => void;
  setValidationErrors: (errors: string[]) => void;
  /** Apply an ad-hoc/edited YAML (Setup) — detaches from any registered project. */
  apply: (summary: ProjectSummary, path: string) => void;
  setProjects: (projects: ProjectMeta[]) => void;
  refreshProjects: () => Promise<void>;
  /** Switch to a registered project: load its summary + bind project_id. */
  switchProject: (id: string) => Promise<boolean>;
  reset: () => void;
}

export const useProjectStore = create<ProjectStore>((set) => ({
  yaml: "",
  yamlPath: "",
  summary: null,
  validationErrors: [],
  isValid: false,
  isApplied: false,
  projectId: null,
  projectName: null,
  projects: [],

  setYaml: (yaml) => set({ yaml }),
  setYamlPath: (yamlPath) => set({ yamlPath }),
  setSummary: (summary) => set({ summary }),
  setValidationErrors: (errors) => set({ validationErrors: errors, isValid: errors.length === 0 }),
  apply: (summary, path) =>
    set({
      summary,
      yamlPath: path,
      isApplied: true,
      isValid: true,
      validationErrors: [],
      // An ad-hoc/edited YAML is not the registered project — clear the binding so a
      // run uses yaml_path (unscoped) rather than silently resolving to project.yaml.
      projectId: null,
      projectName: null,
    }),
  setProjects: (projects) => set({ projects }),
  refreshProjects: async () => {
    try {
      const { projects } = await api.listProjects();
      set({ projects });
    } catch {
      /* best-effort */
    }
  },
  switchProject: async (id) => {
    try {
      const d = await api.getProject(id);
      let yaml = "";
      try {
        const r = await fetch(`/api/yaml-text?path=${encodeURIComponent(d.yaml_path)}`);
        if (r.ok) yaml = await r.text();
      } catch {
        /* editor text is best-effort */
      }
      set({
        summary: d.summary,
        yamlPath: d.yaml_path,
        yaml,
        isApplied: true,
        isValid: true,
        validationErrors: [],
        projectId: d.id,
        projectName: (d.manifest?.name as string) ?? d.id,
      });
      return true;
    } catch {
      return false;
    }
  },
  reset: () =>
    set({
      yaml: "", yamlPath: "", summary: null, validationErrors: [], isValid: false,
      isApplied: false, projectId: null, projectName: null,
    }),
}));
