import { create } from "zustand";
import type { ProjectSummary } from "@/types/api";

interface ProjectStore {
  yaml: string;
  yamlPath: string;
  summary: ProjectSummary | null;
  validationErrors: string[];
  isValid: boolean;
  isApplied: boolean;

  setYaml: (yaml: string) => void;
  setYamlPath: (path: string) => void;
  setSummary: (summary: ProjectSummary | null) => void;
  setValidationErrors: (errors: string[]) => void;
  apply: (summary: ProjectSummary, path: string) => void;
  reset: () => void;
}

export const useProjectStore = create<ProjectStore>((set) => ({
  yaml: "",
  yamlPath: "",
  summary: null,
  validationErrors: [],
  isValid: false,
  isApplied: false,

  setYaml: (yaml) => set({ yaml }),
  setYamlPath: (yamlPath) => set({ yamlPath }),
  setSummary: (summary) => set({ summary }),
  setValidationErrors: (errors) => set({ validationErrors: errors, isValid: errors.length === 0 }),
  apply: (summary, path) =>
    set({ summary, yamlPath: path, isApplied: true, isValid: true, validationErrors: [] }),
  reset: () =>
    set({ yaml: "", yamlPath: "", summary: null, validationErrors: [], isValid: false, isApplied: false }),
}));
