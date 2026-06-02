// Single source of truth for Studio navigation.
//
// The shell renders three navigation surfaces — the vertical ActivityBar, the
// horizontal TabStrip, and (later) the per-activity left rail — all driven from
// this one list so a view is added/renamed in exactly one place. Routes live
// under the `(studio)` App-Router group, so each view has a real URL.
import {
  Wrench,
  SlidersHorizontal,
  Zap,
  Compass,
  CircuitBoard,
  Workflow,
  Activity,
  type LucideIcon,
} from "lucide-react";

export type StudioViewId =
  | "setup"
  | "scoring"
  | "optimize"
  | "compare"
  | "schematic"
  | "pipeline"
  | "health";

/**
 * Which left-rail variant a view shows. `runs` = run history + checkpoints
 * (run-centric views), `specs` = clickable target-spec list (spec-centric
 * views, deep-link to Score Shaping), `outline` = project structure outline.
 */
export type RailKind = "runs" | "specs" | "outline";

export interface StudioView {
  id: StudioViewId;
  /** Full label (TabStrip, tooltips). */
  label: string;
  /** App-Router path under the (studio) group. */
  path: string;
  icon: LucideIcon;
  /** ⌘<key> / bare-<key> shortcut. */
  shortcut: string;
  /**
   * When true the view needs an applied project to be useful, so the
   * ActivityBar/TabStrip render it disabled until `isApplied`. Mirrors the
   * pre-Studio Topbar gating (optimize + shaping were locked until applied).
   */
  requiresProject?: boolean;
  /** Which left-rail variant this view shows (defaults to "runs" if unset). */
  rail: RailKind;
}

/** Primary views — shown as TabStrip tabs and the top of the ActivityBar. */
export const PRIMARY_VIEWS: StudioView[] = [
  { id: "setup", label: "Setup", path: "/setup", icon: Wrench, shortcut: "1", rail: "outline" },
  { id: "scoring", label: "Score Shaping", path: "/scoring", icon: SlidersHorizontal, shortcut: "2", requiresProject: true, rail: "specs" },
  { id: "optimize", label: "Optimize", path: "/optimize", icon: Zap, shortcut: "3", requiresProject: true, rail: "runs" },
  { id: "compare", label: "Explore", path: "/compare", icon: Compass, shortcut: "4", rail: "runs" },
  { id: "schematic", label: "Schematic", path: "/schematic", icon: CircuitBoard, shortcut: "5", rail: "outline" },
  { id: "pipeline", label: "Pipeline", path: "/pipeline", icon: Workflow, shortcut: "6", requiresProject: true, rail: "specs" },
];

/** Settings/diagnostics — reached via the ActivityBar gear (the Health check). */
export const SETTINGS_VIEW: StudioView = {
  id: "health",
  label: "Health",
  path: "/health",
  icon: Activity,
  shortcut: "7",
  rail: "outline",
};

export const ALL_VIEWS: StudioView[] = [...PRIMARY_VIEWS, SETTINGS_VIEW];

/** Resolve the active view from a pathname (e.g. "/optimize" -> optimize view). */
export function viewForPath(pathname: string): StudioView | undefined {
  return ALL_VIEWS.find(
    (v) => pathname === v.path || pathname.startsWith(v.path + "/"),
  );
}
