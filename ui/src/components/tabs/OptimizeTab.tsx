"use client";
import { useEffect, useMemo, useState } from "react";
import { useProjectStore } from "@/stores/projectStore";
import { useRunStore } from "@/stores/runStore";
import { useUIStore } from "@/stores/uiStore";
import { api } from "@/lib/api";
import { COLORS } from "@/components/charts/PlotlyChart";
import { ScoreConvergenceChart } from "@/components/charts/ScoreConvergenceChart";
import { MetricConvergenceChart } from "@/components/charts/MetricConvergenceChart";
import { Panel, PanelBody, PanelHeader } from "@/components/ui/panel";
import { Button } from "@/components/ui/button";
import { Toolbar, ToolbarLabel, ToolbarSpacer } from "@/components/shell/Toolbar";
import { Separator } from "@/components/ui/separator";
import { Segmented } from "@/components/ui/segmented";
import { selectCn } from "@/components/ui/select";
import { EmptyState } from "@/components/ui/empty-state";
import type { AppConfig } from "@/types/api";

interface Props {
  appConfig: AppConfig | null;
}

type ScoreFn = "sigmoid" | "linear";

/**
 * Optimize view — run configuration + convergence charts. Phase 2 moved the
 * live spec status, best params, and run progress to the always-on RightRail,
 * and the per-iteration log to the BottomPanel. The SSE stream is owned by
 * runStore (see startRun), so a run keeps streaming if the user navigates away.
 *
 * Live runs need the IHP PDK; when it's absent (env.live_runs_enabled === false)
 * the Start button is disabled and the user is steered to Replay.
 */
export function OptimizeTab({ appConfig }: Props) {
  const { summary, yamlPath, isApplied } = useProjectStore();
  const { isReplay, isRunning, events, startRun, stopRun } = useRunStore();
  const env = useUIStore((s) => s.env);

  const [algorithm, setAlgorithm] = useState("LhsDE");
  const [runBudget, setRunBudget] = useState(200);
  const [scoreFn, setScoreFn] = useState<ScoreFn>("sigmoid");
  const [replayCheckpoint, setReplayCheckpoint] = useState<string>("");
  const [selectedMetric, setSelectedMetric] = useState<string>("");
  const [startError, setStartError] = useState<string | null>(null);

  const enabledSpecs = useMemo(
    () => summary?.target_specs.filter((s) => s.enable) ?? [],
    [summary],
  );

  useEffect(() => {
    if (enabledSpecs.length > 0 && !selectedMetric) {
      setSelectedMetric(enabledSpecs[0].name);
    }
  }, [enabledSpecs, selectedMetric]);

  // Live runs require the PDK; without it disable Start and steer to Replay.
  const liveDisabled = env != null && !env.live_runs_enabled;
  const canStart = (isApplied && !liveDisabled) || !!replayCheckpoint;

  const handleStart = async () => {
    setStartError(null);
    try {
      const res = replayCheckpoint
        ? await api.startRun({ replay: true, checkpoint_id: replayCheckpoint })
        : await api.startRun({ yaml_path: yamlPath, budget: runBudget });
      // startRun resets run state and opens the SSE stream (handled in the store).
      startRun(res.run_id, res.replay, runBudget);
    } catch (err) {
      setStartError(err instanceof Error ? err.message : "Failed to start run");
    }
  };

  const scoreRuns = useMemo(
    () => [
      {
        label: isReplay ? "Replay" : "Live",
        scores: events.map((e) => e.score ?? null),
        best_scores: events.map((e) => e.best_score ?? null),
        color: COLORS.primary,
      },
    ],
    [events, isReplay],
  );

  const selectedSpec = enabledSpecs.find((s) => s.name === selectedMetric);
  const metricRuns = useMemo(
    () =>
      selectedMetric
        ? [
            {
              label: isReplay ? "Replay" : "Live",
              values: events.map((e) => e.metrics?.[selectedMetric] ?? null),
              color: COLORS.primary,
            },
          ]
        : [],
    [events, isReplay, selectedMetric],
  );

  return (
    <>
      <Toolbar>
        <ToolbarLabel>algorithm</ToolbarLabel>
        <select
          aria-label="Optimization algorithm"
          value={algorithm}
          onChange={(e) => setAlgorithm(e.target.value)}
          disabled={isRunning}
          className={selectCn("sm")}
        >
          <option value="LhsDE">LhsDE</option>
          <option value="LHSSearch">LHSSearch</option>
          <option value="LogBFGSCMAPlus">LogBFGSCMAPlus</option>
        </select>

        <ToolbarLabel>budget</ToolbarLabel>
        <input
          aria-label="Run budget (iterations)"
          type="number"
          min={10}
          max={5000}
          value={runBudget}
          onChange={(e) => setRunBudget(Number(e.target.value))}
          disabled={isRunning}
          className={selectCn("sm") + " w-[72px]"}
        />

        <ToolbarLabel>score</ToolbarLabel>
        <Segmented<ScoreFn>
          value={scoreFn}
          onChange={setScoreFn}
          options={[
            { value: "sigmoid", label: "sigmoid" },
            { value: "linear", label: "linear" },
          ]}
        />

        <Separator />

        <ToolbarLabel>demo replay</ToolbarLabel>
        <select
          aria-label="Replay checkpoint"
          value={replayCheckpoint}
          onChange={(e) => setReplayCheckpoint(e.target.value)}
          disabled={isRunning}
          className={selectCn("sm") + " w-[200px]"}
        >
          <option value="">— live —</option>
          {appConfig?.preset_checkpoints.map((ck) => (
            <option key={ck.id} value={ck.id}>
              {ck.label}
            </option>
          ))}
        </select>

        <ToolbarSpacer />

        {isRunning ? (
          <Button variant="danger" onClick={stopRun}>
            <span className="h-1.5 w-1.5 rounded-full bg-white obs-pulse" /> Stop
          </Button>
        ) : (
          <Button
            variant="primary"
            onClick={handleStart}
            disabled={!canStart}
            title={
              liveDisabled && !replayCheckpoint
                ? "Live optimization needs the IHP sg13g2 PDK, which isn't installed on this machine. Use Replay to drive the demo from cached runs."
                : undefined
            }
          >
            {replayCheckpoint ? "Replay" : "Start"}
          </Button>
        )}
      </Toolbar>

      {liveDisabled && (
        <div className="border-b border-warn-soft bg-warn-soft px-4 py-1.5 text-[11px] text-[#b45309]">
          PDK missing — live runs are disabled. Replay a cached checkpoint to drive the demo.
        </div>
      )}

      <div className="flex min-h-0 flex-1 flex-col gap-2.5 overflow-auto p-3">
        {startError && (
          <div
            role="alert"
            className="rounded-md border border-danger bg-danger-soft px-3 py-2 text-xs text-danger"
          >
            {startError}
          </div>
        )}

        {!canStart && events.length === 0 && (
          <EmptyState bordered minHeight="min-h-32">
            Apply a project or select a preset checkpoint to enable a run.
          </EmptyState>
        )}

        <div className="grid grid-cols-2 gap-2.5">
          <Panel>
            <PanelHeader
              title="F(x) convergence"
              mute="· raw + best-so-far"
              right={
                <span className="font-mono text-[10px]">
                  <span style={{ color: COLORS.primary }}>● best</span>{" "}
                  <span style={{ color: COLORS.muted }}>● raw</span>
                </span>
              }
            />
            <PanelBody>
              {events.length > 0 ? (
                <ScoreConvergenceChart runs={scoreRuns} />
              ) : (
                <EmptyState minHeight="h-[240px]">No data yet.</EmptyState>
              )}
            </PanelBody>
          </Panel>

          <Panel>
            <PanelHeader
              title={
                <>
                  metric · <span className="font-mono">{selectedMetric || "—"}</span> best-so-far
                </>
              }
              right={
                <select
                  aria-label="Metric to chart"
                  value={selectedMetric}
                  onChange={(e) => setSelectedMetric(e.target.value)}
                  className={selectCn("xs")}
                >
                  {enabledSpecs.map((s) => (
                    <option key={s.name} value={s.name}>
                      {s.name}
                    </option>
                  ))}
                </select>
              }
            />
            <PanelBody>
              {events.length > 0 && selectedMetric ? (
                <MetricConvergenceChart
                  metric={selectedMetric}
                  runs={metricRuns}
                  target={selectedSpec?.target}
                  goal={selectedSpec?.goal}
                />
              ) : (
                <EmptyState minHeight="h-[240px]">No data yet.</EmptyState>
              )}
            </PanelBody>
          </Panel>
        </div>
      </div>
    </>
  );
}
