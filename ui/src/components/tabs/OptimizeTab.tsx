"use client";
import { useEffect, useRef, useState } from "react";
import { Play, Square } from "lucide-react";
import { Panel, PanelBody, PanelHeader } from "@/components/ui/panel";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useProjectStore } from "@/stores/projectStore";
import { useRunStore } from "@/stores/runStore";
import { api } from "@/lib/api";
import { COLORS } from "@/components/charts/PlotlyChart";
import { ScoreConvergenceChart } from "@/components/charts/ScoreConvergenceChart";
import { MetricConvergenceChart } from "@/components/charts/MetricConvergenceChart";
import type { DemoConfig } from "@/types/api";

interface Props {
  demoConfig: DemoConfig | null;
}

export function OptimizeTab({ demoConfig }: Props) {
  const { summary, yamlPath, isApplied } = useProjectStore();
  const {
    runId,
    isRunning,
    isReplay,
    budget,
    events,
    bestMetrics,
    currentIter,
    startRun,
    pushEvent,
    stopRun,
    reset,
  } = useRunStore();

  const [algorithm, setAlgorithm] = useState("LhsDE");
  const [runBudget, setRunBudget] = useState(200);
  const [replayCheckpoint, setReplayCheckpoint] = useState<string>("");
  const [selectedMetric, setSelectedMetric] = useState<string>("");
  const [startError, setStartError] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const enabledSpecs = summary?.target_specs.filter((s) => s.enable) ?? [];
  const metricNames = enabledSpecs.map((s) => s.name);

  useEffect(() => {
    if (metricNames.length > 0 && !selectedMetric) {
      setSelectedMetric(metricNames[0]);
    }
  }, [metricNames, selectedMetric]);

  const handleStart = async () => {
    setStartError(null);
    reset();
    try {
      const res = replayCheckpoint
        ? await api.startRun({ replay: true, checkpoint_id: replayCheckpoint })
        : await api.startRun({ yaml_path: yamlPath, budget: runBudget });

      startRun(res.run_id, res.replay, runBudget);

      const src = new EventSource(api.streamUrl(res.run_id));
      eventSourceRef.current = src;
      src.onmessage = (e) => {
        const evt = JSON.parse(e.data);
        if (evt.done || evt.error) {
          stopRun();
          src.close();
          return;
        }
        if (!evt.heartbeat) pushEvent(evt);
      };
      src.onerror = () => {
        stopRun();
        src.close();
      };
    } catch (err) {
      setStartError(err instanceof Error ? err.message : "Failed to start run");
    }
  };

  const handleStop = async () => {
    if (runId) await api.stopRun(runId).catch(() => {});
    stopRun();
    eventSourceRef.current?.close();
  };

  // Cleanup on unmount
  useEffect(() => () => { eventSourceRef.current?.close(); }, []);

  const canStart = isApplied || !!replayCheckpoint;

  const scoreRuns = [
    {
      label: isReplay ? "Replay" : "Live",
      scores: events.map((e) => e.score ?? null),
      best_scores: events.map((e) => e.best_score ?? null),
      color: COLORS.indigo,
    },
  ];

  const selectedSpecObj = enabledSpecs.find((s) => s.name === selectedMetric);
  const metricRuns =
    selectedMetric
      ? [
          {
            label: isReplay ? "Replay" : "Live",
            values: events.map((e) => e.metrics?.[selectedMetric] ?? null),
            color: COLORS.indigo,
          },
        ]
      : [];

  return (
    <div className="space-y-4">
      {/* Config + Status */}
      <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
        <Panel>
          <PanelHeader>
            <span className="text-sm font-semibold">Run Configuration</span>
          </PanelHeader>
          <PanelBody className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-medium uppercase tracking-wide text-zinc-500">
                  Algorithm
                </label>
                <select
                  value={algorithm}
                  onChange={(e) => setAlgorithm(e.target.value)}
                  disabled={isRunning}
                  className="mt-1 block w-full rounded-md border border-zinc-300 bg-white px-2 py-1.5 text-sm disabled:opacity-50 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                >
                  <option value="LhsDE">LhsDE</option>
                  <option value="LHSSearch">LHSSearch (blind)</option>
                  <option value="LogBFGSCMAPlus">LogBFGS CMA+</option>
                </select>
              </div>
              <div>
                <label className="text-xs font-medium uppercase tracking-wide text-zinc-500">
                  Budget
                </label>
                <input
                  type="number"
                  min={10}
                  max={2000}
                  value={runBudget}
                  onChange={(e) => setRunBudget(Number(e.target.value))}
                  disabled={isRunning}
                  className="mt-1 block w-full rounded-md border border-zinc-300 bg-white px-2 py-1.5 text-sm disabled:opacity-50 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                />
              </div>
            </div>
            {demoConfig && (
              <div>
                <label className="text-xs font-medium uppercase tracking-wide text-zinc-500">
                  Demo Checkpoint (Replay)
                </label>
                <select
                  value={replayCheckpoint}
                  onChange={(e) => setReplayCheckpoint(e.target.value)}
                  disabled={isRunning}
                  className="mt-1 block w-full rounded-md border border-zinc-300 bg-white px-2 py-1.5 text-sm disabled:opacity-50 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                >
                  <option value="">— Live run (requires SPICE) —</option>
                  {demoConfig.demo_checkpoints.map((ck) => (
                    <option key={ck.id} value={ck.id}>
                      {ck.label}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </PanelBody>
        </Panel>

        <Panel>
          <PanelHeader>
            <span className="text-sm font-semibold">Status</span>
          </PanelHeader>
          <PanelBody className="space-y-3">
            <div className="flex flex-wrap items-center gap-3">
              <Button
                onClick={isRunning ? handleStop : handleStart}
                disabled={!canStart && !isRunning}
              >
                {isRunning ? (
                  <>
                    <Square className="h-4 w-4" /> Stop
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4" />{" "}
                    {replayCheckpoint ? "Replay" : "Start Live Run"}
                  </>
                )}
              </Button>
              {isRunning && (
                <Badge variant="indigo">{isReplay ? "Replaying" : "Running"}</Badge>
              )}
            </div>

            {startError && (
              <p className="text-xs text-red-600">{startError}</p>
            )}

            {events.length > 0 && (
              <>
                <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-100">
                  <div
                    className="h-full rounded-full bg-indigo-500 transition-all duration-300"
                    style={{ width: `${Math.min(100, ((currentIter) / (budget || 1)) * 100)}%` }}
                  />
                </div>
                <p className="text-xs text-zinc-500">
                  {currentIter} / {budget} iterations
                </p>
              </>
            )}

            {metricNames.length > 0 && events.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {enabledSpecs.map((s) => {
                  const val = bestMetrics[s.name];
                  if (val == null) {
                    return (
                      <Badge key={s.name} variant="neutral">
                        {s.name}
                      </Badge>
                    );
                  }
                  const tol = s.tolerance ?? Math.abs(s.target) * 0.05;
                  const passes =
                    s.goal === "exceed"
                      ? val >= s.target
                      : s.goal === "minimize"
                        ? val <= s.target
                        : Math.abs(val - s.target) <= tol;
                  return (
                    <Badge key={s.name} variant={passes ? "pass" : "fail"}>
                      {s.name}
                    </Badge>
                  );
                })}
              </div>
            )}

            {!canStart && !isRunning && (
              <p className="text-xs text-zinc-400">
                Apply a project or select a demo checkpoint to enable run.
              </p>
            )}
          </PanelBody>
        </Panel>
      </div>

      {/* Charts — only visible when data exists */}
      {events.length > 0 && (
        <>
          <ScoreConvergenceChart runs={scoreRuns} />

          <Panel>
            <PanelHeader>
              <span className="text-sm font-semibold">Metric Convergence</span>
              <select
                value={selectedMetric}
                onChange={(e) => setSelectedMetric(e.target.value)}
                className="rounded-md border border-zinc-300 bg-white px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-indigo-500"
              >
                {metricNames.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </PanelHeader>
            <PanelBody className="p-0">
              <MetricConvergenceChart
                metric={selectedMetric}
                runs={metricRuns}
                target={selectedSpecObj?.target}
                goal={selectedSpecObj?.goal}
              />
            </PanelBody>
          </Panel>
        </>
      )}
    </div>
  );
}
