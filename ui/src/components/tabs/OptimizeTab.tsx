"use client";
import { useEffect, useRef, useState } from "react";
import { Play, Square, FlaskConical, CheckCircle2, XCircle, Loader2, FileText, Clock } from "lucide-react";
import { Panel, PanelBody, PanelHeader } from "@/components/ui/panel";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useProjectStore } from "@/stores/projectStore";
import { useRunStore } from "@/stores/runStore";
import { api } from "@/lib/api";
import { COLORS } from "@/components/charts/PlotlyChart";
import { Select, selectCn } from "@/components/ui/select";
import { ScoreConvergenceChart } from "@/components/charts/ScoreConvergenceChart";
import { MetricConvergenceChart } from "@/components/charts/MetricConvergenceChart";
import type { AppConfig, SanityCheckResponse } from "@/types/api";

interface Props {
  appConfig: AppConfig | null;
}

export function OptimizeTab({ appConfig }: Props) {
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

  const [sanityResult, setSanityResult] = useState<SanityCheckResponse | null>(null);
  const [isSanityRunning, setIsSanityRunning] = useState(false);
  const [sanityError, setSanityError] = useState<string | null>(null);

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
        if (evt.error) {
          setStartError(`Optimizer error: ${evt.error}`);
          stopRun();
          src.close();
          return;
        }
        if (evt.done) {
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

  const handleSanityCheck = async () => {
    if (!yamlPath) return;
    setSanityResult(null);
    setSanityError(null);
    setIsSanityRunning(true);
    try {
      const result = await api.sanityCheck(yamlPath);
      setSanityResult(result);
    } catch (err) {
      setSanityError(err instanceof Error ? err.message : "Sanity check failed");
    } finally {
      setIsSanityRunning(false);
    }
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
              <Select
                label="Algorithm"
                id="algorithm-select"
                value={algorithm}
                onChange={(e) => setAlgorithm(e.target.value)}
                disabled={isRunning}
              >
                <option value="LhsDE">LhsDE</option>
                <option value="LHSSearch">LHSSearch (blind)</option>
                <option value="LogBFGSCMAPlus">LogBFGS CMA+</option>
              </Select>
              <div className="flex flex-col gap-1">
                <label
                  htmlFor="budget-input"
                  className="text-xs font-medium uppercase tracking-wide text-zinc-500"
                >
                  Budget
                </label>
                <input
                  id="budget-input"
                  type="number"
                  min={10}
                  max={2000}
                  value={runBudget}
                  onChange={(e) => setRunBudget(Number(e.target.value))}
                  disabled={isRunning}
                  className={selectCn()}
                />
              </div>
            </div>
            {appConfig && (
              <Select
                label="Preset Checkpoint (Replay)"
                id="checkpoint-select"
                value={replayCheckpoint}
                onChange={(e) => setReplayCheckpoint(e.target.value)}
                disabled={isRunning}
              >
                <option value="">— Live run (requires SPICE) —</option>
                {appConfig.preset_checkpoints.map((ck) => (
                  <option key={ck.id} value={ck.id}>
                    {ck.label}
                  </option>
                ))}
              </Select>
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
              <p role="alert" className="text-xs text-red-600">{startError}</p>
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
                Apply a project or select a preset checkpoint to enable run.
              </p>
            )}
          </PanelBody>
        </Panel>
      </div>

      {/* Sanity Check */}
      {isApplied && (
        <Panel>
          <PanelHeader className="flex items-center justify-between">
            <span className="text-sm font-semibold">Sanity Check</span>
            <Button
              onClick={handleSanityCheck}
              disabled={isSanityRunning || isRunning}
            >
              {isSanityRunning ? (
                <><Loader2 className="h-4 w-4 animate-spin" /> Testing…</>
              ) : (
                <><FlaskConical className="h-4 w-4" /> Run Sanity Check</>
              )}
            </Button>
          </PanelHeader>
          <PanelBody>
            {!sanityResult && !sanityError && !isSanityRunning && (
              <p className="text-xs text-zinc-400">
                Runs one SPICE simulation per testbench and one trial optimization step to verify the full pipeline before committing to a full run.
              </p>
            )}

            {isSanityRunning && (
              <p className="text-xs text-zinc-500">Running SPICE simulations — this may take a moment…</p>
            )}

            {sanityError && (
              <p role="alert" className="text-xs text-red-600">{sanityError}</p>
            )}

            {sanityResult && (
              <div className="space-y-3">
                {/* Overall status */}
                <div className="flex items-center gap-2">
                  {sanityResult.ok ? (
                    <CheckCircle2 className="h-5 w-5 text-green-500 shrink-0" />
                  ) : (
                    <XCircle className="h-5 w-5 text-red-500 shrink-0" />
                  )}
                  <span className={`text-sm font-medium ${sanityResult.ok ? "text-green-700" : "text-red-700"}`}>
                    {sanityResult.ok ? "All checks passed" : "One or more checks failed"}
                  </span>
                  {sanityResult.elapsed_ms_total != null && (
                    <Badge variant="neutral" className="ml-auto">
                      <Clock className="h-3 w-3" /> {fmtMs(sanityResult.elapsed_ms_total)} total
                    </Badge>
                  )}
                </div>

                {(sanityResult.elapsed_ms_load != null || sanityResult.elapsed_ms_optimizer_init != null || sanityResult.ngspice_path) && (
                  <div className="flex flex-wrap items-center gap-2 text-[10px] text-zinc-500">
                    {sanityResult.elapsed_ms_load != null && (
                      <span>YAML load: <span className="font-mono">{fmtMs(sanityResult.elapsed_ms_load)}</span></span>
                    )}
                    {sanityResult.elapsed_ms_optimizer_init != null && (
                      <span>· Optimizer init: <span className="font-mono">{fmtMs(sanityResult.elapsed_ms_optimizer_init)}</span></span>
                    )}
                    {sanityResult.ngspice_path && (
                      <span>· Simulator: <span className="font-mono">{sanityResult.ngspice_path}</span></span>
                    )}
                  </div>
                )}

                {sanityResult.error && (
                  <p className="text-xs text-red-600 bg-red-50 rounded p-2">{sanityResult.error}</p>
                )}

                {/* Per-testbench results */}
                {sanityResult.testbenches.length > 0 && (
                  <div className="space-y-1.5">
                    <p className="text-xs font-medium uppercase tracking-wide text-zinc-500">Testbenches</p>
                    {sanityResult.testbenches.map((tb) => (
                      <div key={tb.name} className="rounded border border-zinc-200 bg-zinc-50/50">
                        <div className="flex items-center gap-2 px-2 py-1.5">
                          {tb.ok ? (
                            <CheckCircle2 className="h-4 w-4 text-green-500 shrink-0" />
                          ) : (
                            <XCircle className="h-4 w-4 text-red-500 shrink-0" />
                          )}
                          <span className="text-xs font-mono">{tb.name}</span>
                          {tb.elapsed_ms != null && (
                            <span className="text-[10px] text-zinc-500"><Clock className="inline h-3 w-3 mr-0.5" />{fmtMs(tb.elapsed_ms)}</span>
                          )}
                          {tb.log_size_bytes != null && (
                            <span className="text-[10px] text-zinc-400">· {fmtBytes(tb.log_size_bytes)} log</span>
                          )}
                        </div>
                        {tb.error && (
                          <p className="text-xs text-red-600 px-2 pb-1.5">{tb.error}</p>
                        )}
                        {tb.log_tail && (
                          <details className="border-t border-zinc-200 px-2 py-1 text-[11px]">
                            <summary className="cursor-pointer text-zinc-500 hover:text-zinc-700">
                              <FileText className="inline h-3 w-3 mr-1" />
                              ngspice log tail
                              {tb.log_path && <span className="text-zinc-400 ml-1 font-mono text-[10px]">{tb.log_path}</span>}
                            </summary>
                            <pre className="mt-1 max-h-60 overflow-auto rounded bg-zinc-900 p-2 font-mono text-[10px] leading-tight text-zinc-100">{tb.log_tail}</pre>
                          </details>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {/* Trial evaluation results */}
                {sanityResult.trial && (
                  <div className="space-y-1.5">
                    <div className="flex items-center gap-2">
                      <p className="text-xs font-medium uppercase tracking-wide text-zinc-500">Trial Evaluation (1 iteration)</p>
                      {sanityResult.trial.elapsed_ms != null && (
                        <Badge variant="neutral"><Clock className="h-3 w-3" /> {fmtMs(sanityResult.trial.elapsed_ms)}</Badge>
                      )}
                    </div>
                    {sanityResult.trial.error ? (
                      <p className="text-xs text-red-600 bg-red-50 rounded p-2">{sanityResult.trial.error}</p>
                    ) : (
                      <>
                        {sanityResult.trial.score != null && (
                          <p className="text-xs text-zinc-600">
                            Score: <span className="font-mono font-medium">{sanityResult.trial.score.toFixed(4)}</span>
                          </p>
                        )}
                        {Object.keys(sanityResult.trial.metrics).length > 0 && (
                          <table className="w-full text-xs border-collapse mt-1">
                            <thead>
                              <tr className="text-zinc-400 text-left">
                                <th className="pb-1 pr-4 font-medium">Metric</th>
                                <th className="pb-1 pr-4 font-medium">Simulated</th>
                                <th className="pb-1 pr-4 font-medium">Target</th>
                                <th className="pb-1 font-medium">Status</th>
                              </tr>
                            </thead>
                            <tbody>
                              {enabledSpecs.map((spec) => {
                                const val = sanityResult.trial!.metrics[spec.name];
                                if (val == null) return null;
                                const tol = spec.tolerance ?? Math.abs(spec.target) * 0.05;
                                const passes =
                                  spec.goal === "exceed"
                                    ? val >= spec.target
                                    : spec.goal === "minimize"
                                      ? val <= spec.target
                                      : Math.abs(val - spec.target) <= tol;
                                return (
                                  <tr key={spec.name} className="border-t border-zinc-100">
                                    <td className="py-1 pr-4 font-mono">{spec.name}</td>
                                    <td className="py-1 pr-4 font-mono">{val.toPrecision(4)}</td>
                                    <td className="py-1 pr-4 text-zinc-400">
                                      {spec.goal === "exceed" ? "≥" : spec.goal === "minimize" ? "≤" : "≈"}{" "}
                                      {spec.target}
                                    </td>
                                    <td className="py-1">
                                      {passes ? (
                                        <CheckCircle2 className="h-3.5 w-3.5 text-green-500" />
                                      ) : (
                                        <XCircle className="h-3.5 w-3.5 text-red-500" />
                                      )}
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        )}
                        {Object.keys(sanityResult.trial.log_tails || {}).length > 0 && (
                          <div className="space-y-1 pt-1">
                            <p className="text-[10px] font-medium uppercase tracking-wide text-zinc-400">Trial sim logs</p>
                            {Object.entries(sanityResult.trial.log_tails).map(([name, tail]) => (
                              <details key={name} className="rounded border border-zinc-200 px-2 py-1 text-[11px]">
                                <summary className="cursor-pointer text-zinc-500 hover:text-zinc-700">
                                  <FileText className="inline h-3 w-3 mr-1" />
                                  <span className="font-mono">{name}</span>
                                  {sanityResult.trial!.log_files[name] && (
                                    <span className="text-zinc-400 ml-1 font-mono text-[10px]">{sanityResult.trial!.log_files[name]}</span>
                                  )}
                                </summary>
                                <pre className="mt-1 max-h-60 overflow-auto rounded bg-zinc-900 p-2 font-mono text-[10px] leading-tight text-zinc-100">{tail}</pre>
                              </details>
                            ))}
                          </div>
                        )}
                      </>
                    )}
                  </div>
                )}
              </div>
            )}
          </PanelBody>
        </Panel>
      )}

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
                className={selectCn("xs")}
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

function fmtMs(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms)) return "—";
  if (ms < 1) return `${ms.toFixed(2)} ms`;
  if (ms < 1000) return `${ms.toFixed(0)} ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(2)} s`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s - m * 60)}s`;
}

function fmtBytes(bytes: number | null | undefined): string {
  if (bytes == null) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
