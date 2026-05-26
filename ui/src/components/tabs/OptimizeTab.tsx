"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { Panel, PanelBody, PanelHeader } from "@/components/ui/panel";
import { Button } from "@/components/ui/button";
import { useProjectStore } from "@/stores/projectStore";
import { useRunStore } from "@/stores/runStore";
import { api } from "@/lib/api";
import { COLORS } from "@/components/charts/PlotlyChart";
import { selectCn } from "@/components/ui/select";
import { ScoreConvergenceChart } from "@/components/charts/ScoreConvergenceChart";
import { MetricConvergenceChart } from "@/components/charts/MetricConvergenceChart";
import { Toolbar, ToolbarLabel, ToolbarSpacer } from "@/components/shell/Toolbar";
import { Separator } from "@/components/ui/separator";
import { Segmented } from "@/components/ui/segmented";
import { Stat } from "@/components/ui/stat";
import { SpecChip } from "@/components/ui/spec-chip";
import { Thead, Th, Tr, Td } from "@/components/ui/table";
import { EmptyState } from "@/components/ui/empty-state";
import { formatEng, statusForGoal } from "@/lib/utils";
import type { AppConfig, TargetSpec } from "@/types/api";

interface Props {
  appConfig: AppConfig | null;
}

type ScoreFn = "sigmoid" | "linear";

function passesSpec(s: TargetSpec, val: number | undefined): boolean | null {
  if (val == null) return null;
  return statusForGoal(s.goal, val, s.target, s.tolerance ?? undefined) === "pass";
}

function goalSym(g: string): string {
  return g === "exceed" ? ">" : g === "minimize" ? "<" : "≈";
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
    bestParams,
    currentIter,
    startRun,
    pushEvent,
    stopRun,
    reset,
  } = useRunStore();

  const [algorithm, setAlgorithm] = useState("LhsDE");
  const [runBudget, setRunBudget] = useState(200);
  const [scoreFn, setScoreFn] = useState<ScoreFn>("sigmoid");
  const [replayCheckpoint, setReplayCheckpoint] = useState<string>("");
  const [selectedMetric, setSelectedMetric] = useState<string>("");
  const [startError, setStartError] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const enabledSpecs = useMemo(
    () => summary?.target_specs.filter((s) => s.enable) ?? [],
    [summary],
  );

  useEffect(() => {
    if (enabledSpecs.length > 0 && !selectedMetric) {
      setSelectedMetric(enabledSpecs[0].name);
    }
  }, [enabledSpecs, selectedMetric]);

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

  useEffect(() => () => { eventSourceRef.current?.close(); }, []);

  const canStart = isApplied || !!replayCheckpoint;

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

  // Stat strip: best F(x) delta vs first
  const firstScore = events.find((e) => e.best_score != null)?.best_score ?? null;
  const lastBest = [...events].reverse().find((e) => e.best_score != null)?.best_score ?? null;
  const bestDelta = firstScore != null && lastBest != null ? lastBest - firstScore : null;

  const specsMet = enabledSpecs.filter((s) => passesSpec(s, bestMetrics[s.name]));

  const recentEvents = events.slice(-4).reverse();

  return (
    <>
      <Toolbar>
        <ToolbarLabel>algorithm</ToolbarLabel>
        <select
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
          <Button variant="danger" onClick={handleStop}>
            <span className="h-1.5 w-1.5 rounded-full bg-white obs-pulse" /> Stop
          </Button>
        ) : (
          <Button variant="primary" onClick={handleStart} disabled={!canStart}>
            {replayCheckpoint ? "Replay" : "Start"}
          </Button>
        )}
        <Button variant="default" disabled={isRunning || events.length === 0}>
          + Save checkpoint
        </Button>
      </Toolbar>

      <div className="flex min-h-0 flex-1 flex-col gap-2.5 overflow-auto p-3">
        {startError && (
          <div role="alert" className="rounded-md border border-danger bg-danger-soft px-3 py-2 text-xs text-danger">
            {startError}
          </div>
        )}

        {!canStart && !events.length && (
          <EmptyState bordered minHeight="min-h-32">
            Apply a project or select a preset checkpoint to enable a run.
          </EmptyState>
        )}

        {/* Stat strip */}
        <div className="grid grid-cols-3 gap-2.5">
          <Stat
            eyebrow={`iteration · budget ${budget || runBudget}`}
            value={`${currentIter} / ${budget || runBudget}`}
            progress={budget > 0 ? currentIter / budget : 0}
          />
          <Stat
            eyebrow={`best F(x) · ${scoreFn}`}
            value={lastBest != null ? lastBest.toFixed(3) : "—"}
            delta={
              bestDelta != null
                ? `${bestDelta >= 0 ? "+" : ""}${bestDelta.toFixed(3)} since iter 0`
                : undefined
            }
            deltaTone={bestDelta != null && bestDelta < 0 ? "up" : "neutral"}
          />
          <Stat
            eyebrow={`specs met · ${enabledSpecs.length} total`}
            value={`${specsMet.length} / ${enabledSpecs.length}`}
            delta={specsMet.map((s) => s.name).join(" · ") || "none yet"}
          />
        </div>

        {/* Top row: F(x) convergence + metric convergence */}
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

        {/* Bottom row: live specs + best design */}
        <div className="grid grid-cols-2 gap-2.5">
          <Panel>
            <PanelHeader
              title="live spec status"
              mute={`· @ iter ${currentIter}`}
              right={
                <span className="font-mono text-[10px] text-muted">
                  {specsMet.length} / {enabledSpecs.length} met
                </span>
              }
            />
            <PanelBody>
              <div className="flex flex-wrap gap-1.5">
                {enabledSpecs.length === 0 && (
                  <span className="text-[11px] text-muted">No specs.</span>
                )}
                {enabledSpecs.map((s) => {
                  const val = bestMetrics[s.name];
                  const p = passesSpec(s, val);
                  return (
                    <SpecChip
                      key={s.name}
                      name={s.name}
                      value={val != null ? formatEng(val) : "—"}
                      target={`${goalSym(s.goal)}${formatEng(s.target)}`}
                      status={p === true ? "ok" : p === false ? "fail" : "neutral"}
                    />
                  );
                })}
              </div>
              <div className="mt-2.5 space-y-0.5 font-mono text-[11px] text-muted">
                {recentEvents.length === 0 && (
                  <div className="text-faint">No events yet.</div>
                )}
                {recentEvents.map((e, i) => {
                  const score = e.score;
                  const isError = e.error != null;
                  const isBest =
                    e.best_score != null &&
                    score != null &&
                    Math.abs(e.best_score - score) < 1e-9;
                  return (
                    <div key={i}>
                      <span>[{e.iter ?? "—"}] </span>
                      {isError ? (
                        <span className="text-danger">eval ✗ {e.error}</span>
                      ) : score != null ? (
                        <>
                          <span>eval ✓ score={score.toFixed(3)}</span>
                          {isBest && (
                            <span className="ml-1 text-primary">★ best</span>
                          )}
                        </>
                      ) : (
                        <span className="text-faint">…</span>
                      )}
                    </div>
                  );
                })}
              </div>
            </PanelBody>
          </Panel>
          <Panel className="flex min-h-0 flex-col">
            <PanelHeader
              title="best design"
              mute={currentIter > 0 ? `· iter ${currentIter}` : ""}
              right={
                <Button variant="default" disabled={Object.keys(bestParams).length === 0}>
                  Export .spice
                </Button>
              }
            />
            <div className="min-h-0 flex-1 overflow-auto">
              {Object.keys(bestParams).length === 0 ? (
                <EmptyState minHeight="min-h-32">No best design yet.</EmptyState>
              ) : (
                <table className="w-full">
                  <Thead>
                    <Th>param</Th>
                    <Th>value</Th>
                    <Th>min..max</Th>
                  </Thead>
                  <tbody>
                    {summary?.dut_params.map((p) => {
                      const val = bestParams[p.name];
                      return (
                        <Tr key={p.name}>
                          <Td className="font-mono">{p.name}</Td>
                          <Td className="font-mono">
                            {val != null ? formatEng(val) : "—"}
                          </Td>
                          <Td className="font-mono text-muted">
                            {formatEng(p.min_val)} .. {formatEng(p.max_val)}
                          </Td>
                        </Tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
            </div>
          </Panel>
        </div>

      </div>
    </>
  );
}
