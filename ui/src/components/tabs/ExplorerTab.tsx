"use client";
import { useEffect, useMemo, useState } from "react";
import { Panel, PanelBody, PanelHeader } from "@/components/ui/panel";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useExplorerStore } from "@/stores/explorerStore";
import { useProjectStore } from "@/stores/projectStore";
import { api } from "@/lib/api";
import { formatEng, statusForGoal } from "@/lib/utils";
import { COLORS } from "@/components/charts/PlotlyChart";
import { ScoreConvergenceChart } from "@/components/charts/ScoreConvergenceChart";
import { MetricConvergenceChart } from "@/components/charts/MetricConvergenceChart";
import { MetricScatterChart } from "@/components/charts/MetricScatterChart";
import { MetricHistogramChart } from "@/components/charts/MetricHistogramChart";
import { EmptyState } from "@/components/ui/empty-state";
import { selectCn } from "@/components/ui/select";
import { Thead, Th, Tr, Td } from "@/components/ui/table";
import type { ScatterPoint, TargetSpec } from "@/types/api";
import { Toolbar, ToolbarLabel, ToolbarSpacer } from "@/components/shell/Toolbar";
import { Separator } from "@/components/ui/separator";

function goalSym(g: string): string {
  return g === "exceed" ? ">" : g === "minimize" ? "<" : "≈";
}

function bestOf(values: number[], goal: string, target?: number | string | null): number | null {
  if (!values.length) return null;
  // reduce, not Math.min(...values): spreading a long run's array as call
  // arguments overflows the stack (RangeError) above ~65k points.
  if (goal === "exact" && target != null && Number.isFinite(Number(target))) {
    // "Best" for an exact target is the sample CLOSEST to it — not the max, which an
    // outlier (e.g. 120° for a 60°±10° phase-margin spec) would otherwise win.
    const t = Number(target);
    return values.reduce((m, v) => (Math.abs(v - t) < Math.abs(m - t) ? v : m), values[0]);
  }
  return values.reduce((m, v) => (goal === "minimize" ? (v < m ? v : m) : v > m ? v : m), values[0]);
}

function passesSpec(s: TargetSpec, val: number | null): boolean | null {
  if (val == null) return null;
  return statusForGoal(s.goal, val, s.target, s.tolerance ?? undefined) === "pass";
}

export function ExplorerTab() {
  const { summary, yamlPath } = useProjectStore();
  const {
    availableCheckpoints,
    runA,
    runB,
    envelopeA,
    scatterMetricX,
    scatterMetricY,
    selectedMetric,
    setRunA,
    setRunB,
    setEnvelopeA,
    setScatterMetrics,
    setSelectedMetric,
  } = useExplorerStore();

  const [runAId, setRunAId] = useState<string>("");
  const [runBId, setRunBId] = useState<string>("");
  const [loadingBoth, setLoadingBoth] = useState(false);
  const [scatterPointsA, setScatterPointsA] = useState<ScatterPoint[]>([]);
  const [scatterPointsB, setScatterPointsB] = useState<ScatterPoint[]>([]);

  // Replace any selection (selectedMetric / scatter X / Y) that the freshly
  // loaded checkpoint doesn't actually contain, so non-cascode projects and
  // checkpoint switches don't leave the metric/scatter panels stuck empty.
  const reconcileMetrics = (keys: string[]) => {
    if (keys.length === 0) return;
    if (!keys.includes(selectedMetric)) setSelectedMetric(keys[0]);
    const x = keys.includes(scatterMetricX) ? scatterMetricX : keys[0];
    const y = keys.includes(scatterMetricY) ? scatterMetricY : keys[1] ?? keys[0];
    if (x !== scatterMetricX || y !== scatterMetricY) setScatterMetrics(x, y);
  };

  const loadBoth = async () => {
    if (!runAId && !runBId) return;
    setLoadingBoth(true);
    try {
      const tasks: Promise<unknown>[] = [];
      if (runAId) {
        tasks.push(
          api.loadCheckpoint(runAId).then(async (data) => {
            setRunA(data);
            reconcileMetrics(Object.keys(data.per_metric));
            const env = await api.envelope(runAId, yamlPath || undefined);
            setEnvelopeA(env);
          }),
        );
      } else {
        // Deselected: clear the stale A slot so it stops rendering.
        setRunA(null);
        setEnvelopeA(null);
        setScatterPointsA([]);
      }
      if (runBId) {
        tasks.push(
          api.loadCheckpoint(runBId).then((data) => {
            setRunB(data);
            if (!runAId) reconcileMetrics(Object.keys(data.per_metric));
          }),
        );
      } else {
        setRunB(null);
        setScatterPointsB([]);
      }
      await Promise.all(tasks);
    } finally {
      setLoadingBoth(false);
    }
  };

  useEffect(() => {
    if (!runA || !scatterMetricX || !scatterMetricY) return;
    api
      .scatter(runA.id, scatterMetricX, scatterMetricY, yamlPath || undefined)
      .then((r) => setScatterPointsA(r.points))
      .catch(() => setScatterPointsA([]));
  }, [runA, scatterMetricX, scatterMetricY, yamlPath]);

  useEffect(() => {
    if (!runB || !scatterMetricX || !scatterMetricY) return;
    api
      .scatter(runB.id, scatterMetricX, scatterMetricY, yamlPath || undefined)
      .then((r) => setScatterPointsB(r.points))
      .catch(() => setScatterPointsB([]));
  }, [runB, scatterMetricX, scatterMetricY, yamlPath]);

  const allMetrics = runA
    ? Object.keys(runA.per_metric)
    : runB
      ? Object.keys(runB.per_metric)
      : [];

  const scoreRuns = useMemo(
    () => [
      ...(runA
        ? [
            {
              label: runA.label,
              scores: runA.scores,
              best_scores: runA.best_scores,
              color: COLORS.primary,
              showRaw: false,
            },
          ]
        : []),
      ...(runB
        ? [
            {
              label: runB.label,
              scores: runB.scores,
              best_scores: runB.best_scores,
              color: COLORS.secondary,
              showRaw: false,
            },
          ]
        : []),
    ],
    [runA, runB],
  );

  const metricRuns = [
    ...(runA && runA.per_metric[selectedMetric]
      ? [{ label: runA.label, values: runA.per_metric[selectedMetric], color: COLORS.primary }]
      : []),
    ...(runB && runB.per_metric[selectedMetric]
      ? [{ label: runB.label, values: runB.per_metric[selectedMetric], color: COLORS.secondary }]
      : []),
  ];

  const histogramRuns = metricRuns;

  // Always emit BOTH slots (A first, B second) so the chart's positional color
  // (A=indigo, B=cyan) stays stable even when one run has no scatter points.
  const scatterRuns = [
    { label: runA?.label ?? "A", points: scatterPointsA },
    { label: runB?.label ?? "B", points: scatterPointsB },
  ];
  const hasScatter = scatterPointsA.length > 0 || scatterPointsB.length > 0;

  const selectedSpecObj = summary?.target_specs.find((s) => s.name === selectedMetric);
  const targetX = summary?.target_specs.find((s) => s.name === scatterMetricX)?.target;
  const targetY = summary?.target_specs.find((s) => s.name === scatterMetricY)?.target;
  const goalX = summary?.target_specs.find((s) => s.name === scatterMetricX)?.goal;
  const goalY = summary?.target_specs.find((s) => s.name === scatterMetricY)?.goal;

  const hasData = !!(runA || runB);

  // Envelope table data
  const enabledSpecs = summary?.target_specs.filter((s) => s.enable) ?? [];
  const envelopeRows = enabledSpecs.map((s) => {
    const aVals = runA?.per_metric[s.name]?.filter((v): v is number => v != null) ?? [];
    const bVals = runB?.per_metric[s.name]?.filter((v): v is number => v != null) ?? [];
    const aBest = bestOf(aVals, s.goal, s.target);
    const bBest = bestOf(bVals, s.goal, s.target);
    // Only a true head-to-head (both runs loaded, values differ) has a winner.
    // A tie, or only one run loaded, is neutral — previously ties silently went
    // to B and a single run was always declared the winner.
    let winner: "A" | "B" | null = null;
    if (aBest != null && bBest != null && aBest !== bBest) {
      const t = Number(s.target);
      if (s.goal === "exact" && Number.isFinite(t)) {
        // Closest-to-target wins; equidistant-but-different samples (e.g. 50 vs 70 for
        // target 60) are a neutral tie, not an arbitrary "B".
        const da = Math.abs(aBest - t);
        const db = Math.abs(bBest - t);
        winner = da === db ? null : da < db ? "A" : "B";
      } else {
        winner = (s.goal === "minimize" ? aBest < bBest : aBest > bBest) ? "A" : "B";
      }
    }
    return { spec: s, aBest, bBest, winner };
  });

  const totalEvals = (runA?.n_iters ?? 0) + (runB?.n_iters ?? 0);

  return (
    <>
      <Toolbar>
        <ToolbarLabel>run A</ToolbarLabel>
        <select
          value={runAId}
          onChange={(e) => setRunAId(e.target.value)}
          className={selectCn("sm") + " w-[200px]"}
        >
          <option value="">— none —</option>
          {availableCheckpoints.map((ck) => (
            <option key={ck.id} value={ck.id}>
              {ck.label}
            </option>
          ))}
        </select>
        <ToolbarLabel>run B</ToolbarLabel>
        <select
          value={runBId}
          onChange={(e) => setRunBId(e.target.value)}
          className={selectCn("sm") + " w-[200px]"}
        >
          <option value="">— none —</option>
          {availableCheckpoints.map((ck) => (
            <option key={ck.id} value={ck.id}>
              {ck.label}
            </option>
          ))}
        </select>
        <Button variant="default" onClick={loadBoth} disabled={loadingBoth || (!runAId && !runBId)}>
          {loadingBoth ? "Loading…" : "Load both"}
        </Button>
        <Separator />
        <span className="font-mono text-[11px] text-muted">
          {[runA, runB].filter(Boolean).length} runs
          {totalEvals > 0 && <> · {totalEvals} evals</>}
        </span>
        <ToolbarSpacer />
        <Button variant="default" disabled={!hasData}>
          Export CSV
        </Button>
        <Button variant="primary" disabled={!hasData}>
          Compare report
        </Button>
      </Toolbar>

      {/* [&>*]:shrink-0 — keep Panels (overflow-hidden) from being flex-crushed/clipped;
          the container scrolls instead. */}
      <div className="flex min-h-0 flex-1 flex-col gap-2.5 overflow-auto p-3 [&>*]:shrink-0">
        {!hasData && (
          <EmptyState bordered minHeight="min-h-32">
            Pick run A and/or B and click Load both to start exploring.
          </EmptyState>
        )}

        {/* Row 1: F(x) overlay + metric overlay */}
        {hasData && (
          <div className="grid grid-cols-2 gap-2.5">
            <Panel>
              <PanelHeader
                title="F(x) convergence"
                mute="· A vs B"
                right={
                  <span className="font-mono text-[10px]">
                    {runA && (
                      <span style={{ color: COLORS.primary }}>● A</span>
                    )}{" "}
                    {runB && (
                      <span style={{ color: COLORS.secondary }}>● B</span>
                    )}
                  </span>
                }
              />
              <PanelBody>
                <ScoreConvergenceChart runs={scoreRuns} />
              </PanelBody>
            </Panel>
            <Panel>
              <PanelHeader
                title={
                  <>
                    <span className="font-mono">{selectedMetric || "—"}</span> · best-so-far
                  </>
                }
                right={
                  <select
                    value={selectedMetric}
                    onChange={(e) => setSelectedMetric(e.target.value)}
                    className={selectCn("xs")}
                  >
                    {allMetrics.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                }
              />
              <PanelBody>
                {metricRuns.length > 0 ? (
                  <MetricConvergenceChart
                    metric={selectedMetric}
                    runs={metricRuns}
                    target={selectedSpecObj?.target}
                    goal={selectedSpecObj?.goal}
                  />
                ) : (
                  <EmptyState minHeight="h-[220px]">No data.</EmptyState>
                )}
              </PanelBody>
            </Panel>
          </div>
        )}

        {/* Row 2: scatter + envelope */}
        {hasData && (
          <div className="grid grid-cols-2 gap-2.5">
            <Panel>
              <PanelHeader
                title="metric scatter"
                right={
                  <div className="flex items-center gap-1.5">
                    <select
                      value={scatterMetricX}
                      onChange={(e) =>
                        setScatterMetrics(e.target.value, scatterMetricY)
                      }
                      className={selectCn("xs")}
                    >
                      {allMetrics.map((m) => (
                        <option key={m} value={m}>
                          {m}
                        </option>
                      ))}
                    </select>
                    <span className="text-[10px] text-muted">vs</span>
                    <select
                      value={scatterMetricY}
                      onChange={(e) =>
                        setScatterMetrics(scatterMetricX, e.target.value)
                      }
                      className={selectCn("xs")}
                    >
                      {allMetrics.map((m) => (
                        <option key={m} value={m}>
                          {m}
                        </option>
                      ))}
                    </select>
                  </div>
                }
              />
              <PanelBody>
                {hasScatter ? (
                  <MetricScatterChart
                    runs={scatterRuns}
                    metricX={scatterMetricX}
                    metricY={scatterMetricY}
                    targetX={targetX}
                    targetY={targetY}
                    goalX={goalX}
                    goalY={goalY}
                  />
                ) : (
                  <EmptyState minHeight="h-[240px]">No scatter data.</EmptyState>
                )}
              </PanelBody>
            </Panel>

            <Panel className="flex min-h-0 flex-col">
              <PanelHeader title="performance envelope" mute="· best per spec" />
              <div className="min-h-0 flex-1 overflow-auto">
                {envelopeRows.length === 0 ? (
                  <EmptyState minHeight="min-h-32">No specs.</EmptyState>
                ) : (
                  <table className="w-full">
                    <Thead>
                      <Th>spec</Th>
                      <Th>target</Th>
                      <Th>run A best</Th>
                      <Th>run B best</Th>
                      <Th>winner</Th>
                    </Thead>
                    <tbody>
                      {envelopeRows.map(({ spec, aBest, bBest, winner }) => (
                        <Tr key={spec.name}>
                          <Td className="font-mono">{spec.name}</Td>
                          <Td className="font-mono text-muted">
                            {goalSym(spec.goal)} {formatEng(spec.target)}
                          </Td>
                          <Td
                            className="font-mono"
                            style={
                              winner === "A" ? { color: COLORS.primary, fontWeight: 500 } : {}
                            }
                          >
                            {aBest != null ? formatEng(aBest) : "—"}
                          </Td>
                          <Td
                            className="font-mono"
                            style={
                              winner === "B" ? { color: COLORS.secondary, fontWeight: 500 } : {}
                            }
                          >
                            {bBest != null ? formatEng(bBest) : "—"}
                          </Td>
                          <Td>
                            {winner ? (
                              <Badge variant={winner === "A" ? "indigo" : "cyan"}>
                                {winner}
                              </Badge>
                            ) : (
                              <Badge variant="neutral">—</Badge>
                            )}
                          </Td>
                        </Tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </Panel>
          </div>
        )}

        {/* Row 3: histogram + spec summary */}
        {hasData && summary && (
          <div className="grid grid-cols-2 gap-2.5">
            <Panel>
              <PanelHeader
                title={
                  <>
                    <span className="font-mono">{selectedMetric}</span> distribution
                  </>
                }
                right={
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[10px]">
                      {runA && (
                        <span style={{ color: COLORS.primary }}>● A</span>
                      )}{" "}
                      {runB && (
                        <span style={{ color: COLORS.secondary }}>● B</span>
                      )}
                    </span>
                    <select
                      value={selectedMetric}
                      onChange={(e) => setSelectedMetric(e.target.value)}
                      className={selectCn("xs")}
                    >
                      {allMetrics.map((m) => (
                        <option key={m} value={m}>
                          {m}
                        </option>
                      ))}
                    </select>
                  </div>
                }
              />
              <PanelBody>
                {histogramRuns.length > 0 ? (
                  <MetricHistogramChart
                    runs={histogramRuns}
                    metric={selectedMetric}
                    target={selectedSpecObj?.target}
                  />
                ) : (
                  <EmptyState minHeight="h-[220px]">No data.</EmptyState>
                )}
              </PanelBody>
            </Panel>

            <Panel className="flex min-h-0 flex-col">
              <PanelHeader title="spec summary" mute="· pass / fail" />
              <div className="min-h-0 flex-1 overflow-auto">
                <table className="w-full">
                  <Thead>
                    <Th>spec</Th>
                    <Th>goal</Th>
                    {runA && <Th>A</Th>}
                    {runB && <Th>B</Th>}
                  </Thead>
                  <tbody>
                    {enabledSpecs.map((s) => {
                      const aVals = runA?.per_metric[s.name]?.filter(
                        (v): v is number => v != null,
                      );
                      const bVals = runB?.per_metric[s.name]?.filter(
                        (v): v is number => v != null,
                      );
                      const aBest = aVals?.length ? bestOf(aVals, s.goal, s.target) : null;
                      const bBest = bVals?.length ? bestOf(bVals, s.goal, s.target) : null;
                      const aPass = passesSpec(s, aBest);
                      const bPass = passesSpec(s, bBest);
                      return (
                        <Tr key={s.name}>
                          <Td className="font-mono">{s.name}</Td>
                          <Td className="font-mono text-muted">
                            {goalSym(s.goal)} {formatEng(s.target)}
                          </Td>
                          {runA && (
                            <Td>
                              {aPass == null ? (
                                <Badge variant="neutral" dot>
                                  —
                                </Badge>
                              ) : (
                                <Badge variant={aPass ? "ok" : "fail"} dot>
                                  {aPass ? "pass" : "fail"}
                                </Badge>
                              )}
                            </Td>
                          )}
                          {runB && (
                            <Td>
                              {bPass == null ? (
                                <Badge variant="neutral" dot>
                                  —
                                </Badge>
                              ) : (
                                <Badge variant={bPass ? "ok" : "fail"} dot>
                                  {bPass ? "pass" : "fail"}
                                </Badge>
                              )}
                            </Td>
                          )}
                        </Tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Panel>
          </div>
        )}

        {envelopeA && (
          <Panel>
            <PanelHeader title="envelope (run A · raw)" mute="· server-side computed" />
            <div className="overflow-auto">
              <table className="w-full">
                <Thead>
                  <Th>metric</Th>
                  <Th>best ever</Th>
                  <Th>target</Th>
                  <Th>pass</Th>
                </Thead>
                <tbody>
                  {envelopeA.map((e) => (
                    <Tr key={e.metric}>
                      <Td className="font-mono">{e.metric}</Td>
                      <Td className="font-mono">{formatEng(e.best_ever)}</Td>
                      <Td className="font-mono text-muted">
                        {e.target != null ? formatEng(e.target) : "—"}
                      </Td>
                      <Td>
                        {e.passes != null ? (
                          <Badge variant={e.passes ? "ok" : "fail"} dot>
                            {e.passes ? "yes" : "no"}
                          </Badge>
                        ) : (
                          <Badge variant="neutral">—</Badge>
                        )}
                      </Td>
                    </Tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        )}
      </div>
    </>
  );
}
