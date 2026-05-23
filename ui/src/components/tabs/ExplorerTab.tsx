"use client";
import { useEffect, useState } from "react";
import { Panel, PanelBody, PanelHeader } from "@/components/ui/panel";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useExplorerStore } from "@/stores/explorerStore";
import { useProjectStore } from "@/stores/projectStore";
import { api } from "@/lib/api";
import { formatEng } from "@/lib/utils";
import { COLORS } from "@/components/charts/PlotlyChart";
import { ScoreConvergenceChart } from "@/components/charts/ScoreConvergenceChart";
import { MetricConvergenceChart } from "@/components/charts/MetricConvergenceChart";
import { MetricScatterChart } from "@/components/charts/MetricScatterChart";
import { MetricHistogramChart } from "@/components/charts/MetricHistogramChart";
import { EmptyState } from "@/components/ui/empty-state";
import { selectCn } from "@/components/ui/select";
import { Thead, Th, Tr, Td } from "@/components/ui/table";
import type { ScatterPoint } from "@/types/api";

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
    setAvailableCheckpoints,
    setRunA,
    setRunB,
    setEnvelopeA,
    setScatterMetrics,
    setSelectedMetric,
  } = useExplorerStore();

  const [runAId, setRunAId] = useState<string>("");
  const [runBId, setRunBId] = useState<string>("");
  const [loadingA, setLoadingA] = useState(false);
  const [loadingB, setLoadingB] = useState(false);
  const [scatterPointsA, setScatterPointsA] = useState<ScatterPoint[]>([]);
  const [scatterPointsB, setScatterPointsB] = useState<ScatterPoint[]>([]);

  // Load checkpoint list on mount
  useEffect(() => {
    api.listCheckpoints().then(setAvailableCheckpoints).catch(console.error);
  }, [setAvailableCheckpoints]);

  const loadA = async () => {
    if (!runAId) return;
    setLoadingA(true);
    try {
      const data = await api.loadCheckpoint(runAId);
      setRunA(data);
      if (!selectedMetric && Object.keys(data.per_metric).length > 0) {
        setSelectedMetric(Object.keys(data.per_metric)[0]);
      }
      const env = await api.envelope(runAId, yamlPath || undefined);
      setEnvelopeA(env);
    } finally {
      setLoadingA(false);
    }
  };

  const loadB = async () => {
    if (!runBId) return;
    setLoadingB(true);
    try {
      const data = await api.loadCheckpoint(runBId);
      setRunB(data);
    } finally {
      setLoadingB(false);
    }
  };

  // Reload scatter when metrics or runs change
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

  const scoreRuns = [
    ...(runA
      ? [{ label: runA.label, scores: runA.scores, best_scores: runA.best_scores, color: COLORS.indigo }]
      : []),
    ...(runB
      ? [{ label: runB.label, scores: runB.scores, best_scores: runB.best_scores, color: COLORS.sky }]
      : []),
  ];

  const metricRuns = [
    ...(runA && runA.per_metric[selectedMetric]
      ? [{ label: runA.label, values: runA.per_metric[selectedMetric], color: COLORS.indigo }]
      : []),
    ...(runB && runB.per_metric[selectedMetric]
      ? [{ label: runB.label, values: runB.per_metric[selectedMetric], color: COLORS.sky }]
      : []),
  ];

  const histogramRuns = [
    ...(runA && runA.per_metric[selectedMetric]
      ? [{ label: runA.label, values: runA.per_metric[selectedMetric], color: COLORS.indigo }]
      : []),
    ...(runB && runB.per_metric[selectedMetric]
      ? [{ label: runB.label, values: runB.per_metric[selectedMetric], color: COLORS.sky }]
      : []),
  ];

  const scatterRuns = [
    ...(scatterPointsA.length ? [{ label: runA?.label ?? "Run A", points: scatterPointsA }] : []),
    ...(scatterPointsB.length ? [{ label: runB?.label ?? "Run B", points: scatterPointsB }] : []),
  ];

  const selectedSpecObj = summary?.target_specs.find((s) => s.name === selectedMetric);
  const targetX = summary?.target_specs.find((s) => s.name === scatterMetricX)?.target;
  const targetY = summary?.target_specs.find((s) => s.name === scatterMetricY)?.target;

  const hasData = !!(runA || runB);

  return (
    <div className="space-y-4">
      {/* Checkpoint Manager */}
      <Panel>
        <PanelHeader>
          <span className="text-sm font-semibold">Checkpoint Manager</span>
        </PanelHeader>
        <PanelBody>
          <div className="grid gap-4 sm:grid-cols-2">
            {/* Run A */}
            <div className="space-y-2">
              <label className="text-xs font-medium uppercase tracking-wide text-zinc-500">
                Run A
              </label>
              <div className="flex items-center gap-2">
                <select
                  value={runAId}
                  onChange={(e) => setRunAId(e.target.value)}
                  className="flex-1 rounded-md border border-zinc-300 bg-white px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500"
                >
                  <option value="">Select checkpoint…</option>
                  {availableCheckpoints.map((ck) => (
                    <option key={ck.id} value={ck.id}>
                      {ck.label}
                    </option>
                  ))}
                </select>
                <Button variant="secondary" onClick={loadA} disabled={!runAId || loadingA}>
                  {loadingA ? "Loading…" : "Load"}
                </Button>
              </div>
              {runA && <Badge variant="indigo">{runA.label} — {runA.n_iters} iters</Badge>}
            </div>

            {/* Run B */}
            <div className="space-y-2">
              <label className="text-xs font-medium uppercase tracking-wide text-zinc-500">
                Run B
              </label>
              <div className="flex items-center gap-2">
                <select
                  value={runBId}
                  onChange={(e) => setRunBId(e.target.value)}
                  className="flex-1 rounded-md border border-zinc-300 bg-white px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500"
                >
                  <option value="">Select checkpoint…</option>
                  {availableCheckpoints.map((ck) => (
                    <option key={ck.id} value={ck.id}>
                      {ck.label}
                    </option>
                  ))}
                </select>
                <Button variant="secondary" onClick={loadB} disabled={!runBId || loadingB}>
                  {loadingB ? "Loading…" : "Load"}
                </Button>
              </div>
              {runB && <Badge variant="neutral">{runB.label} — {runB.n_iters} iters</Badge>}
            </div>
          </div>
        </PanelBody>
      </Panel>

      {!hasData && (
        <EmptyState bordered>Load checkpoint(s) above to start exploring.</EmptyState>
      )}

      {/* Row 1: Score + Metric convergence */}
      {scoreRuns.length > 0 && (
        <div className="grid gap-4 xl:grid-cols-2">
          <ScoreConvergenceChart runs={scoreRuns} />
          <Panel>
            <PanelHeader>
              <span className="text-sm font-semibold">Metric Convergence</span>
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
        </div>
      )}

      {/* Row 2: Scatter + Envelope */}
      {hasData && (
        <div className="grid gap-4 xl:grid-cols-[1fr_380px]">
          <Panel>
            <PanelHeader>
              <span className="text-sm font-semibold">Metric Scatter</span>
              <div className="flex items-center gap-1.5">
                <select
                  value={scatterMetricX}
                  onChange={(e) => setScatterMetrics(e.target.value, scatterMetricY)}
                  className={selectCn("xs")}
                >
                  {allMetrics.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
                <span className="text-xs text-zinc-400">vs</span>
                <select
                  value={scatterMetricY}
                  onChange={(e) => setScatterMetrics(scatterMetricX, e.target.value)}
                  className={selectCn("xs")}
                >
                  {allMetrics.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>
            </PanelHeader>
            <PanelBody className="p-0">
              {scatterRuns.length > 0 ? (
                <MetricScatterChart
                  runs={scatterRuns}
                  metricX={scatterMetricX}
                  metricY={scatterMetricY}
                  targetX={targetX}
                  targetY={targetY}
                />
              ) : (
                <EmptyState minHeight="h-40">Select metrics to load scatter data</EmptyState>
              )}
            </PanelBody>
          </Panel>

          {envelopeA && (
            <Panel>
              <PanelHeader>
                <span className="text-sm font-semibold">Performance Envelope (Run A)</span>
              </PanelHeader>
              <PanelBody className="p-0">
                <div className="overflow-auto">
                  <table className="w-full text-xs">
                    <Thead>
                      <Th>Metric</Th>
                      <Th>Best Ever</Th>
                      <Th>Target</Th>
                      <Th>Pass</Th>
                    </Thead>
                    <tbody>
                      {envelopeA.map((e) => (
                        <Tr key={e.metric}>
                          <Td className="font-mono text-zinc-800">{e.metric}</Td>
                          <Td className="font-mono text-zinc-600">{formatEng(e.best_ever)}</Td>
                          <Td className="font-mono text-zinc-500">
                            {e.target != null ? formatEng(e.target) : "—"}
                          </Td>
                          <Td>
                            {e.passes != null ? (
                              <Badge variant={e.passes ? "pass" : "fail"}>
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
              </PanelBody>
            </Panel>
          )}
        </div>
      )}

      {/* Row 3: Histogram + Best params */}
      {hasData && allMetrics.length > 0 && (
        <div className="grid gap-4 xl:grid-cols-[1fr_320px]">
          <Panel>
            <PanelHeader>
              <span className="text-sm font-semibold">
                Metric Distribution — {selectedMetric}
              </span>
            </PanelHeader>
            <PanelBody className="p-0">
              {histogramRuns.length > 0 ? (
                <MetricHistogramChart
                  runs={histogramRuns}
                  metric={selectedMetric}
                  target={selectedSpecObj?.target}
                />
              ) : (
                <EmptyState minHeight="h-40">Load a checkpoint to see distribution</EmptyState>
              )}
            </PanelBody>
          </Panel>

          {runA && Object.keys(runA.params).length > 0 && (
            <Panel>
              <PanelHeader>
                <span className="text-sm font-semibold">Best Design Params (Run A)</span>
              </PanelHeader>
              <PanelBody className="p-0">
                <div className="max-h-60 overflow-auto">
                  <table className="w-full text-xs">
                    <Thead>
                      <Th>Param</Th>
                      <Th>Value</Th>
                    </Thead>
                    <tbody>
                      {Object.entries(runA.params).map(([name, vals]) => {
                        const lastVal = [...vals]
                          .reverse()
                          .find((v): v is number => v !== null);
                        return (
                          <Tr key={name}>
                            <Td className="font-mono text-zinc-800">{name}</Td>
                            <Td className="font-mono text-zinc-600">
                              {lastVal != null ? formatEng(lastVal) : "—"}
                            </Td>
                          </Tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </PanelBody>
            </Panel>
          )}
        </div>
      )}

      {/* Row 4: Spec summary */}
      {hasData && summary && (
        <Panel>
          <PanelHeader>
            <span className="text-sm font-semibold">Spec Summary</span>
          </PanelHeader>
          <PanelBody className="p-0">
            <div className="overflow-auto">
              <table className="w-full text-xs">
                <Thead>
                  <Th>Spec</Th>
                  <Th>Goal</Th>
                  <Th>Target</Th>
                  {runA && <Th>Run A Best</Th>}
                  {runB && <Th>Run B Best</Th>}
                </Thead>
                <tbody>
                  {summary.target_specs
                    .filter((s) => s.enable)
                    .map((s) => {
                      const aVals = runA?.per_metric[s.name]?.filter(
                        (v): v is number => v !== null,
                      );
                      const bVals = runB?.per_metric[s.name]?.filter(
                        (v): v is number => v !== null,
                      );
                      const aBest =
                        aVals?.length
                          ? s.goal === "minimize"
                            ? Math.min(...aVals)
                            : Math.max(...aVals)
                          : null;
                      const bBest =
                        bVals?.length
                          ? s.goal === "minimize"
                            ? Math.min(...bVals)
                            : Math.max(...bVals)
                          : null;
                      const tol = s.tolerance ?? Math.abs(s.target) * 0.05;
                      const aPasses =
                        aBest != null
                          ? s.goal === "exceed"
                            ? aBest >= s.target
                            : s.goal === "minimize"
                              ? aBest <= s.target
                              : Math.abs(aBest - s.target) <= tol
                          : null;
                      const bPasses =
                        bBest != null
                          ? s.goal === "exceed"
                            ? bBest >= s.target
                            : s.goal === "minimize"
                              ? bBest <= s.target
                              : Math.abs(bBest - s.target) <= tol
                          : null;
                      return (
                        <Tr key={s.name}>
                          <Td className="font-mono text-zinc-800">{s.name}</Td>
                          <Td>
                            <Badge
                              variant={
                                s.goal === "exceed"
                                  ? "indigo"
                                  : s.goal === "minimize"
                                    ? "warning"
                                    : "neutral"
                              }
                            >
                              {s.goal}
                            </Badge>
                          </Td>
                          <Td className="font-mono text-zinc-600">{formatEng(s.target)}</Td>
                          {runA && (
                            <Td>
                              {aBest != null ? (
                                <span className="inline-flex items-center gap-1 font-mono">
                                  {formatEng(aBest)}
                                  <Badge variant={aPasses ? "pass" : "fail"}>
                                    {aPasses ? "✓" : "✗"}
                                  </Badge>
                                </span>
                              ) : (
                                <span className="text-zinc-400">—</span>
                              )}
                            </Td>
                          )}
                          {runB && (
                            <Td>
                              {bBest != null ? (
                                <span className="inline-flex items-center gap-1 font-mono">
                                  {formatEng(bBest)}
                                  <Badge variant={bPasses ? "pass" : "fail"}>
                                    {bPasses ? "✓" : "✗"}
                                  </Badge>
                                </span>
                              ) : (
                                <span className="text-zinc-400">—</span>
                              )}
                            </Td>
                          )}
                        </Tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
          </PanelBody>
        </Panel>
      )}
    </div>
  );
}
