"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Panel, PanelBody, PanelHeader } from "@/components/ui/panel";
import { useProjectStore } from "@/stores/projectStore";
import { api } from "@/lib/api";
import { formatEng } from "@/lib/utils";
import type { ScoreResponse, TargetSpec } from "@/types/api";
import { PenaltyCurveChart } from "@/components/charts/PenaltyCurveChart";
import { EmptyState } from "@/components/ui/empty-state";
import { selectCn } from "@/components/ui/select";
import { Thead, Th, Tr, Td } from "@/components/ui/table";
import { Slider } from "@/components/ui/slider";
import { Toolbar, ToolbarLabel, ToolbarSpacer } from "@/components/shell/Toolbar";
import { Separator } from "@/components/ui/separator";

function goalSym(g: string): string {
  if (g === "exceed") return ">";
  if (g === "minimize") return "<";
  return "≈";
}

export function ScoreShapingTab() {
  const { summary, yamlPath, isApplied } = useProjectStore();
  const [selectedSpec, setSelectedSpec] = useState<string>("");
  const [metricValue, setMetricValue] = useState<number>(0);
  const [scoreData, setScoreData] = useState<ScoreResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const enabledSpecs = useMemo(
    () => summary?.target_specs.filter((s) => s.enable) ?? [],
    [summary],
  );

  useEffect(() => {
    if (!summary) return;
    const first = enabledSpecs[0];
    if (first && !selectedSpec) {
      setSelectedSpec(first.name);
      setMetricValue(first.target);
    }
  }, [summary, enabledSpecs, selectedSpec]);

  const handleSpecChange = (specName: string) => {
    setSelectedSpec(specName);
    const spec = enabledSpecs.find((s) => s.name === specName);
    if (spec) setMetricValue(spec.target);
  };

  const computeScore = useCallback(
    (specName: string, value: number) => {
      if (!yamlPath || !specName) return;
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(async () => {
        setLoading(true);
        try {
          const result = await api.computeScore(
            yamlPath,
            { [specName]: value },
            specName,
          );
          setScoreData(result);
        } finally {
          setLoading(false);
        }
      }, 150);
    },
    [yamlPath],
  );

  useEffect(() => {
    computeScore(selectedSpec, metricValue);
  }, [selectedSpec, metricValue, computeScore]);

  if (!isApplied || !summary) {
    return (
      <div className="flex-1 p-3">
        <EmptyState bordered minHeight="min-h-60">
          Apply a project first to explore score shaping.
        </EmptyState>
      </div>
    );
  }

  const currentSpec: TargetSpec | undefined = enabledSpecs.find(
    (s) => s.name === selectedSpec,
  );
  const range =
    currentSpec?.range && currentSpec.range > 0
      ? currentSpec.range
      : Math.max(Math.abs(currentSpec?.target ?? 1), 1) * 0.5;
  const target = currentSpec?.target ?? 0;
  const sliderMin = target - 3 * range;
  const sliderMax = target + 3 * range;

  const aggregate = scoreData?.aggregate;
  const weightSum = enabledSpecs.reduce((a, s) => a + (s.weight ?? 0), 0);

  // Find dominating spec under each shaping
  const dominantSig = scoreData
    ? Object.entries(scoreData.per_spec)
        .map(([k, v]) => [k, (v.sigmoid ?? 0) * (v.weight ?? 0)] as const)
        .sort(([, a], [, b]) => b - a)[0]
    : null;
  const dominantLin = scoreData
    ? Object.entries(scoreData.per_spec)
        .map(([k, v]) => [k, (v.linear ?? 0) * (v.weight ?? 0)] as const)
        .sort(([, a], [, b]) => b - a)[0]
    : null;

  return (
    <>
      <Toolbar>
        <ToolbarLabel>spec</ToolbarLabel>
        <select
          value={selectedSpec}
          onChange={(e) => handleSpecChange(e.target.value)}
          className={selectCn("sm")}
        >
          {enabledSpecs.map((s) => (
            <option key={s.name} value={s.name}>
              {s.name} · {goalSym(s.goal)} {formatEng(s.target)}
            </option>
          ))}
        </select>
        <Separator />
        <ToolbarLabel>range</ToolbarLabel>
        <span className="font-mono text-[11px] text-fg">
          target ± 3 × {formatEng(range)}
        </span>
        <ToolbarSpacer />
        <span className="font-mono text-[11px] text-muted">
          POST /api/score · 150ms debounce
        </span>
      </Toolbar>

      <div
        className="grid min-h-0 flex-1 gap-3 overflow-auto p-3"
        style={{ gridTemplateColumns: "1.5fr 1fr" }}
      >
        {/* Left: penalty chart + slider + callout */}
        <div className="flex min-w-0 flex-col gap-2.5">
          <Panel>
            <PanelHeader
              title="penalty curve"
              mute="· sigmoid vs linear"
              right={
                currentSpec && (
                  <span className="font-mono text-[10px] text-muted">
                    {currentSpec.name} · {goalSym(currentSpec.goal)}{" "}
                    {formatEng(currentSpec.target)}
                  </span>
                )
              }
            />
            <PanelBody>
              {scoreData?.curve ? (
                <PenaltyCurveChart
                  curve={scoreData.curve}
                  currentValue={metricValue}
                />
              ) : (
                <EmptyState minHeight="h-[280px]">
                  {loading ? "Computing…" : "No curve data."}
                </EmptyState>
              )}
              {currentSpec && (
                <div className="mt-2.5">
                  <Slider
                    value={metricValue}
                    min={sliderMin}
                    max={sliderMax}
                    step={(sliderMax - sliderMin) / 200}
                    onChange={setMetricValue}
                    markerValue={target}
                    markerLabel="target"
                  />
                  <div className="mt-1.5 flex items-center justify-between font-mono text-[11px]">
                    <span className="text-muted">{formatEng(sliderMin)}</span>
                    <span className="font-medium text-fg">
                      now {formatEng(metricValue)}
                    </span>
                    <span className="text-muted">{formatEng(sliderMax)}</span>
                  </div>
                </div>
              )}
            </PanelBody>
          </Panel>

          {dominantSig && dominantLin && (
            <div className="flex gap-2.5 rounded-md border-l-2 border-primary bg-primary-soft px-3 py-2.5 text-xs text-fg">
              <span className="font-mono text-sm text-primary">◆</span>
              <div>
                Under <b className="text-primary">sigmoid</b> shaping,{" "}
                <span className="font-mono">{dominantSig[0]}</span> dominates
                F(x) (weighted P̂ = {dominantSig[1].toFixed(3)}). Under{" "}
                <b className="text-secondary">linear</b> shaping,{" "}
                <span className="font-mono">{dominantLin[0]}</span> dominates
                (weighted P̂ = {dominantLin[1].toFixed(3)}). Switching shaping
                rebalances which constraint the optimizer chases.
              </div>
            </div>
          )}
        </div>

        {/* Right: per-spec breakdown */}
        <Panel className="flex min-w-0 flex-col">
          <PanelHeader
            title="per-spec breakdown"
            mute="· F(x) = Σ wᵢ · P̂ᵢ"
            right={
              loading && (
                <span aria-live="polite" className="text-[10px] text-muted">
                  computing…
                </span>
              )
            }
          />
          <div className="min-h-0 flex-1 overflow-auto">
            <table className="w-full">
              <Thead>
                <Th>spec</Th>
                <Th>current</Th>
                <Th>sigmoid P̂</Th>
                <Th>linear P̂</Th>
                <Th>w</Th>
              </Thead>
              <tbody>
                {enabledSpecs.map((s) => {
                  const entry = scoreData?.per_spec[s.name];
                  const passes = entry?.passes;
                  return (
                    <Tr key={s.name} highlight={s.name === selectedSpec}>
                      <Td className="font-mono">
                        <span className={s.name === selectedSpec ? "font-medium" : ""}>
                          {s.name}
                        </span>
                      </Td>
                      <Td
                        className={
                          "font-mono " +
                          (passes === true
                            ? "text-ok"
                            : passes === false
                              ? "text-danger"
                              : "text-fg")
                        }
                      >
                        {entry?.value != null ? formatEng(entry.value) : "—"}
                      </Td>
                      <Td className="font-mono">
                        <PenaltyBar
                          value={entry?.sigmoid ?? null}
                          color="bg-primary"
                        />
                      </Td>
                      <Td className="font-mono">
                        <PenaltyBar
                          value={entry?.linear ?? null}
                          color="bg-secondary"
                        />
                      </Td>
                      <Td className="font-mono">{(s.weight ?? 0).toFixed(1)}</Td>
                    </Tr>
                  );
                })}
              </tbody>
              {aggregate && (
                <tfoot>
                  <tr className="border-t border-border bg-hairline font-medium">
                    <td className="px-2.5 py-1.5 font-mono text-[11px]" colSpan={2}>
                      F(x) aggregate
                    </td>
                    <td className="px-2.5 py-1.5 font-mono text-[11px] text-primary">
                      {aggregate.sigmoid.toFixed(3)}
                    </td>
                    <td className="px-2.5 py-1.5 font-mono text-[11px] text-secondary">
                      {aggregate.linear.toFixed(3)}
                    </td>
                    <td className="px-2.5 py-1.5 font-mono text-[11px]">
                      Σ {weightSum.toFixed(1)}
                    </td>
                  </tr>
                </tfoot>
              )}
            </table>
          </div>
        </Panel>
      </div>
    </>
  );
}

function PenaltyBar({
  value,
  color,
}: {
  value: number | null;
  color: string;
}) {
  const pct =
    value == null
      ? 0
      : Math.max(0, Math.min(1, value)) * 100;
  return (
    <span className="inline-flex items-center">
      <span className="relative mr-1.5 inline-block h-1 w-9 rounded-sm bg-hairline align-middle">
        <span
          className={`absolute left-0 top-0 h-full rounded-sm ${color}`}
          style={{ width: `${pct}%` }}
        />
      </span>
      {value != null ? value.toFixed(2) : "—"}
    </span>
  );
}
