"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle } from "lucide-react";
import { Panel, PanelBody, PanelHeader } from "@/components/ui/panel";
import { Badge } from "@/components/ui/badge";
import { useProjectStore } from "@/stores/projectStore";
import { api } from "@/lib/api";
import { formatEng } from "@/lib/utils";
import type { ScoreResponse, TargetSpec } from "@/types/api";
import { PenaltyCurveChart } from "@/components/charts/PenaltyCurveChart";

export function ScoreShapingTab() {
  const { summary, yamlPath, isApplied } = useProjectStore();
  const [selectedSpec, setSelectedSpec] = useState<string>("");
  const [metricValue, setMetricValue] = useState<number>(0);
  const [scoreData, setScoreData] = useState<ScoreResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const enabledSpecs = summary?.target_specs.filter((s) => s.enable) ?? [];

  // Init selected spec to first enabled spec when summary loads
  useEffect(() => {
    if (!summary) return;
    const first = enabledSpecs[0];
    if (first && !selectedSpec) {
      setSelectedSpec(first.name);
      setMetricValue(first.target);
    }
  }, [summary]); // eslint-disable-line react-hooks/exhaustive-deps

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
          const result = await api.computeScore(yamlPath, { [specName]: value }, specName);
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
      <div className="flex min-h-60 items-center justify-center rounded-lg border border-dashed border-zinc-300 bg-white text-sm text-zinc-400">
        Apply a project first to explore score shaping.
      </div>
    );
  }

  const currentSpecObj: TargetSpec | undefined = enabledSpecs.find((s) => s.name === selectedSpec);
  const rang =
    currentSpecObj?.range && currentSpecObj.range > 0
      ? currentSpecObj.range
      : Math.max(Math.abs(currentSpecObj?.target ?? 1), 1);
  const sliderMin = (currentSpecObj?.target ?? 0) - 3 * rang;
  const sliderMax = (currentSpecObj?.target ?? 0) + 3 * rang;

  const highestPenaltyEntry = scoreData
    ? Object.entries(scoreData.per_spec)
        .filter(([, s]) => (s.linear ?? 0) > 0)
        .sort(([, a], [, b]) => (b.linear ?? 0) - (a.linear ?? 0))[0]
    : null;

  return (
    <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
      {/* Left: spec selector + penalty curve + callout */}
      <div className="space-y-4">
        <Panel>
          <PanelHeader>
            <span className="text-sm font-semibold">Spec &amp; Value</span>
          </PanelHeader>
          <PanelBody className="space-y-4">
            <div>
              <label className="text-xs font-medium uppercase tracking-wide text-zinc-500">
                Spec
              </label>
              <select
                value={selectedSpec}
                onChange={(e) => handleSpecChange(e.target.value)}
                className="mt-1 block w-full rounded-md border border-zinc-300 bg-white px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500"
              >
                {enabledSpecs.map((s) => (
                  <option key={s.name} value={s.name}>
                    {s.name}
                  </option>
                ))}
              </select>
            </div>
            {currentSpecObj && (
              <div>
                <label className="block text-xs font-medium uppercase tracking-wide text-zinc-500">
                  Current value —{" "}
                  <span className="font-semibold text-indigo-600">{formatEng(metricValue)}</span>
                </label>
                <input
                  type="range"
                  min={sliderMin}
                  max={sliderMax}
                  step={(sliderMax - sliderMin) / 200}
                  value={metricValue}
                  onChange={(e) => setMetricValue(Number(e.target.value))}
                  className="mt-2 w-full accent-indigo-600"
                />
                <div className="flex justify-between text-xs text-zinc-400">
                  <span>{formatEng(sliderMin)}</span>
                  <span className="text-amber-600">
                    target: {formatEng(currentSpecObj.target)}
                  </span>
                  <span>{formatEng(sliderMax)}</span>
                </div>
              </div>
            )}
          </PanelBody>
        </Panel>

        {scoreData?.curve && (
          <PenaltyCurveChart curve={scoreData.curve} currentValue={metricValue} />
        )}

        {highestPenaltyEntry && (
          <Panel>
            <PanelBody>
              <div className="flex items-start gap-2">
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
                <p className="text-sm text-zinc-700">
                  <span className="font-semibold">{highestPenaltyEntry[0]}</span> has the
                  highest linear penalty (
                  {(highestPenaltyEntry[1].linear ?? 0).toFixed(3)}). Sigmoid shaping bounds
                  its contribution so severe violations don&apos;t dominate the aggregate
                  score.
                </p>
              </div>
            </PanelBody>
          </Panel>
        )}
      </div>

      {/* Right: per-spec breakdown table */}
      <Panel>
        <PanelHeader>
          <span className="text-sm font-semibold">Per-Spec Breakdown</span>
          {loading && <span className="text-xs text-zinc-400">Computing…</span>}
        </PanelHeader>
        <PanelBody className="p-0">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-zinc-100 bg-zinc-50 text-left text-zinc-500">
                <th className="px-3 py-2 font-medium">Spec</th>
                <th className="px-3 py-2 font-medium">Value</th>
                <th className="px-3 py-2 font-medium">Linear P̂</th>
                <th className="px-3 py-2 font-medium">Sigmoid P̂</th>
                <th className="px-3 py-2 font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {enabledSpecs.map((s) => {
                const entry = scoreData?.per_spec[s.name];
                return (
                  <tr
                    key={s.name}
                    className={`border-b border-zinc-50 last:border-0 hover:bg-zinc-50 ${selectedSpec === s.name ? "bg-indigo-50" : ""}`}
                  >
                    <td className="px-3 py-2 font-mono text-zinc-800">{s.name}</td>
                    <td className="px-3 py-2 font-mono text-zinc-600">
                      {entry?.value != null ? formatEng(entry.value) : "—"}
                    </td>
                    <td className="px-3 py-2 font-mono text-zinc-600">
                      {entry?.linear != null ? entry.linear.toFixed(4) : "—"}
                    </td>
                    <td className="px-3 py-2 font-mono text-zinc-600">
                      {entry?.sigmoid != null ? entry.sigmoid.toFixed(4) : "—"}
                    </td>
                    <td className="px-3 py-2">
                      {entry?.passes != null ? (
                        <Badge variant={entry.passes ? "pass" : "fail"}>
                          {entry.passes ? "pass" : "fail"}
                        </Badge>
                      ) : (
                        <Badge variant="neutral">n/a</Badge>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
            {scoreData && (
              <tfoot>
                <tr className="border-t border-zinc-200 bg-zinc-50 font-semibold">
                  <td className="px-3 py-2 text-zinc-700" colSpan={2}>
                    Aggregate F(x)
                  </td>
                  <td className="px-3 py-2 font-mono text-zinc-700">
                    {scoreData.aggregate.linear.toFixed(4)}
                  </td>
                  <td className="px-3 py-2 font-mono text-zinc-700">
                    {scoreData.aggregate.sigmoid.toFixed(4)}
                  </td>
                  <td />
                </tr>
              </tfoot>
            )}
          </table>
        </PanelBody>
      </Panel>
    </div>
  );
}
