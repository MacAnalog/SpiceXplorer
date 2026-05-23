"use client";
import { useCallback, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { CheckCircle, AlertCircle, Upload, Zap } from "lucide-react";
import { Panel, PanelBody, PanelHeader } from "@/components/ui/panel";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useProjectStore } from "@/stores/projectStore";
import { api } from "@/lib/api";
import { formatEng } from "@/lib/utils";
import { EmptyState } from "@/components/ui/empty-state";
import { selectCn } from "@/components/ui/select";
import { Thead, Th, Tr, Td } from "@/components/ui/table";
import type { AppConfig } from "@/types/api";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

interface Props {
  appConfig: AppConfig | null;
}

export function SetupTab({ appConfig }: Props) {
  const { yaml, yamlPath, summary, validationErrors, isApplied, setYaml, setValidationErrors, apply } =
    useProjectStore();
  const [loading, setLoading] = useState(false);
  const [validating, setValidating] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const loadDemoYaml = useCallback(async (path: string) => {
    setLoading(true);
    try {
      // Read via backend project/load then display YAML
      const result = await api.loadProject(path);
      if (result.ok) {
        apply(result.summary, path);
        // Load raw text via a simple fetch from our Next.js API proxy
        const rawRes = await fetch(`/api/yaml-text?path=${encodeURIComponent(path)}`);
        if (rawRes.ok) {
          setYaml(await rawRes.text());
        }
      }
    } finally {
      setLoading(false);
    }
  }, [apply, setYaml]);

  // Debounced live validation
  const handleEditorChange = useCallback(
    (value: string | undefined) => {
      const v = value ?? "";
      setYaml(v);
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(async () => {
        if (!v.trim()) return;
        setValidating(true);
        try {
          const res = await api.validateYaml(v);
          setValidationErrors(res.errors);
        } finally {
          setValidating(false);
        }
      }, 600);
    },
    [setYaml, setValidationErrors],
  );

  const handleValidate = async () => {
    if (!yaml.trim()) return;
    setValidating(true);
    try {
      const res = await api.validateYaml(yaml);
      setValidationErrors(res.errors);
    } finally {
      setValidating(false);
    }
  };

  const handleApply = async () => {
    if (!yamlPath) return;
    setLoading(true);
    try {
      const res = await api.loadProject(yamlPath);
      if (res.ok) apply(res.summary, yamlPath);
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target?.result as string;
      setYaml(text);
    };
    reader.readAsText(file);
  };

  return (
    <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
      {/* Left: YAML editor */}
      <Panel>
        <PanelHeader>
          <div className="flex items-center gap-2 text-sm font-semibold">
            <Zap className="h-4 w-4 text-indigo-500" />
            Project YAML
          </div>
          <div className="flex items-center gap-2">
            {appConfig && (
              <select
                className={selectCn("xs")}
                value=""
                onChange={(e) => { if (e.target.value) loadDemoYaml(e.target.value); }}
              >
                <option value="">Load example...</option>
                <option value={appConfig.default_yaml}>OTA Cascode (default)</option>
              </select>
            )}
            <label className="cursor-pointer" aria-label="Upload YAML file">
              <input type="file" accept=".yaml,.yml" className="hidden" onChange={handleUpload} />
              <span className="inline-flex items-center gap-1 rounded-md border border-zinc-300 bg-white px-2 py-1 text-xs text-zinc-600 hover:bg-zinc-50">
                <Upload className="h-3 w-3" /> Upload
              </span>
            </label>
            <Button variant="secondary" onClick={handleValidate} disabled={validating || !yaml}>
              {validating ? "Checking…" : "Validate"}
            </Button>
            <Button onClick={handleApply} disabled={loading || !yamlPath}>
              {loading ? "Loading…" : "Apply"}
            </Button>
          </div>
        </PanelHeader>
        <PanelBody className="p-0">
          <MonacoEditor
            height="460px"
            language="yaml"
            value={yaml}
            onChange={handleEditorChange}
            options={{
              minimap: { enabled: false },
              fontSize: 12,
              lineNumbers: "on",
              scrollBeyondLastLine: false,
              wordWrap: "on",
              fontFamily: "ui-monospace, monospace",
            }}
            loading={<div className="flex h-60 items-center justify-center text-sm text-zinc-400">Loading editor…</div>}
          />
        </PanelBody>
        {/* Validation status */}
        <div className="border-t border-zinc-100 p-3">
          {validationErrors.length === 0 && yaml ? (
            <div className="flex items-center gap-2 text-xs text-emerald-600">
              <CheckCircle className="h-3.5 w-3.5" /> Valid schema
            </div>
          ) : validationErrors.length > 0 ? (
            <div role="alert" className="space-y-1">
              {validationErrors.map((e, i) => (
                <div key={i} className="flex items-start gap-2 text-xs text-red-600">
                  <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" /> {e}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-xs text-zinc-400">No YAML loaded yet — select an example or upload a file.</div>
          )}
        </div>
      </Panel>

      {/* Right: Parsed summary */}
      <div className="space-y-4">
        {!summary ? (
          <Panel>
            <PanelBody>
              <EmptyState>Apply a project to see the parsed summary.</EmptyState>
            </PanelBody>
          </Panel>
        ) : (
          <>
            {/* Project meta */}
            <Panel>
              <PanelHeader>
                <span className="text-sm font-semibold">Project</span>
                {isApplied && <Badge variant="pass">Applied</Badge>}
              </PanelHeader>
              <PanelBody>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <MetaItem label="Name" value={summary.name} />
                  <MetaItem label="Simulator" value={summary.simulator} />
                  <MetaItem label="Technology" value={summary.tech.name} />
                  <MetaItem label="Budget" value={String(summary.optimizer.budget)} />
                  <MetaItem label="Algorithm" value={summary.optimizer.name} />
                  <MetaItem label="PVT Corners" value={String(summary.pvt_corners.length)} />
                </div>
              </PanelBody>
            </Panel>

            {/* Testbenches */}
            <Panel>
              <PanelHeader>
                <span className="text-sm font-semibold">Testbenches ({summary.testbenches.length})</span>
              </PanelHeader>
              <PanelBody>
                <div className="space-y-1.5">
                  {summary.testbenches.map((tb) => (
                    <div key={tb.name} className="flex items-center justify-between rounded-md border border-zinc-100 bg-zinc-50 px-3 py-2 text-xs">
                      <span className="font-medium text-zinc-800">{tb.name}</span>
                      <div className="flex items-center gap-2">
                        {tb.description && <span className="text-zinc-400">{tb.description}</span>}
                        <Badge variant={tb.enable ? "pass" : "neutral"}>{tb.enable ? "enabled" : "disabled"}</Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </PanelBody>
            </Panel>

            {/* DUT Params table */}
            <Panel>
              <PanelHeader>
                <span className="text-sm font-semibold">DUT Parameters ({summary.dut_params.length})</span>
              </PanelHeader>
              <PanelBody className="p-0">
                <div className="overflow-auto">
                  <table className="w-full text-xs">
                    <Thead>
                      <Th>Name</Th>
                      <Th>Min</Th>
                      <Th>Max</Th>
                      <Th>Flags</Th>
                    </Thead>
                    <tbody>
                      {summary.dut_params.map((p) => (
                        <Tr key={p.name}>
                          <Td className="font-mono text-zinc-800">{p.name}</Td>
                          <Td className="font-mono text-zinc-600">{formatEng(p.min_val)}</Td>
                          <Td className="font-mono text-zinc-600">{formatEng(p.max_val)}</Td>
                          <Td>
                            <div className="flex gap-1">
                              {p.is_integer && <Badge variant="indigo">int</Badge>}
                              {p.log_scale && <Badge variant="neutral">log</Badge>}
                            </div>
                          </Td>
                        </Tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </PanelBody>
            </Panel>

            {/* Target Specs */}
            <Panel>
              <PanelHeader>
                <span className="text-sm font-semibold">Target Specs ({summary.target_specs.length})</span>
              </PanelHeader>
              <PanelBody className="p-0">
                <table className="w-full text-xs">
                  <Thead>
                    <Th>Name</Th>
                    <Th>Goal</Th>
                    <Th>Target</Th>
                    <Th>TB</Th>
                  </Thead>
                  <tbody>
                    {summary.target_specs.map((s) => (
                      <Tr key={s.name}>
                        <Td className="font-mono text-zinc-800">{s.name}</Td>
                        <Td>
                          <Badge variant={s.goal === "exceed" ? "indigo" : s.goal === "minimize" ? "warning" : "neutral"}>
                            {s.goal}
                          </Badge>
                        </Td>
                        <Td className="font-mono text-zinc-600">{formatEng(s.target)}</Td>
                        <Td className="text-zinc-500">{s.testbench}</Td>
                      </Tr>
                    ))}
                  </tbody>
                </table>
              </PanelBody>
            </Panel>
          </>
        )}
      </div>
    </div>
  );
}

function MetaItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-zinc-100 bg-zinc-50 p-2.5">
      <div className="text-[10px] font-medium uppercase tracking-wide text-zinc-400">{label}</div>
      <div className="mt-0.5 text-xs font-semibold text-zinc-800">{value}</div>
    </div>
  );
}
