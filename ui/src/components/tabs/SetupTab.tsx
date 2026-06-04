"use client";
import { useCallback, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { Panel, PanelBody, PanelHeader } from "@/components/ui/panel";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useProjectStore } from "@/stores/projectStore";
import { useWizardStore } from "@/stores/wizardStore";
import { api } from "@/lib/api";
import { formatEng } from "@/lib/utils";
import { EmptyState } from "@/components/ui/empty-state";
import { selectCn } from "@/components/ui/select";
import { Thead, Th, Tr, Td } from "@/components/ui/table";
import { Segmented } from "@/components/ui/segmented";
import { Toolbar, ToolbarSpacer } from "@/components/shell/Toolbar";
import { Separator } from "@/components/ui/separator";
import type { AppConfig } from "@/types/api";
import { WizardShell } from "@/components/wizard/WizardShell";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

interface Props {
  appConfig: AppConfig | null;
}

type SetupMode = "load" | "wizard";

function goalSym(g: string): string {
  return g === "exceed" ? ">" : g === "minimize" ? "<" : "≈";
}

export function SetupTab({ appConfig }: Props) {
  const {
    yaml,
    yamlPath,
    summary,
    validationErrors,
    isApplied,
    setYaml,
    setYamlPath,
    setValidationErrors,
    apply,
  } = useProjectStore();
  const [loading, setLoading] = useState(false);
  const [validating, setValidating] = useState(false);
  const [mode, setMode] = useState<SetupMode>("load");
  const [hydrateError, setHydrateError] = useState<string | null>(null);
  const [showSummary, setShowSummary] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const { setForm } = useWizardStore();

  const loadDemoYaml = useCallback(
    async (path: string) => {
      setLoading(true);
      try {
        const result = await api.loadProject(path);
        if (result.ok) {
          apply(result.summary, path);
          const rawRes = await fetch(
            `/api/yaml-text?path=${encodeURIComponent(path)}`,
          );
          if (rawRes.ok) setYaml(await rawRes.text());
        }
      } finally {
        setLoading(false);
      }
    },
    [apply, setYaml],
  );

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
    if (!yamlPath && !yaml.trim()) return;
    setLoading(true);
    try {
      // Prefer the on-disk path (keeps relative netlist/ws_root resolution for
      // live runs); fall back to the editor content for uploaded/edited YAML.
      const res = yamlPath
        ? await api.loadProject(yamlPath)
        : await api.loadProjectContent(yaml);
      if (res.ok) apply(res.summary, res.yaml_path);
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
      // Uploaded content has no on-disk path; clear any stale path so Apply uses
      // the uploaded YAML (and the button enables on content alone).
      setYamlPath("");
      setYaml(text);
    };
    reader.readAsText(file);
  };

  const editInWizard = useCallback(async () => {
    setHydrateError(null);
    if (!yaml.trim() && !yamlPath) {
      setHydrateError("Load a YAML first.");
      return;
    }
    try {
      const res = yaml.trim()
        ? await api.parseProjectToForm({ yaml_content: yaml })
        : await api.parseProjectToForm({ yaml_path: yamlPath });
      setForm(res.form);
      setMode("wizard");
    } catch (e) {
      setHydrateError(e instanceof Error ? e.message : String(e));
    }
  }, [yaml, yamlPath, setForm]);

  const handleWizardSaved = useCallback(
    async (path: string) => {
      setMode("load");
      await loadDemoYaml(path);
    },
    [loadDemoYaml],
  );

  const lineCount = yaml.split("\n").length;
  const isValid = validationErrors.length === 0 && !!yaml.trim();

  return (
    <>
      <Toolbar>
        <Segmented
          value={mode}
          onChange={(m) => setMode(m as SetupMode)}
          options={[
            { value: "load", label: "Load / Edit YAML" },
            { value: "wizard", label: "Create Wizard" },
          ]}
        />
        <Separator />
        {mode === "load" ? (
          <>
            {appConfig && (
              <select
                className={selectCn("sm")}
                value=""
                onChange={(e) => {
                  if (e.target.value) loadDemoYaml(e.target.value);
                }}
              >
                <option value="">Load example…</option>
                <option value={appConfig.default_yaml}>
                  OTA Cascode (default)
                </option>
              </select>
            )}
            <label className="cursor-pointer" aria-label="Upload YAML file">
              <input
                type="file"
                accept=".yaml,.yml"
                className="hidden"
                onChange={handleUpload}
              />
              <span className="inline-flex h-[26px] cursor-pointer items-center rounded border border-border bg-panel px-2.5 text-[11px] font-medium hover:bg-hairline">
                Upload .yaml
              </span>
            </label>
            <Button variant="default" onClick={editInWizard} disabled={!yaml && !yamlPath}>
              Edit in wizard
            </Button>
          </>
        ) : null}
        <ToolbarSpacer />
        <Button
          variant={showSummary ? "primary" : "default"}
          onClick={() => setShowSummary((v) => !v)}
          disabled={!summary}
          title={
            summary
              ? showSummary
                ? "Hide project summary panels"
                : "Show project summary, target specs, and schematic"
              : "Apply a project first"
          }
        >
          {showSummary ? "Hide summary" : "Show summary"}
        </Button>
        {mode === "load" && (
          <>
            <Button
              variant="default"
              onClick={handleValidate}
              disabled={validating || !yaml}
            >
              {validating ? "Checking…" : "Validate"}
            </Button>
            <Button
              variant="primary"
              onClick={handleApply}
              disabled={loading || (!yamlPath && !yaml.trim())}
            >
              {loading ? "Loading…" : "Apply →"}
            </Button>
          </>
        )}
      </Toolbar>

      <div
        className="grid min-h-0 flex-1 gap-3 overflow-auto p-3"
        style={{
          // Summary panels appear on the right only when the user toggles
          // them on (and a project is loaded).
          gridTemplateColumns: showSummary && summary ? "1fr 1fr" : "1fr",
        }}
      >
        {/* LEFT: Monaco editor or Wizard */}
        {mode === "load" ? (
          <Panel className="flex min-h-0 flex-col">
            <PanelHeader
              title="project_setup.yaml"
              mute={yamlPath ? `· ${yamlPath}` : ""}
              right={
                <>
                  <span className="font-mono text-[10px] text-muted">
                    UTF-8 · {lineCount} lines
                  </span>
                  {yaml.trim() && (
                    <Badge variant={isValid ? "ok" : "fail"} dot>
                      {isValid ? "valid" : "invalid"}
                    </Badge>
                  )}
                </>
              }
            />
            <div className="min-h-0 flex-1">
              <MonacoEditor
                height="100%"
                language="yaml"
                value={yaml}
                onChange={handleEditorChange}
                theme="vs-dark"
                options={{
                  minimap: { enabled: false },
                  fontSize: 11.5,
                  lineNumbers: "on",
                  scrollBeyondLastLine: false,
                  wordWrap: "on",
                  fontFamily: "JetBrains Mono, ui-monospace, monospace",
                  renderLineHighlight: "none",
                }}
                loading={
                  <div className="flex h-60 items-center justify-center text-xs text-faint">
                    Loading editor…
                  </div>
                }
              />
            </div>
            {(validationErrors.length > 0 || hydrateError) && (
              <div className="border-t border-border bg-danger-soft px-3 py-2">
                {validationErrors.map((e, i) => (
                  <div key={i} className="font-mono text-[11px] text-danger">
                    • {e}
                  </div>
                ))}
                {hydrateError && (
                  <div className="font-mono text-[11px] text-danger">
                    • {hydrateError}
                  </div>
                )}
              </div>
            )}
          </Panel>
        ) : (
          <div className="flex min-h-0 min-w-0 flex-col">
            <WizardShell onSaved={handleWizardSaved} defaultSavePath={yamlPath} />
          </div>
        )}

        {/* RIGHT: stacked summary panels — toggled via the Show summary button */}
        {showSummary && summary && (
        <div className="flex min-w-0 flex-col gap-2.5">
          {!summary ? (
            <Panel>
              <PanelBody>
                <EmptyState minHeight="min-h-32">
                  Apply a project to see the parsed summary.
                </EmptyState>
              </PanelBody>
            </Panel>
          ) : (
            <>
              <Panel>
                <PanelHeader
                  title="project summary"
                  mute="· live preview"
                  right={
                    <span className="font-mono text-[10px] text-muted">
                      {summary.name}
                    </span>
                  }
                />
                <PanelBody>
                  <dl
                    className="grid gap-y-1 gap-x-3 text-xs"
                    style={{ gridTemplateColumns: "100px 1fr" }}
                  >
                    <dt className="text-muted">name</dt>
                    <dd className="m-0 font-mono text-[11px]">{summary.name}</dd>
                    <dt className="text-muted">simulator</dt>
                    <dd className="m-0 font-mono text-[11px]">
                      {summary.simulator}
                    </dd>
                    <dt className="text-muted">technology</dt>
                    <dd className="m-0 font-mono text-[11px]">
                      {summary.tech.name || "—"}
                    </dd>
                    <dt className="text-muted">workspace</dt>
                    <dd className="m-0 font-mono text-[11px]">
                      {summary.ws_root || "—"}
                    </dd>
                    <dt className="text-muted">params</dt>
                    <dd className="m-0 font-mono text-[11px]">
                      {summary.dut_params.length} ·{" "}
                      {summary.dut_params.filter((p) => p.log_scale).length} log ·{" "}
                      {summary.dut_params.filter((p) => p.is_integer).length} int ·{" "}
                      {summary.dut_params.filter((p) => p.freeze).length} frozen
                    </dd>
                    <dt className="text-muted">pvt corners</dt>
                    <dd className="m-0 font-mono text-[11px]">
                      {summary.pvt_corners.length}
                    </dd>
                    <dt className="text-muted">testbenches</dt>
                    <dd className="m-0 font-mono text-[11px]">
                      {summary.testbenches.length} ·{" "}
                      {summary.testbenches.filter((t) => t.enable).length} enabled
                    </dd>
                    <dt className="text-muted">specs</dt>
                    <dd className="m-0 font-mono text-[11px]">
                      {summary.target_specs.length} · weight Σ ={" "}
                      {summary.target_specs
                        .reduce((a, s) => a + (s.weight ?? 0), 0)
                        .toFixed(1)}
                    </dd>
                    <dt className="text-muted">optimizer</dt>
                    <dd className="m-0 font-mono text-[11px]">
                      {summary.optimizer.name} · budget{" "}
                      {summary.optimizer.budget}
                    </dd>
                  </dl>
                  {isApplied && (
                    <div className="mt-2.5">
                      <Badge variant="ok" dot>
                        applied
                      </Badge>
                    </div>
                  )}
                </PanelBody>
              </Panel>

              <Panel>
                <PanelHeader
                  title="target specs"
                  mute={`· ${summary.target_specs.length}`}
                />
                <table className="w-full">
                  <Thead>
                    <Th>name</Th>
                    <Th>goal</Th>
                    <Th>target</Th>
                    <Th>tol</Th>
                    <Th>weight</Th>
                  </Thead>
                  <tbody>
                    {summary.target_specs.map((s) => (
                      <Tr key={s.name}>
                        <Td className="font-mono">{s.name}</Td>
                        <Td className="font-mono">{goalSym(s.goal)}</Td>
                        <Td className="font-mono">{formatEng(s.target)}</Td>
                        <Td className="font-mono">
                          {s.tolerance != null ? `±${formatEng(s.tolerance)}` : "—"}
                        </Td>
                        <Td className="font-mono">{(s.weight ?? 0).toFixed(1)}</Td>
                      </Tr>
                    ))}
                  </tbody>
                </table>
              </Panel>

              <Panel>
                <PanelHeader
                  title="device under test"
                  mute="· schematic"
                  right={
                    <span className="font-mono text-[10px] text-muted">
                      {summary.netlist || "—"}
                    </span>
                  }
                />
                <div className="bg-hairline/40 p-1.5">
                  <SchematicImg />
                </div>
              </Panel>
            </>
          )}
        </div>
        )}
      </div>
    </>
  );
}

function SchematicImg() {
  const [errored, setErrored] = useState(false);
  if (errored) {
    return (
      <div className="flex h-32 items-center justify-center text-[11px] text-faint">
        No schematic available.
      </div>
    );
  }
  return (
    <img
      src={api.schematicUrl()}
      alt="DUT schematic"
      className="max-h-[220px] w-full object-contain"
      onError={() => setErrored(true)}
    />
  );
}
