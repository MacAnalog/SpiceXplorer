"use client";
import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { ArrowLeft, ArrowRight, Save, RotateCcw } from "lucide-react";
import { Panel, PanelBody, PanelHeader } from "@/components/ui/panel";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";
import { STEP_LABELS, WIZARD_STEPS, useWizardStore } from "@/stores/wizardStore";
import { BasicInfoStep } from "./steps/BasicInfoStep";
import { PDKRulesStep } from "./steps/PDKRulesStep";
import { DutParamsStep } from "./steps/DutParamsStep";
import { PVTStep } from "./steps/PVTStep";
import { TestbenchesStep } from "./steps/TestbenchesStep";
import { TargetSpecsStep } from "./steps/TargetSpecsStep";
import { OptimizerStep } from "./steps/OptimizerStep";
import { TextInput } from "./wizard-controls";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

interface Props {
  onSaved?: (savedPath: string) => void;
}

export function WizardShell({ onSaved }: Props) {
  const { form, stepIdx, setStep, next, back, reset } = useWizardStore();
  const [yamlText, setYamlText] = useState("");
  const [yamlErrors, setYamlErrors] = useState<string[]>([]);
  const [savePath, setSavePath] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);

  // Default save path tracks ws_root
  useEffect(() => {
    if (!savePath && form.project.ws_root) {
      setSavePath(`${form.project.ws_root.replace(/\/$/, "")}/project_setup.yaml`);
    }
  }, [form.project.ws_root, savePath]);

  // Debounced live YAML preview
  useEffect(() => {
    const t = setTimeout(async () => {
      try {
        const res = await api.generateProject(form);
        setYamlText(res.yaml);
        setYamlErrors(res.errors);
      } catch (e) {
        setYamlText("");
        setYamlErrors([e instanceof Error ? e.message : String(e)]);
      }
    }, 350);
    return () => clearTimeout(t);
  }, [form]);

  const StepBody = useMemo(() => {
    switch (WIZARD_STEPS[stepIdx]) {
      case "basic": return <BasicInfoStep />;
      case "pdk": return <PDKRulesStep />;
      case "dut": return <DutParamsStep />;
      case "pvt": return <PVTStep />;
      case "testbenches": return <TestbenchesStep />;
      case "specs": return <TargetSpecsStep />;
      case "optimizer": return <OptimizerStep />;
    }
  }, [stepIdx]);

  const handleSave = async () => {
    if (!savePath) {
      setSaveMsg("Specify a save path first.");
      return;
    }
    setSaving(true);
    setSaveMsg(null);
    try {
      const res = await api.generateProject(form, savePath);
      if (res.saved_path) {
        setSaveMsg(`Saved to ${res.saved_path}`);
        onSaved?.(res.saved_path);
      }
      setYamlErrors(res.errors);
    } catch (e) {
      setSaveMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
      {/* Left: step navigator + step body */}
      <div className="space-y-3">
        {/* Step pills */}
        <div className="flex flex-wrap gap-2">
          {WIZARD_STEPS.map((id, i) => {
            const active = i === stepIdx;
            return (
              <button
                key={id}
                type="button"
                onClick={() => setStep(i)}
                className={
                  "rounded-full border px-3 py-1 text-xs transition " +
                  (active
                    ? "border-indigo-500 bg-indigo-50 font-semibold text-indigo-700"
                    : "border-zinc-200 bg-white text-zinc-600 hover:bg-zinc-50")
                }
              >
                {i + 1}. {STEP_LABELS[id]}
              </button>
            );
          })}
        </div>

        {/* Progress bar */}
        <div className="h-1 w-full overflow-hidden rounded-full bg-zinc-100">
          <div
            className="h-full bg-indigo-500 transition-all"
            style={{ width: `${((stepIdx + 1) / WIZARD_STEPS.length) * 100}%` }}
          />
        </div>

        <Panel>{StepBody}</Panel>

        <div className="flex items-center justify-between">
          <Button variant="ghost" onClick={reset} className="!h-8 !px-2 !text-xs">
            <RotateCcw className="h-3 w-3" /> Reset wizard
          </Button>
          <div className="flex gap-2">
            <Button variant="secondary" onClick={back} disabled={stepIdx === 0}>
              <ArrowLeft className="h-3.5 w-3.5" /> Back
            </Button>
            <Button
              variant="secondary"
              onClick={next}
              disabled={stepIdx === WIZARD_STEPS.length - 1}
            >
              Next <ArrowRight className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      </div>

      {/* Right: live YAML preview + save controls */}
      <Panel>
        <PanelHeader>
          <div className="flex items-center gap-2 text-sm font-semibold">
            Live YAML preview
            {yamlErrors.length === 0 && yamlText
              ? <Badge variant="pass">valid</Badge>
              : yamlErrors.length > 0 && <Badge variant="fail">invalid</Badge>}
          </div>
        </PanelHeader>
        <PanelBody className="p-0">
          <MonacoEditor
            height="480px"
            language="yaml"
            value={yamlText}
            options={{
              minimap: { enabled: false },
              fontSize: 12,
              lineNumbers: "on",
              readOnly: true,
              scrollBeyondLastLine: false,
              wordWrap: "on",
              fontFamily: "ui-monospace, monospace",
            }}
            loading={<div className="flex h-60 items-center justify-center text-sm text-zinc-400">Rendering…</div>}
          />
        </PanelBody>
        {yamlErrors.length > 0 && (
          <div className="border-t border-zinc-100 px-3 py-2 text-xs text-red-600">
            {yamlErrors.map((e, i) => <div key={i}>• {e}</div>)}
          </div>
        )}
        <div className="border-t border-zinc-100 p-3">
          <div className="flex items-end gap-2">
            <label className="flex flex-1 flex-col gap-1">
              <span className="text-[10px] font-medium uppercase tracking-wide text-zinc-500">Save path</span>
              <TextInput
                value={savePath}
                onChange={(e) => setSavePath(e.target.value)}
                placeholder="/absolute/path/to/project_setup.yaml"
              />
            </label>
            <Button onClick={handleSave} disabled={saving || !yamlText}>
              <Save className="h-3.5 w-3.5" /> {saving ? "Saving…" : "Save YAML"}
            </Button>
          </div>
          {saveMsg && <div className="mt-2 text-xs text-zinc-600">{saveMsg}</div>}
        </div>
      </Panel>
    </div>
  );
}
