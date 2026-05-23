"use client";
import { useState } from "react";
import { ChevronDown, ChevronRight, Plus, Trash2 } from "lucide-react";
import { useWizardStore } from "@/stores/wizardStore";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { StepHeader, TextInput, Field } from "../wizard-controls";
import { selectCn } from "@/components/ui/select";
import type { WizardTargetSpec } from "@/types/api";

const SIM_TYPES: WizardTargetSpec["sim_type"][] = ["ac", "dc", "op", "tran", "noise", "noise_spectrum"];
const GOALS: WizardTargetSpec["goal"][] = ["exceed", "minimize", "exact"];
const ERROR_TYPES = ["relative-absolute", "absolute", "squared", "relative-squared", "relative-exponential", "relative-sigmoid"];
const REWARD_TYPES = ["none", "log", "relative-log", "relative-absolute", "absolute", "relative-sigmoid"];

function blankSpec(testbench = ""): WizardTargetSpec {
  return {
    name: "",
    testbench,
    sim_type: "ac",
    goal: "exceed",
    target: "",
    range: "",
    tolerance: "",
    weight: "1.0",
    log_scale: false,
    error_type: "relative-absolute",
    reward_type: "none",
    enable: true,
    description: "",
  };
}

export function TargetSpecsStep() {
  const { form, setTargetSpecs } = useWizardStore();
  const specs = form.target_specs;
  const tbNames = form.testbenches.map((t) => t.name).filter(Boolean);
  const [openIdx, setOpenIdx] = useState<number | null>(0);

  const add = () => {
    const next = [...specs, blankSpec(tbNames[0] ?? "")];
    setTargetSpecs(next);
    setOpenIdx(next.length - 1);
  };
  const remove = (i: number) => {
    setTargetSpecs(specs.filter((_, idx) => idx !== i));
    if (openIdx === i) setOpenIdx(null);
  };
  const update = (i: number, patch: Partial<WizardTargetSpec>) =>
    setTargetSpecs(specs.map((s, idx) => (idx === i ? { ...s, ...patch } : s)));

  return (
    <div>
      <StepHeader
        title="Target Specs"
        description="Each spec maps a measurement to a goal/target/tolerance — these drive the loss function."
      />
      <div className="space-y-3 p-4">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-zinc-600">Specs ({specs.length})</span>
          <Button variant="secondary" onClick={add} className="!h-7 !px-2 !text-xs">
            <Plus className="h-3 w-3" /> Add spec
          </Button>
        </div>

        {specs.length === 0 && (
          <div className="rounded-md border border-dashed border-zinc-300 bg-zinc-50 p-4 text-center text-xs text-zinc-500">
            No specs yet — add at least one for the optimizer to chase.
          </div>
        )}

        {specs.map((s, i) => {
          const isOpen = openIdx === i;
          return (
            <div key={i} className="rounded-md border border-zinc-200 bg-white">
              <button
                type="button"
                className="flex w-full items-center justify-between px-3 py-2 text-left text-xs hover:bg-zinc-50"
                onClick={() => setOpenIdx(isOpen ? null : i)}
              >
                <span className="flex items-center gap-2">
                  {isOpen ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
                  <span className="font-mono font-medium text-zinc-800">{s.name || `spec-${i + 1}`}</span>
                  <Badge variant={s.goal === "exceed" ? "indigo" : s.goal === "minimize" ? "warning" : "neutral"}>
                    {s.goal}
                  </Badge>
                  <span className="text-zinc-500">target: {s.target || "—"}</span>
                  <span className="text-zinc-400">@ {s.testbench || "—"}</span>
                  {!s.enable && <Badge variant="neutral">disabled</Badge>}
                </span>
                <span
                  className="rounded-md border border-zinc-200 px-2 py-1 text-zinc-400 hover:bg-red-50 hover:text-red-600"
                  role="button"
                  aria-label="Remove spec"
                  onClick={(e) => { e.stopPropagation(); remove(i); }}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </span>
              </button>

              {isOpen && (
                <div className="grid grid-cols-3 gap-3 border-t border-zinc-100 p-3">
                  <Field label="Name"><TextInput value={s.name} onChange={(e) => update(i, { name: e.target.value })} placeholder="ugf" /></Field>
                  <Field label="Testbench">
                    <select className={selectCn("sm")} value={s.testbench} onChange={(e) => update(i, { testbench: e.target.value })}>
                      <option value="">— select —</option>
                      {tbNames.map((n) => <option key={n} value={n}>{n}</option>)}
                    </select>
                  </Field>
                  <Field label="Sim type">
                    <select className={selectCn("sm")} value={s.sim_type} onChange={(e) => update(i, { sim_type: e.target.value as WizardTargetSpec["sim_type"] })}>
                      {SIM_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
                    </select>
                  </Field>
                  <Field label="Goal">
                    <select className={selectCn("sm")} value={s.goal} onChange={(e) => update(i, { goal: e.target.value as WizardTargetSpec["goal"] })}>
                      {GOALS.map((g) => <option key={g} value={g}>{g}</option>)}
                    </select>
                  </Field>
                  <Field label="Target"><TextInput value={s.target} onChange={(e) => update(i, { target: e.target.value })} placeholder="200e6" /></Field>
                  <Field label="Range"><TextInput value={s.range} onChange={(e) => update(i, { range: e.target.value })} placeholder="100e6" /></Field>
                  <Field label="Tolerance"><TextInput value={s.tolerance} onChange={(e) => update(i, { tolerance: e.target.value })} placeholder="10e6" /></Field>
                  <Field label="Weight"><TextInput value={s.weight} onChange={(e) => update(i, { weight: e.target.value })} placeholder="1.0" /></Field>
                  <Field label="Error type">
                    <select className={selectCn("sm")} value={s.error_type} onChange={(e) => update(i, { error_type: e.target.value })}>
                      {ERROR_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
                    </select>
                  </Field>
                  <Field label="Reward type">
                    <select className={selectCn("sm")} value={s.reward_type} onChange={(e) => update(i, { reward_type: e.target.value })}>
                      {REWARD_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
                    </select>
                  </Field>
                  <Field label="Description" className="col-span-3">
                    <TextInput value={s.description} onChange={(e) => update(i, { description: e.target.value })} placeholder="Unity gain frequency" />
                  </Field>
                  <div className="col-span-3 flex items-center gap-4">
                    <label className="flex items-center gap-2 text-xs text-zinc-700">
                      <input type="checkbox" checked={s.log_scale} onChange={(e) => update(i, { log_scale: e.target.checked })} /> log_scale
                    </label>
                    <label className="flex items-center gap-2 text-xs text-zinc-700">
                      <input type="checkbox" checked={s.enable} onChange={(e) => update(i, { enable: e.target.checked })} /> enable
                    </label>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
