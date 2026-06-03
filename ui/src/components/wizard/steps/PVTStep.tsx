"use client";
import { Plus, Trash2 } from "lucide-react";
import { useWizardStore } from "@/stores/wizardStore";
import { Button } from "@/components/ui/button";
import { TextInput, StepHeader } from "../wizard-controls";

export function PVTStep() {
  const { form, setPvt } = useWizardStore();
  const rows = form.pvt_corners;

  const addRow = () => setPvt([...rows, { name: "", temp: 25, corner: "tt", supply: 1.8 }]);
  const removeRow = (i: number) => setPvt(rows.filter((_, idx) => idx !== i));
  const updateRow = (i: number, patch: Partial<typeof rows[number]>) =>
    setPvt(rows.map((r, idx) => (idx === i ? { ...r, ...patch } : r)));

  return (
    <div>
      <StepHeader title="PVT Corners" description="Process / temperature / supply corners the optimizer sweeps over." />
      <div className="space-y-3 p-4">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-zinc-600">Corners ({rows.length})</span>
          <Button variant="secondary" onClick={addRow} className="!h-7 !px-2 !text-xs">
            <Plus className="h-3 w-3" /> Add corner
          </Button>
        </div>

        <div className="grid grid-cols-[1.4fr_0.8fr_0.8fr_0.8fr_auto] gap-2 text-[10px] font-medium uppercase tracking-wide text-zinc-400">
          <div>Name</div><div>Temp (°C)</div><div>Corner</div><div>Supply (V)</div><div></div>
        </div>

        {rows.map((r, i) => (
          <div key={i} className="grid grid-cols-[1.4fr_0.8fr_0.8fr_0.8fr_auto] gap-2">
            <TextInput value={r.name} placeholder="typical_25C" onChange={(e) => updateRow(i, { name: e.target.value })} />
            <TextInput type="number" value={String(r.temp)} onChange={(e) => updateRow(i, { temp: e.target.value })} />
            <TextInput value={r.corner} placeholder="tt / ss / ff" onChange={(e) => updateRow(i, { corner: e.target.value })} />
            <TextInput type="number" step="0.1" value={String(r.supply)} onChange={(e) => updateRow(i, { supply: e.target.value })} />
            <button
              type="button"
              className="rounded-md border border-zinc-200 px-2 text-zinc-400 hover:bg-red-50 hover:text-red-600"
              onClick={() => removeRow(i)}
              aria-label="Remove corner"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>
        ))}

        {rows.length === 0 && (
          <div className="rounded-md border border-dashed border-zinc-300 bg-zinc-50 p-4 text-center text-xs text-zinc-500">
            No corners yet — at least one is required.
          </div>
        )}
      </div>
    </div>
  );
}
