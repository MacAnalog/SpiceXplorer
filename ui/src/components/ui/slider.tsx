import { cn } from "@/lib/utils";

interface SliderProps {
  label?: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  displayValue?: string;
  className?: string;
}

export function Slider({ label, value, min, max, step = 0.01, onChange, displayValue, className }: SliderProps) {
  return (
    <div className={cn("flex flex-col gap-1", className)}>
      {label && (
        <div className="flex items-center justify-between">
          <label className="text-xs font-medium text-zinc-500">{label}</label>
          <span className="text-xs font-mono text-zinc-700">{displayValue ?? value}</span>
        </div>
      )}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className={cn(
          "h-1.5 w-full cursor-pointer appearance-none rounded-full bg-zinc-200",
          "accent-indigo-600",
        )}
      />
      <div className="flex justify-between text-[10px] text-zinc-400">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}
