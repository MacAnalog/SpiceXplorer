import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

const sizeStyles = {
  sm: "py-1.5 text-sm",
  xs: "py-1 text-xs",
} as const;

type SelectSize = keyof typeof sizeStyles;

export function selectCn(selectSize: SelectSize = "sm") {
  return cn(
    "rounded-md border border-zinc-300 bg-white px-2",
    sizeStyles[selectSize],
    "focus:outline-none focus:ring-1 focus:ring-indigo-500",
    "disabled:bg-zinc-50 disabled:text-zinc-400 disabled:opacity-50",
  );
}

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  selectSize?: SelectSize;
  options?: { value: string; label: string }[];
  children?: ReactNode;
  className?: string;
}

export function Select({
  label,
  selectSize = "sm",
  options,
  children,
  className,
  ...props
}: SelectProps) {
  return (
    <div className="flex flex-col gap-1">
      {label && (
        <label
          className="text-xs font-medium uppercase tracking-wide text-zinc-500"
          htmlFor={props.id}
        >
          {label}
        </label>
      )}
      <select
        className={cn("block w-full", selectCn(selectSize), className)}
        {...props}
      >
        {children ??
          options?.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
      </select>
    </div>
  );
}
