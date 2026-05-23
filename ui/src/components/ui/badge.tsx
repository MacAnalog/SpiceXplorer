import { cn } from "@/lib/utils";

type BadgeVariant = "pass" | "fail" | "neutral" | "warning" | "indigo";

const variants: Record<BadgeVariant, string> = {
  pass: "bg-emerald-100 text-emerald-800 border-emerald-200",
  fail: "bg-red-100 text-red-700 border-red-200",
  neutral: "bg-zinc-100 text-zinc-600 border-zinc-200",
  warning: "bg-amber-100 text-amber-800 border-amber-200",
  indigo: "bg-indigo-100 text-indigo-800 border-indigo-200",
};

interface BadgeProps {
  children: React.ReactNode;
  variant?: BadgeVariant;
  className?: string;
}

export function Badge({ children, variant = "neutral", className }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium",
        variants[variant],
        className,
      )}
    >
      {children}
    </span>
  );
}
