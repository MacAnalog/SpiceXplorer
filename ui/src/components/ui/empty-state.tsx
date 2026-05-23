import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface EmptyStateProps {
  children: ReactNode;
  minHeight?: string;
  bordered?: boolean;
  className?: string;
}

export function EmptyState({
  children,
  minHeight = "min-h-40",
  bordered = false,
  className,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        "flex items-center justify-center text-sm text-zinc-400",
        minHeight,
        bordered && "rounded-lg border border-dashed border-zinc-300 bg-white",
        className,
      )}
    >
      {children}
    </div>
  );
}
