import type { ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "danger";
};

export function Button({
  className,
  variant = "secondary",
  ...props
}: ButtonProps) {
  const variants = {
    primary: "bg-circuit text-white hover:bg-teal-800",
    secondary: "border border-zinc-300 bg-white text-zinc-900 hover:bg-zinc-100",
    ghost: "text-zinc-700 hover:bg-zinc-100",
    danger: "bg-red-700 text-white hover:bg-red-800"
  };

  return (
    <button
      className={cn(
        "inline-flex h-9 items-center justify-center gap-2 rounded-md px-3 text-sm font-medium transition disabled:pointer-events-none disabled:opacity-50",
        variants[variant],
        className
      )}
      {...props}
    />
  );
}
