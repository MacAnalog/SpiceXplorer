import type { ReactNode, TdHTMLAttributes, ThHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export function Thead({ children }: { children: ReactNode }) {
  return (
    <thead>
      <tr className="border-b border-zinc-100 bg-zinc-50 text-left text-zinc-500">
        {children}
      </tr>
    </thead>
  );
}

export function Th({ children, className, ...props }: ThHTMLAttributes<HTMLTableCellElement>) {
  return (
    <th className={cn("px-3 py-2 font-medium", className)} {...props}>
      {children}
    </th>
  );
}

export function Tr({
  children,
  highlight = false,
  className,
  ...props
}: { children: ReactNode; highlight?: boolean; className?: string } & React.HTMLAttributes<HTMLTableRowElement>) {
  return (
    <tr
      className={cn(
        "border-b border-zinc-50 last:border-0 hover:bg-zinc-50",
        highlight && "bg-indigo-50",
        className,
      )}
      {...props}
    >
      {children}
    </tr>
  );
}

export function Td({ children, className, ...props }: TdHTMLAttributes<HTMLTableCellElement>) {
  return (
    <td className={cn("px-3 py-1.5", className)} {...props}>
      {children}
    </td>
  );
}
