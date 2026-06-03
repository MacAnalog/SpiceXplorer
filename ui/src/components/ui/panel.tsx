import type { HTMLAttributes, ReactNode } from "react";
import { cn } from "@/lib/utils";

export function Panel({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <section
      className={cn(
        "overflow-hidden rounded-md border border-border bg-panel",
        className,
      )}
      {...props}
    />
  );
}

interface PanelHeaderProps extends Omit<HTMLAttributes<HTMLDivElement>, "title"> {
  title?: ReactNode;
  mute?: ReactNode;
  right?: ReactNode;
}

export function PanelHeader({
  className,
  title,
  mute,
  right,
  children,
  ...props
}: PanelHeaderProps) {
  if (title !== undefined || mute !== undefined || right !== undefined) {
    return (
      <div
        className={cn(
          "flex items-center justify-between border-b border-border px-3 py-[7px]",
          className,
        )}
        {...props}
      >
        <div className="text-[11px] font-medium leading-none text-fg">
          {title}
          {mute && (
            <span className="ml-1 font-normal text-muted">{mute}</span>
          )}
        </div>
        {right && <div className="flex items-center gap-1.5">{right}</div>}
      </div>
    );
  }
  return (
    <div
      className={cn(
        "flex items-center justify-between border-b border-border px-3 py-[7px] text-[11px] font-medium leading-none",
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export function PanelBody({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("px-3 py-2.5", className)} {...props} />;
}
