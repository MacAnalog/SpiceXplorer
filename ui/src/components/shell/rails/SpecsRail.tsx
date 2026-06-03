"use client";
import type { Route } from "next";
import { useRouter } from "next/navigation";
import { useProjectStore } from "@/stores/projectStore";
import { useUIStore } from "@/stores/uiStore";
import { cn, formatEng } from "@/lib/utils";
import { RailHeading, RailHint } from "./parts";

/**
 * Spec-centric rail (Score Shaping, Pipeline): the project's target specs,
 * clickable to deep-link into Score Shaping (same selection the ⌘K palette
 * sets). The active spec is highlighted; disabled specs render dimmed.
 */
export function SpecsRail() {
  const router = useRouter();
  const { summary, isApplied } = useProjectStore();
  const selectedSpec = useUIStore((s) => s.selectedSpec);
  const setSelectedSpec = useUIStore((s) => s.setSelectedSpec);

  const specs = summary?.target_specs ?? [];
  if (!isApplied || specs.length === 0) {
    return (
      <>
        <RailHeading>Target specs</RailHeading>
        <RailHint>Apply a project on Setup to list its target specs.</RailHint>
      </>
    );
  }

  return (
    <>
      <RailHeading>Target specs</RailHeading>
      {specs.map((s) => {
        const active = selectedSpec === s.name;
        const sym = s.goal === "exceed" ? "≥" : s.goal === "minimize" ? "≤" : "≈";
        return (
          <button
            key={s.name}
            type="button"
            onClick={() => {
              setSelectedSpec(s.name);
              router.push("/scoring" as Route);
            }}
            title={`${s.name} — shape this spec's score`}
            className={cn(
              "flex w-full items-center justify-between gap-2 rounded px-1.5 py-1 text-left transition",
              active ? "bg-primary-soft text-primary" : "hover:bg-hairline",
              !s.enable && "opacity-50",
            )}
          >
            <span className="truncate text-[11px]">{s.name}</span>
            <span className="shrink-0 font-mono text-[10px] text-muted">
              {sym}
              {formatEng(s.target)}
            </span>
          </button>
        );
      })}
    </>
  );
}
