export function cn(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}

const ENG_PREFIXES: [number, string][] = [
  [1e9, "G"], [1e6, "M"], [1e3, "k"],
  [1, ""], [1e-3, "m"], [1e-6, "µ"], [1e-9, "n"], [1e-12, "p"],
];

export function formatEng(value: number | null | undefined, unit = "", digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "n/a";
  const abs = Math.abs(value);
  for (const [scale, prefix] of ENG_PREFIXES) {
    if (abs >= scale * 0.9999 || scale === 1e-12) {
      return `${(value / scale).toFixed(digits)} ${prefix}${unit}`.trim();
    }
  }
  return value.toExponential(2);
}

/**
 * Canonical goal → comparison glyph. One source of truth so every surface
 * agrees (the codebase previously had two divergent glyph sets — `>`/`<` and
 * `≥`/`≤` — scattered across ~8 components). `≥`/`≤` are correct because the
 * pass band includes the tolerance boundary (see statusForGoal).
 */
export function goalSymbol(goal: string): string {
  return goal === "exceed" ? "≥" : goal === "minimize" ? "≤" : "≈";
}

/** Format a millisecond duration as `m:ss` (or `h:mm:ss` past an hour). */
export function formatDuration(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms) || ms < 0) return "—";
  const total = Math.floor(ms / 1000);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

export function statusForGoal(
  goal: string,
  value: number | null | undefined,
  target: number | string,
  tolerance?: number | string
): "pass" | "fail" | "unknown" {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "unknown";
  }
  const targetValue = Number(target);
  const tolValue = tolerance === undefined ? Math.abs(targetValue * 0.05) : Number(tolerance);
  if (!Number.isFinite(targetValue) || !Number.isFinite(tolValue)) {
    return "unknown";
  }
  if (goal === "exceed") {
    return value >= targetValue - tolValue ? "pass" : "fail";
  }
  if (goal === "minimize") {
    return value <= targetValue + tolValue ? "pass" : "fail";
  }
  if (goal === "exact") {
    return Math.abs(value - targetValue) <= tolValue ? "pass" : "fail";
  }
  return "unknown";
}
