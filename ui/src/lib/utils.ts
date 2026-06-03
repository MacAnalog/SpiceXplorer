export function cn(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}

export function formatNumber(value: number | string | null | undefined, digits = 3): string {
  if (value === null || value === undefined || value === "") {
    return "n/a";
  }
  const numeric = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(numeric)) {
    return String(value);
  }
  const abs = Math.abs(numeric);
  if ((abs > 0 && abs < 0.001) || abs >= 100000) {
    return numeric.toExponential(2);
  }
  return numeric.toLocaleString(undefined, {
    maximumFractionDigits: digits
  });
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
