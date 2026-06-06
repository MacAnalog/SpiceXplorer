/**
 * O(n²) non-dominated (Pareto) front over 2-D points. `n` is small (typically
 * < 2000 evaluated points), so the quadratic sweep is fine. Direction per axis
 * follows the spec goal: `minimize` → smaller is better, anything else → larger
 * is better (matching the feasible-region logic in MetricScatterChart).
 */
export interface XYPoint {
  x: number;
  y: number;
}

export function paretoFront<T extends XYPoint>(
  pts: T[],
  goalX: string,
  goalY: string,
): T[] {
  const atLeastX = (a: number, b: number) => (goalX === "minimize" ? a <= b : a >= b);
  const atLeastY = (a: number, b: number) => (goalY === "minimize" ? a <= b : a >= b);
  const strictX = (a: number, b: number) => (goalX === "minimize" ? a < b : a > b);
  const strictY = (a: number, b: number) => (goalY === "minimize" ? a < b : a > b);
  const dominated = (p: T) =>
    pts.some(
      (q) =>
        q !== p &&
        atLeastX(q.x, p.x) &&
        atLeastY(q.y, p.y) &&
        (strictX(q.x, p.x) || strictY(q.y, p.y)),
    );
  const front = pts.filter((p) => !dominated(p));
  // Sort along x so a connecting line reads as a frontier.
  return front.sort((a, b) => a.x - b.x);
}
