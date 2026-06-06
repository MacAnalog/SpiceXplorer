"""Read and normalize checkpoint data from both JSON (OptimizationLog) and CSV trace files."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from ui.backend.services.num import safe_float as _safe_float


# ---------- JSON checkpoint reader ----------

def read_json_checkpoint(path: Path, limit: int | None = None) -> dict[str, Any]:
    from spicexplorer.viz.plotting import Optimization_Log_Visualizer

    vis = Optimization_Log_Visualizer.load_checkpoint(path)
    log = vis.optimization_log

    scores, best_scores, iterations = [], [], []
    per_metric: dict[str, list[float | None]] = {}
    params_out: dict[str, list[float | None]] = {}
    best = -math.inf

    for i, entry in enumerate(log):
        if limit and i >= limit:
            break  # honor `limit` like the CSV reader's df.head(limit)
        s = _safe_float(entry.get_score())
        if s is not None and s > best:
            best = s
        scores.append(s)
        best_scores.append(best if math.isfinite(best) else None)
        iterations.append(i)

        fs = entry.fit_summary or {}
        for metric, vals in fs.items():
            # Some optimizers (e.g. the Bode path) store bare scalars in fit_summary
            # rather than {"curr_val": ...} dicts — skip those instead of crashing.
            if not isinstance(vals, dict):
                continue
            per_metric.setdefault(metric, []).append(_safe_float(vals.get("curr_val")))

        for pname, pval in entry.get_params().items():
            params_out.setdefault(pname, []).append(_safe_float(pval))

    return {
        "scores": scores,
        "best_scores": best_scores,
        "iterations": iterations,
        "per_metric": per_metric,
        "params": params_out,
        "n_iters": len(scores),
    }


# ---------- CSV trace reader ----------

METRIC_PREFIX = "fit_summary."
METRIC_VALUE_SUFFIX = ".curr_val"
PARAM_PREFIX = "point.params."


def read_csv_checkpoint(path: Path, limit: int | None = None) -> dict[str, Any]:
    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)

    scores, best_scores, iterations = [], [], []
    per_metric: dict[str, list[float | None]] = {}
    params_out: dict[str, list[float | None]] = {}
    best = -math.inf

    metric_cols = [c for c in df.columns if c.startswith(METRIC_PREFIX) and c.endswith(METRIC_VALUE_SUFFIX)]
    param_cols = [c for c in df.columns if c.startswith(PARAM_PREFIX)]
    metric_names = [c[len(METRIC_PREFIX):-len(METRIC_VALUE_SUFFIX)] for c in metric_cols]
    param_names = [c[len(PARAM_PREFIX):] for c in param_cols]

    for i, row in df.iterrows():
        s = _safe_float(row.get("point.score"))
        if s is not None and s > best:
            best = s
        scores.append(s)
        best_scores.append(best if math.isfinite(best) else None)
        iterations.append(i)

        for mn, mc in zip(metric_names, metric_cols):
            per_metric.setdefault(mn, []).append(_safe_float(row.get(mc)))
        for pn, pc in zip(param_names, param_cols):
            params_out.setdefault(pn, []).append(_safe_float(row.get(pc)))

    return {
        "scores": scores,
        "best_scores": best_scores,
        "iterations": iterations,
        "per_metric": per_metric,
        "params": params_out,
        "n_iters": len(scores),
    }


# ---------- unified loader ----------

def read_checkpoint(path: Path, limit: int | None = None) -> dict[str, Any]:
    if path.suffix == ".json":
        return read_json_checkpoint(path, limit=limit)
    return read_csv_checkpoint(path, limit=limit)


# ---------- envelope ----------

def compute_envelope(
    data: dict[str, Any],
    target_specs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Per-metric best-ever value across all sampled designs, independent of other specs."""
    results = []
    per_metric = data.get("per_metric", {})

    spec_map: dict[str, dict] = {}
    if target_specs:
        for s in target_specs:
            spec_map[s["name"]] = s

    for metric, values in per_metric.items():
        clean = [v for v in values if v is not None]
        if not clean:
            continue
        spec = spec_map.get(metric, {})
        goal = spec.get("goal", "exceed")
        target = spec.get("target")

        if goal == "minimize":
            best_ever = min(clean)
        elif goal == "exact" and target is not None:
            # Closest sample to the target — NOT the max. An exact-goal spec (e.g.
            # phase margin 60°±10°) would otherwise report a 120° outlier as "best".
            best_ever = min(clean, key=lambda v: abs(v - target))
        else:
            best_ever = max(clean)

        passes: bool | None = None
        if target is not None:
            tol = spec.get("tolerance", abs(0.05 * target))
            if goal == "exceed":
                passes = best_ever >= target - tol
            elif goal == "minimize":
                passes = best_ever <= target + tol
            else:  # exact
                passes = abs(best_ever - target) <= tol

        results.append({
            "metric": metric,
            "best_ever": best_ever,
            "target": target,
            "goal": goal,
            "passes": passes,
        })
    return results


# ---------- scatter ----------

def compute_scatter(
    data: dict[str, Any],
    metric_x: str,
    metric_y: str,
    target_specs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return all design points in 2D metric space with feasibility tags."""
    per_metric = data.get("per_metric", {})
    scores = data.get("scores", [])
    xs = per_metric.get(metric_x, [])
    ys = per_metric.get(metric_y, [])

    # Build spec map for feasibility check (against ALL specs that have target info)
    spec_map: dict[str, dict] = {}
    if target_specs:
        for s in target_specs:
            spec_map[s["name"]] = s

    def _is_feasible(i: int) -> bool:
        for mn, vals in per_metric.items():
            if mn not in spec_map:
                continue
            spec = spec_map[mn]
            target = spec.get("target")
            if target is None:
                continue
            v = vals[i] if i < len(vals) else None
            if v is None:
                return False
            goal = spec.get("goal", "exceed")
            tol = spec.get("tolerance", abs(0.05 * target))
            if goal == "exceed" and v < target - tol:
                return False
            if goal == "minimize" and v > target + tol:
                return False
            if goal == "exact" and abs(v - target) > tol:
                return False
        return True

    points = []
    n = min(len(xs), len(ys))
    for i in range(n):
        xv, yv = xs[i], ys[i]
        if xv is None or yv is None:
            continue
        points.append({
            "x": xv,
            "y": yv,
            "feasible": _is_feasible(i),
            "score": scores[i] if i < len(scores) else None,
            "iter": i,
        })
    return points
