"""Compute sigmoid vs. linear score penalties for a project's target specs."""
from __future__ import annotations

import numpy as np
from typing import Any

from spicexplorer.core.domains import Project_Setup, OptimizationGoalType
from spicexplorer.core.utils import compute_relative_absolute_error, compute_relative_sigmoid_error


def _raw_directional_error(value: float, target: float, tolerance: float, goal: OptimizationGoalType) -> float:
    """Returns the raw (non-normalized, directional) constraint violation. Zero when met."""
    tol = abs(tolerance) if tolerance else 0.0
    if goal == OptimizationGoalType.EXCEED:
        return max(0.0, (target - tol) - value)
    if goal == OptimizationGoalType.MINIMIZE:
        return max(0.0, value - (target + tol))
    # EXACT
    return max(0.0, abs(value - target) - tol)


def compute_score(
    project: Project_Setup,
    metric_values: dict[str, float],
    selected_spec: str | None = None,
    n_curve_points: int = 200,
) -> dict[str, Any]:
    """
    Compute per-spec linear and sigmoid penalties for the given metric values.

    Returns per_spec penalties, aggregate scores, and a curve for the selected_spec.
    """
    specs = project.optimizer_config.target_specs.enabled_targets()

    per_spec: dict[str, Any] = {}
    total_linear = 0.0
    total_sigmoid = 0.0

    for spec in specs:
        value = metric_values.get(spec.name)
        target = float(spec.target)
        tolerance = float(spec.tolerance) if spec.tolerance else abs(0.05 * target)
        weight = float(spec.weight) if spec.weight else 1.0
        rang = float(spec.range) if spec.range and spec.range > 0 else max(abs(target), 1.0)

        if value is None:
            per_spec[spec.name] = {
                "linear": None, "sigmoid": None,
                "value": None, "target": target, "goal": spec.goal.value,
                "passes": None, "weight": weight,
            }
            continue

        raw = _raw_directional_error(float(value), target, tolerance, spec.goal)
        passes = raw <= 0.0

        # Normalized penalties (always ≥ 0; zero when constraint is met)
        if raw <= 0.0:
            linear_p = 0.0
            sigmoid_p = 0.0
        else:
            linear_p = float(compute_relative_absolute_error(np.float64(raw), np.float64(0.0), np.float64(rang)))
            sigmoid_p = float(compute_relative_sigmoid_error(np.float64(raw), np.float64(0.0), np.float64(rang)))

        per_spec[spec.name] = {
            "linear": linear_p,
            "sigmoid": sigmoid_p,
            "value": float(value),
            "target": target,
            "tolerance": tolerance,
            "goal": spec.goal.value,
            "passes": passes,
            "weight": weight,
        }
        total_linear += weight * linear_p
        total_sigmoid += weight * sigmoid_p

    # Build penalty curve for selected spec (used by PenaltyCurveChart)
    curve: dict[str, Any] | None = None
    if selected_spec:
        spec_obj = next((s for s in specs if s.name == selected_spec), None)
        if spec_obj:
            target = float(spec_obj.target)
            tolerance = float(spec_obj.tolerance) if spec_obj.tolerance else abs(0.05 * target)
            rang = float(spec_obj.range) if spec_obj.range and spec_obj.range > 0 else max(abs(target), 1.0)
            lo = target - 3 * rang
            hi = target + 3 * rang
            xs = np.linspace(lo, hi, n_curve_points).tolist()
            linears, sigmoids = [], []
            for x in xs:
                raw = _raw_directional_error(x, target, tolerance, spec_obj.goal)
                if raw <= 0.0:
                    linears.append(0.0)
                    sigmoids.append(0.0)
                else:
                    linears.append(float(compute_relative_absolute_error(np.float64(raw), np.float64(0.0), np.float64(rang))))
                    sigmoids.append(float(compute_relative_sigmoid_error(np.float64(raw), np.float64(0.0), np.float64(rang))))
            curve = {"values": xs, "linear": linears, "sigmoid": sigmoids,
                     "target": target, "tolerance": tolerance, "goal": spec_obj.goal.value}

    return {
        "per_spec": per_spec,
        "aggregate": {"linear": -total_linear, "sigmoid": -total_sigmoid},
        "curve": curve,
    }
