"""Compute sigmoid vs. linear score penalties for a project's target specs."""
from __future__ import annotations

import logging
import numpy as np
from typing import Any

from spicexplorer.core.domains import Project_Setup, OptimizationGoalType, parse_value
from spicexplorer.core.utils import compute_relative_absolute_error, compute_relative_sigmoid_error

logger = logging.getLogger(__name__)


def apply_spec_overrides(
    project: Project_Setup,
    overrides: dict[str, dict] | None,
) -> None:
    """Apply ephemeral, request-scoped target-spec edits to the loaded project.

    Score Shaping lets the user tune spec fields (target/tolerance/weight/range/
    goal/enable) for a *what-if* preview. Because /api/score reloads the project
    from disk every call (stateless), these edits travel in the request payload and
    are applied here to the freshly-loaded, in-memory project before scoring. This
    **never** rewrites the YAML — the edits vanish with the request, exactly the
    ephemeral semantics the UI promises.

    `overrides` maps spec name → a partial dict of fields to set. Numeric fields
    accept engineering strings ("250u") via parse_value. Unknown spec names and
    unknown goal values are ignored (rather than 500-ing a preview).
    """
    if not overrides:
        return
    by_name = {s.name: s for s in project.optimizer_config.target_specs.targets}
    for name, patch in overrides.items():
        spec = by_name.get(name)
        if spec is None or not isinstance(patch, dict):
            continue
        for field in ("target", "tolerance", "weight", "range"):
            if field in patch and patch[field] is not None and patch[field] != "":
                try:
                    setattr(spec, field, float(parse_value(patch[field])))
                except (ValueError, TypeError):
                    logger.warning("score override: bad %s for spec '%s': %r",
                                   field, name, patch[field])
        if "enable" in patch and patch["enable"] is not None:
            spec.enable = bool(patch["enable"])
        if patch.get("goal"):
            try:
                spec.goal = OptimizationGoalType(str(patch["goal"]).lower())
            except ValueError:
                logger.warning("score override: unknown goal for spec '%s': %r",
                               name, patch["goal"])


def _normalized_penalties(raw: float, rang: float) -> tuple[float, float]:
    """raw violation → (linear, sigmoid) normalized penalties; (0, 0) when met.

    Single source for the per-spec loop and the penalty-curve loop (previously
    copy-pasted verbatim).
    """
    if raw <= 0.0:
        return 0.0, 0.0
    r, zero, rng = np.float64(raw), np.float64(0.0), np.float64(rang)
    return (
        float(compute_relative_absolute_error(r, zero, rng)),
        float(compute_relative_sigmoid_error(r, zero, rng)),
    )


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
        weight = float(spec.weight) if spec.weight is not None else 1.0
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
        linear_p, sigmoid_p = _normalized_penalties(raw, rang)

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
                lin_p, sig_p = _normalized_penalties(raw, rang)
                linears.append(lin_p)
                sigmoids.append(sig_p)
            curve = {"values": xs, "linear": linears, "sigmoid": sigmoids,
                     "target": target, "tolerance": tolerance, "goal": spec_obj.goal.value}

    return {
        "per_spec": per_spec,
        # F(x) = Σ wᵢ · P̂ᵢ — the non-negative weighted penalty sum the UI header and the
        # per-spec columns show. Do NOT negate here: the optimizer's maximize-score
        # convention lives in the library scorer, not this preview service; negating made
        # the footer print a negative number under its own "sum of penalties" label.
        "aggregate": {"linear": total_linear, "sigmoid": total_sigmoid},
        "curve": curve,
    }
