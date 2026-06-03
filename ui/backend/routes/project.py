"""Project loading and YAML validation routes."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from spicexplorer.core.domains import Project_Setup
from ui.backend.services.yaml_generator import generate_yaml, project_dict_to_form

router = APIRouter()


class LoadRequest(BaseModel):
    yaml_path: str


class ValidateRequest(BaseModel):
    yaml_content: str


class GenerateRequest(BaseModel):
    form: Dict[str, Any]
    save_path: Optional[str] = None  # absolute or workspace-relative path; if set, write to disk


class ParseToFormRequest(BaseModel):
    yaml_path: Optional[str] = None
    yaml_content: Optional[str] = None


def _summarise(project: Project_Setup) -> dict[str, Any]:
    specs = []
    for s in project.optimizer_config.target_specs.targets:
        specs.append({
            "name": s.name,
            "testbench": s.testbench,
            "goal": s.goal.value,
            "target": float(s.target),
            "tolerance": float(s.tolerance) if s.tolerance else None,
            "range": float(s.range) if s.range else None,
            "weight": float(s.weight) if s.weight else 1.0,
            "error_type": s.error_type.value if hasattr(s.error_type, "value") else str(s.error_type),
            "reward_type": s.reward_type.value if hasattr(s.reward_type, "value") else str(s.reward_type),
            "enable": s.enable,
            "description": s.description,
        })

    dut_params = []
    for p in project.dut_params:
        dut_params.append({
            "name": p.name,
            "min_val": float(p.min_val) if p.min_val is not None else None,
            "max_val": float(p.max_val) if p.max_val is not None else None,
            "is_integer": p.is_integer,
            "log_scale": p.log_scale,
            "freeze": p.freeze,
        })

    testbenches = []
    for tb in project.testbenches:
        testbenches.append({
            "name": tb.name,
            "netlist": str(tb.netlist),
            "enable": tb.enable,
            "description": tb.description,
            "params": [
                {"name": p.name, "val": str(p.val) if p.val is not None else None, "description": p.description}
                for p in tb.params
            ],
        })

    pvt = []
    for corner in project.pvt_corners:
        pvt.append({"temp": corner.temp, "corner": corner.corner, "supply": corner.supply})

    return {
        "name": project.name,
        "description": project.description,
        "simulator": project.simulator,
        "ws_root": str(project.ws_root),
        "netlist": str(project.netlist),
        "schematic": str(project.schematic) if project.schematic is not None else None,
        "tech": {
            "name": project.tech_spec.name,
            "constraints": {k: float(v) for k, v in project.tech_spec.constraints.items()},
        },
        "pvt_corners": pvt,
        "dut_params": dut_params,
        "testbenches": testbenches,
        "optimizer": {
            "type": project.optimizer_config.type,
            "name": project.optimizer_config.name,
            "budget": project.optimizer_config.budget,
            "random_seed": project.optimizer_config.random_seed,
        },
        "target_specs": specs,
    }


@router.post("/project/load")
def load_project(body: LoadRequest):
    path = Path(body.yaml_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"YAML file not found: {path}")
    try:
        project = Project_Setup.from_yaml(path)
        return {"ok": True, "summary": _summarise(project), "yaml_path": str(path)}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/project/validate")
def validate_yaml(body: ValidateRequest):
    errors: list[str] = []
    try:
        data = yaml.safe_load(body.yaml_content)
        if not isinstance(data, dict) or "project" not in data:
            errors.append("Top-level 'project:' key is required")
            return {"ok": False, "errors": errors}
    except yaml.YAMLError as e:
        return {"ok": False, "errors": [f"YAML parse error: {e}"]}

    # Write to a temp file and try full parse
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp.write(body.yaml_content)
        tmp_path = tmp.name
    try:
        Project_Setup.from_yaml(tmp_path)
        return {"ok": True, "errors": []}
    except Exception as e:
        return {"ok": False, "errors": [str(e)]}
    finally:
        os.unlink(tmp_path)


@router.post("/project/generate")
def generate_project(body: GenerateRequest):
    """Render wizard form → YAML string. If `save_path` is given, also write to disk."""
    try:
        yaml_text = generate_yaml(body.form)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"YAML generation failed: {e}")

    saved_path: Optional[str] = None
    if body.save_path:
        path = Path(body.save_path).expanduser()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml_text)
            saved_path = str(path)
        except OSError as e:
            raise HTTPException(status_code=400, detail=f"Could not write YAML to {path}: {e}")

    # Best-effort validation; surface errors but still return the rendered YAML
    errors: list[str] = []
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp.write(yaml_text)
        tmp_path = tmp.name
    try:
        Project_Setup.from_yaml(tmp_path)
    except Exception as e:
        errors.append(str(e))
    finally:
        os.unlink(tmp_path)

    return {
        "ok": not errors,
        "yaml": yaml_text,
        "errors": errors,
        "saved_path": saved_path,
    }


@router.post("/project/parse-to-form")
def parse_project_to_form(body: ParseToFormRequest):
    """Inverse of /project/generate — load a YAML and return a wizard-form dict."""
    if not body.yaml_path and not body.yaml_content:
        raise HTTPException(status_code=400, detail="Provide yaml_path or yaml_content")
    try:
        if body.yaml_content:
            data = yaml.safe_load(body.yaml_content)
        else:
            path = Path(body.yaml_path)  # type: ignore[arg-type]
            if not path.exists():
                raise HTTPException(status_code=404, detail=f"YAML file not found: {path}")
            with path.open() as f:
                data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=422, detail=f"YAML parse error: {e}")

    if not isinstance(data, dict) or "project" not in data:
        raise HTTPException(status_code=422, detail="Top-level 'project:' key is required")

    form = project_dict_to_form(data)
    return {"ok": True, "form": form}
