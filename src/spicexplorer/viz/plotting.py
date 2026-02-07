import logging
import json
import numpy        as np
import plotly.graph_objects as go

from    pathlib     import PosixPath, WindowsPath, Path
from    dacite      import from_dict, Config
from    typing      import Dict, List, Tuple, Any, Mapping



from spicexplorer.core.domains import OptimizationLog, OptimizationLogEntry, Project_Setup
from spicexplorer.optimization.base import CHECKPOINT_SCHEMA_VERSION, Spice_Constraint_Satisfaction


logger = logging.getLogger("spicexplorer.viz.plotting")

# ----------------------------
# --- Global Constants ---
# ----------------------------





# ----------------------------
# --- Class Definitions ---
# ----------------------------

class Optimization_Log_Visualizer:

    def __init__(self, optimization_log: OptimizationLog):
        self.optimization_log = optimization_log

    # ------------------------------------------------------------
    # Load/Save Method
    # ------------------------------------------------------------
    @classmethod
    def load_checkpoint(cls, path_to_checkpoint: str | Path, **kwargs) -> "Optimization_Log_Visualizer":
        """Load optimizer and project setup from JSON checkpoint with version validation."""
        path = Path(path_to_checkpoint)
        with open(path, "r") as f:
            data = json.load(f)

        # Validate schema version
        version = data.get("schema_version")
        if version != CHECKPOINT_SCHEMA_VERSION:
            logger.warning(f"⚠️ Checkpoint version mismatch: {version} != {CHECKPOINT_SCHEMA_VERSION}")


        raw_log_data = data.get("optimization_log", [])

        # PATCH: specific fix for stringified log_file dictionaries containing PosixPath
        for entry in raw_log_data:
            if "log_file" in entry and isinstance(entry["log_file"], str):
                try:
                    # CAUTION: eval() executes the string as code. 
                    # Only use this on trusted checkpoint files.
                    entry["log_file"] = eval(entry["log_file"])
                except Exception as e:
                    print(f"Warning: Could not parse log_file entry: {e}")

        # Rebuild optimization log
        optimization_log = OptimizationLog([
            from_dict(OptimizationLogEntry, entry, Config(strict=False))
            for entry in raw_log_data
        ])

        # Recreate optimizer instance
        obj = cls(optimization_log=optimization_log, **kwargs)

        logger.info(f"✅ Checkpoint loaded successfully from {path}")
        return obj
    
    # ------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------
    def is_empty(self) -> bool:
        if len(self.optimization_log) < 1:
            logger.warning("No optimization trace to plot")
            return False
        return True
    
    def has_param(self, param_name: str) -> bool:
        if not self.optimization_log.has_param(param_name=param_name):
            logger.warning(f"param '{param_name}' not found in optimization trace")
            return False
        return True

    def list_available_metrics(self) -> List[str]:
        return self.optimization_log.list_available_metrics()
    
    def list_available_params(self) -> List[str]:
        return self.optimization_log.list_available_params()

    # ------------------------------------------------------------
    # Re-computing Loss
    # ------------------------------------------------------------
    def recompute_loss_from_optimization_config(self, optimizer: Spice_Constraint_Satisfaction) -> None:
        for i, entry in enumerate(self.optimization_log):
            performance_array  = entry.get_performance_params()
            total_score, fit_summary = optimizer.compute_fitness(performance_array=performance_array)
            self.optimization_log.update_entry_fit_summary(index=i, fit_summary=fit_summary)

    # ------------------------------------------------------------
    # Plotting Methods
    # ------------------------------------------------------------
    def plot_design_space_exploration(
        self,
        param_x: str,
        param_y: str,
        save_path: Path | None = None,
        show: bool = False,
        log_x: bool = False,
        log_y: bool = False
    ) -> Tuple[np.ndarray, np.ndarray] | None:

        logger = logging.getLogger("SpiceXplorer.plotter")

        if not self.is_empty():
            return None

        if not self.has_param(param_x) or not self.has_param(param_y):
            return None

        x_values = np.array(
            [entry.get_param_val(param_x) for entry in self.optimization_log],
            dtype=float
        )
        y_values = np.array(
            [entry.get_param_val(param_y) for entry in self.optimization_log],
            dtype=float
        )
        loss = np.array(
            [entry.get_score() for entry in self.optimization_log],
            dtype=float
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers",
            marker=dict(
                size=10,
                color=loss,
                colorscale="Viridis",
                colorbar=dict(title="Score"),
                showscale=True
            ),
            name="Design Space Exploration"
        ))

        fig.update_layout(
            title=f"Design Space Exploration: {param_y} vs. {param_x}",
            xaxis_title=param_x,
            yaxis_title=param_y,
            template="plotly_dark",
            showlegend=False
        )

        fig.update_xaxes(type="log" if log_x else "linear")
        fig.update_yaxes(type="log" if log_y else "linear")

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(save_path))
            logger.info(f"📊 Plot saved to {save_path}")

        if show:
            fig.show()

        return x_values, y_values
    
    def plot_optimization_trace(
        self,
        metric_x: str,
        metric_y: str,
        save_path: Path | None = None,
        show: bool = False,
        log_x: bool = False,
        log_y: bool = False
    ) -> Tuple[np.ndarray, np.ndarray] | None:

        if not self.is_empty():
            return None

        fit_summary = self.optimization_log[0].get_fit_summary()

        if metric_x not in fit_summary:
            logger.warning(f"metric_x '{metric_x}' not found in optimization log")
            return None

        if metric_y not in fit_summary:
            logger.warning(f"metric_y '{metric_y}' not found in optimization log")
            return None

        x_values = np.array(
            [entry.get_fit_summary()[metric_x]["curr_val"] for entry in self.optimization_log],
            dtype=float
        )
        y_values = np.array(
            [entry.get_fit_summary()[metric_y]["curr_val"] for entry in self.optimization_log],
            dtype=float
        )
        fom = np.array(
            [entry.get_score() for entry in self.optimization_log],
            dtype=float
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers",
            marker=dict(
                size=10,
                color=fom,
                colorscale="Viridis",
                colorbar=dict(title="FOM"),
                showscale=True
            ),
            name="Optimization Trace"
        ))

        fig.update_layout(
            title=f"Optimization Trace: {metric_y} vs. {metric_x}",
            xaxis_title=metric_x,
            yaxis_title=metric_y,
            template="plotly_dark",
            showlegend=False
        )

        fig.update_xaxes(type="log" if log_x else "linear")
        fig.update_yaxes(type="log" if log_y else "linear")

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(save_path))
            logger.info(f"📊 Plot saved to {save_path}")

        if show:
            fig.show()

        return x_values, y_values
