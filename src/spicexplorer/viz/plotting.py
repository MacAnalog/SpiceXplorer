import logging
import json
import numpy                as np
import plotly.graph_objects as go
import pandas               as pd

from    pathlib     import PosixPath, WindowsPath, Path
from    dacite      import from_dict, Config
from    typing      import Dict, List, Tuple, Any, Mapping
from    dataclasses import asdict

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
            return True
        return False
    
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

    def plot_loss_breakdown(
        self,
        save_path: Path | None = None,
        show: bool = False,
        log_y: bool = False
    ) -> go.Figure | None:
        """
        Plots the Total Score, Best Score So Far, and individual metric contributions.
        """
        logger = logging.getLogger("SpiceXplorer.plotter")

        if self.is_empty():
            return None

        # --- 1. Extract Data ---
        iterations = np.arange(len(self.optimization_log))
        
        # Current Score per iteration
        total_scores = np.array(
            [entry.get_score() for entry in self.optimization_log],
            dtype=float
        )

        # Best Score So Far (Cumulative Maximum)
        # We use maximum because you specified "most positive" is best.
        best_so_far = np.maximum.accumulate(total_scores)

        # Get metric keys for breakdown
        first_summary = self.optimization_log[0].get_fit_summary()
        metric_keys = list(first_summary.keys())

        # --- 2. Create Plotly Figure ---
        fig = go.Figure()

        # TRACE: Best Score So Far (Green Dashed Line)
        fig.add_trace(go.Scatter(
            x=iterations,
            y=best_so_far,
            mode='lines',
            name='<b>BEST SO FAR</b>',
            line=dict(width=3, color='#00FF00', dash='dash'), # Green, Dashed
            hovertemplate="Iter: %{x}<br>Best: %{y:.4f}<extra></extra>"
        ))

        # TRACE: Current Total Score (Solid White Line)
        fig.add_trace(go.Scatter(
            x=iterations,
            y=total_scores,
            mode='lines+markers',
            name='Current Score',
            line=dict(width=2, color='white'),
            marker=dict(size=5, opacity=0.8),
            hovertemplate="Iter: %{x}<br>Curr: %{y:.4f}<extra></extra>"
        ))

        # TRACES: Individual Metrics (Thinner, colored lines)
        for metric in metric_keys:
            metric_scores = []
            for entry in self.optimization_log:
                summary = entry.get_fit_summary()
                # Use .get() twice to be safe against missing keys
                val = summary.get(metric, {}).get("score", np.nan)
                metric_scores.append(val)
            
            fig.add_trace(go.Scatter(
                x=iterations,
                y=metric_scores,
                mode='lines',
                name=f"{metric}",
                opacity=0.5, # Transparent to keep main lines dominant
                line=dict(width=1),
                hovertemplate=f"Iter: %{{x}}<br>{metric}: %{{y:.4f}}<extra></extra>"
            ))

        # --- 3. Styling ---
        fig.update_layout(
            title="<b>Optimization Progress</b>: Best vs. Current vs. Components",
            xaxis_title="Iteration",
            yaxis_title="Score (Higher is Better)",
            template="plotly_dark",
            hovermode="x unified", # Shows all traces for an iteration on hover
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)"
            )
        )

        if log_y:
            fig.update_yaxes(type="log")

        # --- 4. Save/Show ---
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(save_path))
            logger.info(f"📊 Loss breakdown plot saved to {save_path}")

        if show:
            fig.show()

        return fig

    def plot_best_score_evolution(
        self,
        save_path: Path | None = None,
        show: bool = False,
        log_y: bool = False
    ) -> go.Figure | None:
        """
        Plots the 'Best Score So Far' (Cumulative Maximum) for the Total Score 
        and for each individual metric independently.
        """
        logger = logging.getLogger("SpiceXplorer.plotter")

        if self.is_empty():
            return None

        # --- 1. Extract and Process Data ---
        iterations = np.arange(len(self.optimization_log))
        
        # A. Total Score Best So Far
        total_scores = np.array(
            [entry.get_score() for entry in self.optimization_log],
            dtype=float
        )
        # Calculate cumulative maximum (best seen up to index i)
        total_best_so_far = np.maximum.accumulate(total_scores)

        # B. Get Metrics
        first_summary = self.optimization_log[0].get_fit_summary()
        metric_keys = list(first_summary.keys())

        # --- 2. Create Plotly Figure ---
        fig = go.Figure()

        # TRACE: Total Score Best (Thick, White)
        fig.add_trace(go.Scatter(
            x=iterations,
            y=total_best_so_far,
            mode='lines',
            name='<b>TOTAL BEST</b>',
            line=dict(width=4, color='white'), 
            hovertemplate="Iter: %{x}<br>Total Best: %{y:.4f}<extra></extra>"
        ))

        # TRACES: Individual Metrics Best So Far
        for metric in metric_keys:
            metric_scores = []
            for entry in self.optimization_log:
                summary = entry.get_fit_summary()
                # Default to -inf if missing so it doesn't affect max calculation
                val = summary.get(metric, {}).get("score", -np.inf)
                metric_scores.append(val)
            
            metric_scores = np.array(metric_scores, dtype=float)
            
            # Calculate cumulative max for this specific metric
            # We use fmax to ignore NaNs if they sneak in, though we defaulted to -inf
            metric_best_so_far = np.maximum.accumulate(metric_scores)

            fig.add_trace(go.Scatter(
                x=iterations,
                y=metric_best_so_far,
                mode='lines',
                name=f"{metric} (Best)",
                opacity=0.8,
                line=dict(width=2), # Slightly thinner than total
                hovertemplate=f"Iter: %{{x}}<br>{metric} Best: %{{y:.4f}}<extra></extra>"
            ))

        # --- 3. Styling ---
        fig.update_layout(
            title="<b>Evolution of Best Scores</b>: Independent Cumulative Maximums",
            xaxis_title="Iteration",
            yaxis_title="Best Score Achieved (Higher is Better)",
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)"
            )
        )

        if log_y:
            fig.update_yaxes(type="log")

        # --- 4. Save/Show ---
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(save_path))
            logger.info(f"📊 Best score evolution plot saved to {save_path}")

        if show:
            fig.show()

        return fig
    # ------------------------------------------------------------
    # Exporting Methods
    # ------------------------------------------------------------
    def to_df(self) -> pd.DataFrame:
        if self.is_empty():
            return pd.DataFrame()

        # Convert entries to dicts and remove 'log_file' before flattening
        data = []
        for entry in self.optimization_log:
            entry_dict = asdict(entry)
            entry_dict.pop("log_file", None) # Remove log_file key if it exists
            data.append(entry_dict)

        # Flatten nested dictionaries into dot-notation columns
        return pd.json_normalize(data)
    
    def to_csv(self, path: Path, index: bool = False) -> None:
        self.to_df().to_csv(path, index=index)