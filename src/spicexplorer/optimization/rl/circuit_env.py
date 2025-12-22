
import numpy as np
import os
import logging

from typing import Optional
from decimal import Decimal, getcontext, ROUND_HALF_UP
from pathlib import Path

import torch

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces.space import Space

from .utils.hyperparameters import EnvHyperparameters

from spicexplorer.core.domains import Project_Setup

# ------------------ Module Logger ------------------

logger = logging.getLogger("spicexplorer.optimization.rl.gymenv")

# ------------------ Classes ------------------
FLOAT_TYPE = np.float32

class SpiceGymEnv(gym.Env):
    """
    A Gym Adapter that connects an RL Agent to the SpiceXplorer Framework.
    It delegates the 'step' logic to the Optimizer's existing evaluate() method.
    """
    metadata = {'render_modes': ['human']}
    def __init__(self, eval_callback, setup_obj: Project_Setup, config: EnvHyperparameters, run_name: Optional[str] = "default", render_mode=None):
        super().__init__()
        self.setup_obj = setup_obj
        self.eval_callback = eval_callback  # Points to Optimizer.evaluate()
        
        self.run_name = run_name
        self.config = config
        self.render_mode = render_mode

        # 1. Define Action Space (Normalized [-1, 1])
        # The optimizer handles denormalization later
        n_actions = len(setup_obj.dut_params)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_actions,), dtype=np.float32)

        # 2. Define Observation Space (Normalized Specs)
        # We observe the current value of every target spec
        n_obs = len(setup_obj.optimizer_config.target_specs.targets)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

        self.current_episode_step = 0

    # ------------------
    # Gym Environment Abstract Methods
    # ------------------
    def step(self, action: np.ndarray):
        # A. Map Action -> Dictionary (The optimizer expects a dict)
        # Note: We assume the RL agent outputs [-1, 1]. Your denormalize_params handles the rest.
        param_dict = {}
        for i, param in enumerate(self.setup_obj.dut_params):
             # We pass the raw normalized value to the optimizer's helper
             # We assume your denormalize_params handles mapping -1..1 to min..max
             param_dict[param.name] = float(action[i]) 

        # B. Delegate Simulation to Framework
        # score is the fitness, summary contains raw values
        score, fit_summary = self.eval_callback(param_dict, is_normalized_input=True)

        # C. Construct Observation State
        # Extract the 'curr_val' from fit_summary for the agent to see
        obs_list = []
        for spec_name in self.setup_obj.optimizer_config.target_specs.list_target_names():
            val = fit_summary.get(spec_name, {}).get('curr_val', 0.0)
            obs_list.append(val if not np.isnan(val) else 0.0)
        
        observation = np.array(obs_list, dtype=np.float32)
        
        # D. Return standard Gym tuple
        # Terminated can be True if score > threshold (Goal met)
        terminated = False 
        truncated = False
        
        return observation, float(score), terminated, truncated, fit_summary
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Return a zero vector or run a random initial simulation
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}
    
    def render(self):
        if self.render_mode == "human":
            # Logic to print circuit performance or plot curves
            pass

    def close(self):
        # Cleanup temporary spice files or close simulator handles
        pass
    
    # ------------------
    # GenericHelper Methods
    # ------------------
    def _get_initial_observation(self) -> np.ndarray:
        """Return the initial observation of the environment."""
        return np.zeros(self.state_dim, dtype=FLOAT_TYPE)
    
    def _get_initial_action(self) -> np.ndarray:
        """Return the initial action (e.g., mid-point of action space)."""
        return np.zeros(self.action_dim, dtype=FLOAT_TYPE)

    def get_observation_keys(self) -> list:
        """Return the list of observation keys."""
        return []
    