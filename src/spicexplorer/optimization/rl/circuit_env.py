
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

# ------------------ Module Logger ------------------

logger = logging.getLogger("spicexplorer.optimization.rl.gymenv")

# ------------------ Classes ------------------
FLOAT_TYPE = np.float32

class SpiceCircuitEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, config: EnvHyperparameters, run_name: Optional[str] = "default", render_mode=None):
        super().__init__()
        self.config = config
        self.render_mode = render_mode

        # Define spaces
        self._create_action_space(param_space_path=config.param_space_path)
        self._create_observation_space(target_space_path=config.target_specs_path)

        self.current_episode_step = 0

    # ------------------
    # Gym Environment Abstract Methods
    # ------------------
    def step(self, action: np.ndarray):
        self.current_episode_step += 1

        # 1. Map normalized action [-1, 1] to physical circuit parameters
        # 2. Run simulation (Spectre/Ngspice)
        # 3. Calculate Reward
        
        observation = np.zeros(self.state_dim, dtype=FLOAT_TYPE)
        reward = 0.0
        
        # Determine termination
        # 'terminated' usually means the goal was reached or the circuit failed
        terminated = False 
        
        # 'truncated' usually means the time limit (max_steps) was hit
        truncated = self.current_episode_step >= self.config.max_episode_steps
        
        info = {"raw_metrics": {}}

        return observation, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # Initialize the RNG
        super().reset(seed=seed)
        
        self.current_episode_step = 0
        
        # Reset your spice simulation/state here
        observation = self._get_initial_observation()
        info = {} # Useful for debugging or passing metrics

        return observation, info
    
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
    def _create_action_space(self, param_space_path: str) -> Space: 
        """"""
        self.action_dim = ...
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=FLOAT_TYPE)
        return self.action_space
    
    def _create_observation_space(self, target_space_path: str) -> Space: 
        """"""
        self.state_dim = ...
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=FLOAT_TYPE)
        return self.observation_space
    
    def _get_initial_observation(self) -> np.ndarray:
        """Return the initial observation of the environment."""
        return np.zeros(self.state_dim, dtype=FLOAT_TYPE)
    
    def _get_initial_action(self) -> np.ndarray:
        """Return the initial action (e.g., mid-point of action space)."""
        return np.zeros(self.action_dim, dtype=FLOAT_TYPE)

    def get_observation_keys(self) -> list:
        """Return the list of observation keys."""
        return []
    
    