import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .hyperparameters import BaseHyperparameters

logger = logging.getLogger("SpiceXplorer")


class BaseActor(torch.nn.Module, ABC):
    """Abstract base class for actor networks."""

    @abstractmethod
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seed: int,
        hyperparams: BaseHyperparameters,
    ) -> None:
        """Initialize the actor network.
        Args:
            state_dim: Dimension of the state space.
            action_dim: Dimension of the action space.
            seed: Random seed.
            hyperparams: Dictionary of hyperparameters.
        """
        super().__init__()

    @abstractmethod
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Define the forward pass of the actor network.
        Args:
            state: Input tensor of states.
        Returns:
            Output tensor of actions.
        """
        raise NotImplementedError


class BaseCritic(torch.nn.Module, ABC):
    """Abstract base class for critic networks."""

    @abstractmethod
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seed: int,
        hyperparams: BaseHyperparameters,
    ) -> None:
        """Initialize the critic network.
        Args:
            state_dim: Dimension of the state space.
            action_dim: Dimension of the action space.
            seed: Random seed.
            hyperparams: Dictionary of hyperparameters.
        """
        super().__init__()

    @abstractmethod
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Define the forward pass of the critic network.
        Args:
            state: Input tensor of states.
            action: Input tensor of actions.
        Returns:
            Output tensor of Q-values.
        """
        raise NotImplementedError


class BaseRLAgent(ABC):
    """Abstract base class for reinforcement learning agents."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hyperparams: BaseHyperparameters,
        device: torch.device,
        seed: int,
    ) -> None:
        """Initialize the reinforcement learning agent.
        Args:
            state_dim: Dimension of the state space.
            action_dim: Dimension of the action space.
            hyperparams: Dictionary of hyperparameters.
            device: PyTorch device.
            seed: Random seed.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hyperparams = hyperparams
        self.device = device
        self.seed = seed
        self.total_env_steps = 0

        # To be populated by the concrete agent implementation
        self.models: Dict[str, torch.nn.Module] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.agent_var_keys: List[str] = []

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

    @abstractmethod
    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Process a single step of the environment."""
        raise NotImplementedError

    @abstractmethod
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select an action based on the current policy."""
        raise NotImplementedError

    @abstractmethod
    def learn(self, experiences: Any, gamma: float) -> None:
        """Update the agent's networks."""
        raise NotImplementedError

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(local_param.data)

    def save_state(self, path_prefix: str) -> None:
        """Save the agent's state."""
        try:
            os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
            for name, model in self.models.items():
                torch.save(model.state_dict(), f"{path_prefix}_{name}.pth")
            for name, optimizer in self.optimizers.items():
                torch.save(optimizer.state_dict(), f"{path_prefix}_{name}.pth")

            agent_vars = {key: getattr(self, key) for key in self.agent_var_keys}
            with open(f"{path_prefix}_agent_vars.pkl", "wb") as f:
                pickle.dump(agent_vars, f)

            logger.info(
                f"Agent state saved to prefix: {path_prefix} (Total steps: {self.total_env_steps})"
            )
        except Exception as e:
            logger.error(f"Saving agent state: {e}")

    def load_state(self, path_prefix: str) -> None:
        """Load the agent's state."""
        logger.info(f"Attempting to load agent state from prefix: {path_prefix}")
        loaded_something = False

        for name, model in self.models.items():
            path = f"{path_prefix}_{name}.pth"
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=self.device))
                loaded_something = True

        for name, optimizer in self.optimizers.items():
            path = f"{path_prefix}_{name}.pth"
            if os.path.exists(path):
                optimizer.load_state_dict(torch.load(path, map_location=self.device))

        vars_path = f"{path_prefix}_agent_vars.pkl"
        if os.path.exists(vars_path):
            with open(vars_path, "rb") as f:
                agent_vars = pickle.load(f)
            for key, value in agent_vars.items():
                setattr(self, key, value)
            loaded_something = True

        if loaded_something:
            logger.info(
                f"Agent state loaded successfully. Resuming at {self.total_env_steps} env steps."
            )
        else:
            raise FileNotFoundError(
                "No agent state files found at prefix. Agent will start fresh."
            )
