from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


class BaseActor(torch.nn.Module, ABC):
    """Abstract base class for actor networks."""

    @abstractmethod
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seed: int,
        hyperparams: Optional[Dict[str, Any]] = None,
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
        hyperparams: Optional[Dict[str, Any]] = None,
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

    @abstractmethod
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_model_class: BaseActor,
        critic_model_class: BaseCritic,
        hyperparams: Dict[str, Any],
        device: torch.device,
        seed: int,
        model_load_path: Optional[str] = None,
    ) -> None:
        """Initialize the reinforcement learning agent.
        Args:
            state_dim: Dimension of the state space.
            action_dim: Dimension of the action space.
            actor_model_class: Class for the actor network.
            critic_model_class: Class for the critic network.
            hyperparams: Dictionary of hyperparameters.
            device: PyTorch device.
            seed: Random seed.
            model_load_path: Path to load a pre-trained model.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hyperparams = hyperparams
        self.device = device
        self.seed = seed

    @abstractmethod
    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Process a single step of the environment.
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether the episode is done.
        """
        raise NotImplementedError

    @abstractmethod
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select an action based on the current policy.
        Args:
            state: Current state.
            add_noise: Whether to add noise for exploration.
        Returns:
            Action to take.
        """
        raise NotImplementedError

    @abstractmethod
    def learn(self, experiences: Any, gamma: float) -> None:
        """Update the agent's networks.
        Args:
            experiences: A batch of experiences.
            gamma: Discount factor.
        """
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

    @abstractmethod
    def save_state(self, path_prefix: str) -> None:
        """Save the agent's state.
        Args:
            path_prefix: Prefix for the file paths.
        """
        raise NotImplementedError

    @abstractmethod
    def load_state(self, path_prefix: str) -> None:
        """Load the agent's state.
        Args:
            path_prefix: Prefix for the file paths.
        """
        raise NotImplementedError
