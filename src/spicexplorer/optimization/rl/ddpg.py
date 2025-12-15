import logging
import os
import pickle
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .base import BaseRLAgent, BaseActor, BaseCritic
from .hyperparameters import DDPGHyperparameters
from .replay_buffer import ReplayBuffer
from .utils import trunc_normal

logger = logging.getLogger("SpiceXplorer")


class DDPGAgent(BaseRLAgent):
    """Interacts with and learns from the environment using DDPG."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_model_class: BaseActor,
        critic_model_class: BaseCritic,
        hyperparams: DDPGHyperparameters,
        device: torch.device,
        seed: int = 0,
        model_load_path: Optional[str] = None,
    ):
        """
        Initialize an Agent object.
        """
        super().__init__(
            state_dim,
            action_dim,
            actor_model_class,
            critic_model_class,
            hyperparams.to_dict(),
            device,
            seed,
            model_load_path,
        )

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

        logger.info(f"DDPG Agent Initializing with seed: {self.seed}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Hyperparameters: {self.hyperparams}")

        # Actor Network (w/ Target Network)
        self.actor_local = actor_model_class(
            state_dim, action_dim, seed=self.seed, hyperparams=self.hyperparams
        ).to(self.device)
        self.actor_target = actor_model_class(
            state_dim, action_dim, seed=self.seed, hyperparams=self.hyperparams
        ).to(self.device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=self.hyperparams["lr_actor"]
        )
        self.hard_update(self.actor_local, self.actor_target)

        # Critic Network (w/ Target Network)
        self.critic_local = critic_model_class(
            state_dim, action_dim, seed=self.seed, hyperparams=self.hyperparams
        ).to(self.device)
        self.critic_target = critic_model_class(
            state_dim, action_dim, seed=self.seed, hyperparams=self.hyperparams
        ).to(self.device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(),
            lr=self.hyperparams["lr_critic"],
            weight_decay=self.hyperparams.get("weight_decay_critic", 0),
        )
        self.hard_update(self.critic_local, self.critic_target)

        # Noise process
        self.current_noise_sigma = self.hyperparams["noise_sigma_initial"]

        # Replay memory
        self.memory = ReplayBuffer(
            self.hyperparams["buffer_size"],
            self.hyperparams["batch_size"],
            self.device,
            self.seed,
        )

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.total_env_steps = 0

        if model_load_path:
            self.load_state(model_load_path)

    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(state, action, reward, next_state, done)
        self.total_env_steps += 1

        self.t_step = (self.t_step + 1) % self.hyperparams["update_every"]
        if self.t_step == 0:
            if (
                len(self.memory) >= self.hyperparams["batch_size"]
                and self.total_env_steps > self.hyperparams.get("initial_random_steps", 0)
            ):
                experiences = self.memory.sample()
                if experiences:
                    self.learn(experiences, self.hyperparams["gamma"])

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Returns actions for given state as per current policy."""
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        if add_noise and self.total_env_steps < self.hyperparams.get(
            "initial_random_steps", 0
        ):
            action = np.random.uniform(-1.0, 1.0, self.action_dim)
            return action.astype(np.float32)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state_tensor).cpu().data.numpy().flatten()
        self.actor_local.train()

        if add_noise:
            if self.hyperparams["noise_type"] == "uniform":
                noise_val = np.random.uniform(
                    -self.current_noise_sigma, self.current_noise_sigma, self.action_dim
                )
                action = action + noise_val
            elif self.hyperparams["noise_type"] == "truncnorm":
                action = trunc_normal(action, self.current_noise_sigma, low=-1.0, high=1.0)
            elif self.hyperparams["noise_type"] == "gaussian":
                noise_val = np.random.normal(
                    0, self.current_noise_sigma, size=self.action_dim
                )
                action = action + noise_val

            action = np.clip(action, -1.0, 1.0)

            self.current_noise_sigma = max(
                self.hyperparams["noise_sigma_min"],
                self.current_noise_sigma * self.hyperparams["noise_sigma_decay"],
            )

        return action.astype(np.float32)

    def learn(self, experiences, gamma: float):
        states, actions, rewards, next_states, dones = experiences

        # Update critic
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic_local.parameters(), self.hyperparams["grad_clip_critic"]
        )
        self.critic_optimizer.step()

        # Update actor
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.soft_update(self.critic_local, self.critic_target, self.hyperparams["tau"])
        self.soft_update(self.actor_local, self.actor_target, self.hyperparams["tau"])

        self.actor_loss_val = actor_loss.item()
        self.critic_loss_val = critic_loss.item()

    def save_state(self, path_prefix: str):
        try:
            os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
            torch.save(
                self.actor_local.state_dict(), f"{path_prefix}_actor_local.pth"
            )
            torch.save(
                self.actor_target.state_dict(), f"{path_prefix}_actor_target.pth"
            )
            torch.save(
                self.critic_local.state_dict(), f"{path_prefix}_critic_local.pth"
            )
            torch.save(
                self.critic_target.state_dict(), f"{path_prefix}_critic_target.pth"
            )
            torch.save(
                self.actor_optimizer.state_dict(),
                f"{path_prefix}_actor_optimizer.pth",
            )
            torch.save(
                self.critic_optimizer.state_dict(),
                f"{path_prefix}_critic_optimizer.pth",
            )

            agent_vars = {
                "total_env_steps": self.total_env_steps,
                "current_noise_sigma": self.current_noise_sigma,
                "t_step": self.t_step,
            }
            with open(f"{path_prefix}_agent_vars.pkl", "wb") as f:
                pickle.dump(agent_vars, f)

            logger.info(
                f"Agent state saved to prefix: {path_prefix} (Total steps: {self.total_env_steps})"
            )
        except Exception as e:
            logger.error(f"Saving DDPG agent state: {e}")

    def load_state(self, path_prefix: str):
        logger.info(
            f"Attempting to load DDPG agent state from prefix: {path_prefix}"
        )
        loaded_something = False
        try:
            if os.path.exists(f"{path_prefix}_actor_local.pth"):
                self.actor_local.load_state_dict(
                    torch.load(
                        f"{path_prefix}_actor_local.pth", map_location=self.device
                    )
                )
                loaded_something = True
            if os.path.exists(f"{path_prefix}_actor_target.pth"):
                self.actor_target.load_state_dict(
                    torch.load(
                        f"{path_prefix}_actor_target.pth", map_location=self.device
                    )
                )
            else:
                self.hard_update(self.actor_local, self.actor_target)

            if os.path.exists(f"{path_prefix}_critic_local.pth"):
                self.critic_local.load_state_dict(
                    torch.load(
                        f"{path_prefix}_critic_local.pth", map_location=self.device
                    )
                )
                loaded_something = True
            if os.path.exists(f"{path_prefix}_critic_target.pth"):
                self.critic_target.load_state_dict(
                    torch.load(
                        f"{path_prefix}_critic_target.pth", map_location=self.device
                    )
                )
            else:
                self.hard_update(self.critic_local, self.critic_target)

            if os.path.exists(f"{path_prefix}_actor_optimizer.pth"):
                self.actor_optimizer.load_state_dict(
                    torch.load(
                        f"{path_prefix}_actor_optimizer.pth",
                        map_location=self.device,
                    )
                )
            if os.path.exists(f"{path_prefix}_critic_optimizer.pth"):
                self.critic_optimizer.load_state_dict(
                    torch.load(
                        f"{path_prefix}_critic_optimizer.pth",
                        map_location=self.device,
                    )
                )

            if os.path.exists(f"{path_prefix}_agent_vars.pkl"):
                with open(f"{path_prefix}_agent_vars.pkl", "rb") as f:
                    agent_vars = pickle.load(f)
                self.total_env_steps = agent_vars.get("total_env_steps", 0)
                self.current_noise_sigma = agent_vars.get(
                    "current_noise_sigma", self.hyperparams["noise_sigma_initial"]
                )
                self.t_step = agent_vars.get("t_step", 0)
                loaded_something = True

            if loaded_something:
                logger.info(
                    f"Agent state loaded successfully. Resuming at {self.total_env_steps} env steps, noise sigma {self.current_noise_sigma:.4f}."
                )
            else:
                logger.warning(
                    "No agent state files found at prefix. Starting fresh."
                )

        except Exception as e:
            logger.error(f"Loading DDPG agent state: {e}. Agent will start fresh.")
            self.__init__(
                self.state_dim,
                self.action_dim,
                self.actor_local.__class__,
                self.critic_local.__class__,
                self.hyperparams,
                self.device,
                self.seed,
            )
