import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseActor, BaseCritic
from .hyperparameters import DDPGHyperparameters

# ------------------------
# Actor (Policy) Model
# ------------------------
class MLPActor(BaseActor):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seed: int,
        hyperparams: DDPGHyperparameters,
    ):
        super().__init__(state_dim, action_dim, seed, hyperparams)
        hidden_units = hyperparams.actor.hidden_units

        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)
        self.network.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, state):
        return F.tanh(self.network(state))



# ------------------------
# Critic (Value) Model
# ------------------------
class MLPCritic(BaseCritic):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seed: int,
        hyperparams: DDPGHyperparameters,
    ):
        super().__init__(state_dim, action_dim, seed, hyperparams)
        hidden_units = hyperparams.critic.hidden_units

        # Q1 architecture
        layers1 = []
        input_dim = state_dim + action_dim
        for hidden_dim in hidden_units:
            layers1.append(nn.Linear(input_dim, hidden_dim))
            layers1.append(nn.ReLU())
            input_dim = hidden_dim
        layers1.append(nn.Linear(input_dim, 1))
        self.net1 = nn.Sequential(*layers1)

        self.net1.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        return self.net1(torch.cat([state, action], dim=1))
    
