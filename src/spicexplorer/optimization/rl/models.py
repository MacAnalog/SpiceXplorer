import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .base import BaseActor, BaseCritic

class Actor(BaseActor):
    def __init__(self, state_dim: int, action_dim: int, seed: int, hyperparams: Dict[str, Any]):
        super().__init__(state_dim, action_dim, seed)
        self.seed = torch.manual_seed(seed)
        hidden_units = hyperparams.get("hidden_units_actor", (256, 128))
        
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

class Critic(BaseCritic):
    def __init__(self, state_dim: int, action_dim: int, seed: int, hyperparams: Dict[str, Any]):
        super().__init__(state_dim, action_dim, seed)
        self.seed = torch.manual_seed(seed)
        hidden_units = hyperparams.get("hidden_units_critic", (256, 128))

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
