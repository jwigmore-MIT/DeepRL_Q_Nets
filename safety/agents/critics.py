import torch.nn as nn

import torch
import numpy as np
from safety.agents.utils import layer_init, mlp_init, get_activation


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        hidden_layers: int = 2,
        activation: str = "tanh",
    ):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._mlp = mlp_init(state_dim, 1, hidden_layers, hidden_dim, activation)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        value = self._mlp(state)
        return value


class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(np.array(state_dim).prod() + np.prod(action_dim), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def init_critic(config, state_dim, action_dim):

    if config.critic.type == "Standard":
        critic = Critic(state_dim, config.critic.hidden_dim)
        return critic
    elif config.critic.type == "SAC":
        qf1 = SoftQNetwork(state_dim, action_dim, config.critic.hidden_dim)
        qf2 = SoftQNetwork(state_dim, action_dim, config.critic.hidden_dim)
        qf1_target = SoftQNetwork(state_dim, action_dim, config.critic.hidden_dim)
        qf2_target = SoftQNetwork(state_dim, action_dim, config.critic.hidden_dim)
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())
        return qf1, qf2, qf1_target, qf2_target

