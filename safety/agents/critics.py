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
        # self._mlp = nn.Sequential(
        #     nn.Linear(state_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 1),
        # )

        self._mlp = mlp_init(state_dim, 1, hidden_layers, hidden_dim, activation)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        value = self._mlp(state)
        return value

class PopArtCritic(nn.Module):
    # Implementation of https://arxiv.org/pdf/1602.07714.pdf

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        hidden_layers: int = 2,
        activation: str = "tanh",
        beta: float= 4e-4,
    ):
        super().__init__()
        self.beta = beta
        self._mlp = mlp_init(state_dim, 1, hidden_layers, hidden_dim, activation, final_linear=False) # norm_value

        self.weight = torch.nn.Parameter(torch.Tensor(1, hidden_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(1))

        self.register_buffer('mu', torch.zeros(1, requires_grad=False))
        self.register_buffer('sigma', torch.ones(1, requires_grad=False))

        self.reset_parameters()

    def __str__(self):
        return "PopArtCritic"
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        # Get output of final non-linear layer
        z = self._mlp(inputs)
        # Apply Linear Layer (w.T * z + b)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        normalized_output = z.mm(self.weight.t())
        normalized_output += self.bias.unsqueeze(0).expand_as(normalized_output)

        with torch.no_grad():
            output = normalized_output * self.sigma + self.mu

        return [output, normalized_output]

    def update_parameters(self, vs):
        oldmu = self.mu
        oldsigma = self.sigma

        vs = vs
        n = vs.shape[0]
        mu = vs.sum((0, 1)) / n
        nu = torch.sum(vs ** 2, (0, 1)) / n
        sigma = torch.sqrt(nu - mu ** 2)
        sigma = torch.clamp(sigma, min=1e-4, max=1e+6)

        # mu[torch.isnan(mu)] = self.mu[torch.isnan(mu)]
        # sigma[torch.isnan(sigma)] = self.sigma[torch.isnan(sigma)]
        # mu = self.mu
        # sigma = self.sigma

        self.mu = (1 - self.beta) * self.mu + self.beta * mu
        self.sigma = (1 - self.beta) * self.sigma + self.beta * sigma

        self.weight.data = (self.weight.t() * oldsigma / self.sigma).t()
        self.bias.data = (oldsigma * self.bias + oldmu - self.mu) / self.sigma



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

