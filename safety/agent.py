from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
TensorBatch = List[torch.Tensor]


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self._log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self._min_log_std = min_log_std
        self._max_log_std = max_log_std
        self._min_action = min_action
        self._max_action = max_action

    def _get_policy(self, state: torch.Tensor) -> torch.distributions.Distribution:
        mean = self._mlp(state)
        log_std = self._log_std.clamp(self._min_log_std, self._max_log_std)
        policy = torch.distributions.Normal(mean, log_std.exp())
        return policy

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        policy = self._get_policy(state)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return log_prob

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy = self._get_policy(state)
        action = policy.rsample()
        action.clamp_(self._min_action, self._max_action)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        state_t = torch.tensor(state[None], dtype=torch.float32, device=device)
        policy = self._get_policy(state_t)
        if self._mlp.training:
            action = policy.rsample()
        else:
            action = policy.mean
        action.clamp_(self._min_action, self._max_action)
        return action.cpu().data.numpy().flatten()


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
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

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        value = self._mlp(state)
        return value

class Interventioner(nn.Module):

    def __init__(self, safe_actor, trigger_state):
        super().__init__()
        self.safe_actor = safe_actor
        self.trigger_state = trigger_state


    def check_safety(self, state):
        # In unsafe state, return False, otherwise True
        if np.sum(state) > self.trigger_state:
            return False
        else:
            return True

    def act(self, state, device = None):
        return self.safe_actor.act(state)


class SafeAgent:
    """
    Combines Actor, Critic, and Intervention Policy
    """
    def __init__(
            self,
            actor: nn.Module,
            critic: nn.Module,
            actor_optimizer: torch.optim.Optimizer,
            critic_optimizer: torch.optim.Optimizer,
            interventioner: nn.Module,
            gamma: float = 0.99,
            lambda_: float = 0.95,
            policy_clip: float = 1.0,
            value_clip: float = 1.0,
            updates_per_rollout: int = 10,

    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.interventioner = interventioner

        self.gamma = gamma
        self.lamda_ = lambda_
        self.policy_clip = policy_clip
        self.value_clip = value_clip

    # Acting
    def act(self, state: np.ndarray, device: str) -> [np.ndarray, bool]:
        # check if state is safe
        if self.interventioner.check_safety(state):
            action =  self.actor.act(state, device)
            intervention = False
        else:
            action =  self.interventioner.act(state)
            intervention = True

        return action, intervention

    def update_v2(self, states, actions, rewards, next_states, dones, intervention, **kwargs):
        # compute values
        values = self.critic(states)
        next_values = self.critic(next_states)

        # compute nn policy log_probs
        log_probs = self.actor.log_prob(states, actions)

        # compute GAE
        advantages = self.compute_GAE(rewards, values, next_values, dones)
        returns = advantages + values

        # compute critic loss
        critic_loss = self.critic_loss(values, returns)

        # compute actor loss
        actor_loss = self.actor_loss(advantages, log_probs)

        # update actor and critic
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}
    def update(self, batch):
        # Compute values
        batch["values"] = self.critic(batch["obs"])
        batch["next_values"] = self.critic(batch["next_obs"])

        # Compute nn policy log_probs
        batch["log_probs"] = self.actor.log_prob(batch["obs"], batch["actions"])

        # Compute GAE
        batch["advantages"] = self.compute_GAE(batch["rewards"], batch["values"].detach().numpy(), batch["next_values"].detach().numpy(), batch["dones"])
        batch["returns"] = batch["advantages"] + batch["values"]
        # Compute critic loss
        critic_loss = self.critic_loss(batch["values"], batch["returns"])
        # Compute actor loss
        actor_loss = self.actor_loss(batch["advantages"], batch["log_probs"])
        # Update actor and critic
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}

    def state_dict(self) -> Dict[str, Any]:
        # overloading this method to save interventioner as well
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "interventioner": self.interventioner
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        # also loads the interventioner
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic_1"])
        self.interventioner = state_dict["interventioner"]




    # Updates
    def actor_loss(self, advantages, log_probs):
        loss = -(advantages * log_probs).mean()
        return loss

    def critic_loss(self, values, target):
        loss = (1/2 * (values - target)**2).mean()
        return loss



    def compute_GAE(self, rewards, values, next_values, dones):
        # Need to fix this and make sure there are no dones...
        with torch.no_grad:
            deltas = rewards + self.gamma * next_values * (1 - dones) - values
            gaes = torch.zeros_like(deltas)
            last_gae = 0
            for t in reversed(range(len(deltas) - 1)):
                gaes[t] last_gae = delta +  self.gamma * self.lamda_* gaes[t + 1] * (1 - dones[t])
        return gaes



