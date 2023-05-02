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
            pretrain: bool = False,

    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.interventioner = interventioner


        self.gamma = gamma
        self.lambda_ = lambda_
        self.policy_clip = policy_clip
        self.value_clip = value_clip

        self.pretrain = pretrain

    # Acting
    def act(self, state: np.ndarray, device: str) -> [np.ndarray, bool]:
        # check if state is safe

        if self.interventioner.check_safety(state) and not self.pretrain:
            action =  self.actor.act(state, device)
            intervention = False
        else:
            action =  self.interventioner.act(state)
            intervention = True

        return action, intervention
    def set_safe_threshold(self, threshold):
        self.interventioner.trigger_state = threshold
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
    def update_old(self, batch):
        # Compute values
        batch["values"] = self.critic(batch["obs"])
        batch["values"].retain_grad()
        batch["next_values"] = self.critic(batch["next_obs"])

        # Compute nn policy log_probs
        batch["log_probs"] = self.actor.log_prob(batch["obs"], batch["actions"])
        batch["log_probs"].requires_grad_(True)

        # Compute GAE
        with torch.no_grad():
            batch["advantages"] = torch.Tensor(self.compute_GAE(batch["rewards"], batch["values"], batch["next_values"], batch["dones"]))
        batch["returns"] = batch["advantages"] + batch["values"]
        # Compute critic loss
        # Set requires grad to true for batch["values"] to compute gradient
        #batch["values"].requires_grad_(True)
        #critic_loss = self.critic_loss(batch["values"], batch["returns"]+batch["values"])
        critic_loss = self._critic_loss(batch["obs"], batch["rewards"], batch["next_obs"], batch["dones"])
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = self.actor_loss(batch["advantages"], batch["log_probs"])
        # Update actor and critic
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(),
                "values": batch["values"].mean().item(), "returns": batch["returns"].mean().item()}
    # Saving and Loading

    def fit_critic(self, batch, fit_epochs = 100):
        metrics = {"critic_loss": [],
                   "values": [],
                   "targets": [],
                   "deviation": []}

        for i in range(fit_epochs):
            critic_results = self.update_critic(batch)
            metrics["critic_loss"].append(critic_results["critic_loss"])
            metrics["values"].append(critic_results["values"].mean().item())
            metrics["targets"].append(critic_results["targets"].mean().item())
            metrics["deviation"].append((critic_results["values"]-critic_results["targets"]).abs().mean().item())
        for key, value in metrics.items():
            metrics[key] = np.array(value)
        return metrics
    def update(self, batch):
        critic_results = self.update_critic(batch)
        actor_results = self.update_actor(batch)

        results = {}
        results["critic_loss"] = critic_results["critic_loss"]
        results["values"] = critic_results["values"].mean().item()
        results["targets"] = critic_results["targets"].mean().item()

        results["actor_loss"] = actor_results["actor_loss"]


        return results

    def state_dict(self) -> Dict[str, Any]:
        # overloading this method to save interventioner as well
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            #"interventioner": self.interventioner
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        # also loads the interventioner
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic_1"])
        #self.interventioner = state_dict["interventioner"]



    def update_critic(self, batch):
        # Compute the advantage
        with torch.no_grad():
            values = self.critic(batch["obs"])
            #values2 = self.TD_V_estimate(batch)
            next_value = self.critic(batch["next_obs"][-1])
            rewards = torch.Tensor(batch["rewards"])
            advantages = self.compute_GAE(rewards, values, next_value, batch["dones"])
            target = advantages + values
        new_values = self.critic(batch["obs"])
        critic_loss = self.critic_loss(new_values, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return {"critic_loss": critic_loss.item(),
                "values": new_values,
                "targets": target}

    def update_actor(self, batch):
        # Get advtantages
        with torch.no_grad():
            values = self.critic(batch["obs"])
            next_value = self.critic(batch["next_obs"][-1])
            rewards = torch.Tensor(batch["rewards"])
            advantages = self.compute_GAE(rewards, values, next_value, batch["dones"])
        # Get log probs
        log_probs = self.actor.log_prob(batch["obs"], batch["actions"])
        # Compute actor loss
        actor_loss = self.actor_loss(advantages, log_probs)
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return {"actor_loss": actor_loss.item(),
                "log_probs": log_probs}
    # Updates
    def actor_loss(self, advantages, log_probs):
        loss = (advantages * log_probs).mean()
        return loss

    def critic_loss(self, values, target):
        loss = torch.nn.functional.mse_loss(values, target)
        #loss = (1/2 * (values - target)**2).mean()
        return loss

    def _critic_loss(self, states, rewards, next_states, dones):
        with torch.no_grad():
            values = self.critic(states)
            next_value = self.critic(next_states[-1])
            adv = torch.Tensor(self.compute_GAE(rewards, values, next_value, dones))
            target = adv + values

        values = self.critic(states)
        loss = torch.nn.functional.mse_loss(values, target)
        return loss



    def compute_GAE(self, rewards, values, next_val, dones):
        # Need to fix this and make sure there are no dones...
        with torch.no_grad():

            adv = torch.zeros_like(rewards)
            last_gae_lam = 0
            for t in reversed(range(len(adv))):
                if t == len(adv) - 1:
                    next_value = next_val
                else:
                    next_value = values[t + 1]
                delta = rewards[t] + self.gamma * next_value  - values[t]
                adv[t] = last_gae_lam =  delta +  self.gamma * self.lambda_* last_gae_lam
        return adv

    def TD_V_estimate(self, batch, N = 100):
        # TD estimate of the value function using rewards
        # N is the number of steps to look ahead
        with torch.no_grad():
            values  = np.zeros(len(batch["rewards"]))
            for i in range(len(batch["rewards"])):
                steps_left = min(N, len(batch["rewards"]) - i)
                for j in range(steps_left):
                    values[i] += self.gamma**j * batch["rewards"][i+j]
                values[i] += self.critic(batch["next_obs"][i]).item() * self.gamma**N
        return values




def compute_GAE(rewards, values, next_values, dones, gamma = 0.99, lambda_ = 0.95):
    # Need to fix this and make sure there are no dones...
    with torch.no_grad():
        adv = torch.zeros_like(rewards)
        last_gae_lambda = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_values[t]
            else:
                next_val = values[t + 1]
            delta = rewards[t] + gamma  * next_val  - values[t]
            adv[t] = last_gae_lambda = delta + gamma * lambda_ * last_gae_lambda

        # deltas = rewards + gamma * next_values - values
        # adv = torch.zeros_like(deltas)
        #
        # for t in reversed(range(len(deltas))):
        #     if t == len(deltas) - 1:
        #         adv[t] = deltas[t]
        #     else:
        #         adv[t] = deltas[t] +  gamma * lambda_* adv[t + 1]
    return adv


def GAE_test():
    rewards = torch.Tensor([1,2,3])
    values = torch.Tensor([4,5,6])
    next_values = torch.Tensor([5,6,7])
    dones = torch.Tensor([0,0,0])
    adv = compute_GAE(rewards, values, next_values, dones, gamma = 1.0, lambda_ = 1.0)
    return adv


if __name__ == "__main__":
    adv = GAE_test()
