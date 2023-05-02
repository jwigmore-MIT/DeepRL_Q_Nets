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
        init_std: float = 1.0,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
        clamp_min: float = -1.0,
        clamp_max: float = 1.0,
        min_actions: np.ndarray = None,
        max_actions: np.ndarray = None,
        bias: np.ndarray = None,
        mask_ranges: [np.ndarray] = None,
):
        super().__init__()
        final_layer = nn.Linear(hidden_dim, action_dim, bias=True)

        if bias is not None:
            # check to make sure the bias is the right shape
            assert bias.shape == (action_dim,)
            final_layer.bias.data += torch.tensor(bias)

        self._mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            final_layer,
        )
        init_log_std = torch.ones(action_dim, dtype=torch.float32) * np.log(init_std)
        self._log_std = nn.Parameter(init_log_std)
        self._min_log_std = min_log_std
        self._max_log_std = max_log_std
        if mask_ranges is not None:
            self._mask_ranges = torch.Tensor(mask_ranges)
        # self._clamp_min = clamp_min
        # self._clamp_max = clamp_max
        # if min_actions is None:
        #     min_actions = -np.ones(action_dim, dtype=np.float32)
        # if max_actions is None:
        #     max_actions = np.ones(action_dim, dtype=np.float32)
        # self._min_actions = torch.Tensor(min_actions)
        # self._max_actions = torch.Tensor(max_actions)

    def _get_policy(self, state: torch.Tensor) -> torch.distributions.Distribution:
        mean = self._mlp(state)
        log_std = self._log_std.clamp(self._min_log_std, self._max_log_std)
        policy = torch.distributions.Normal(mean, log_std.exp())
        return policy

    def log_prob(self, state: torch.Tensor, action: torch.Tensor, extra= False) -> torch.Tensor:
        policy = self._get_policy(state)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        # if log_prob.min().item() < -100:
        #     print('log_prob is less than -100')
        if not extra:
            return log_prob
        else:
            policy_means = policy.mean
            policy_stds = policy.scale[-1]
            return log_prob, policy_means, policy_stds

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy = self._get_policy(state)
        action = policy.rsample()
        #action.clamp_(self._min_actions)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        state_t = torch.tensor(state[None], dtype=torch.float32, device=device)
        policy = self._get_policy(state_t)
        if self._mlp.training:
            action = policy.rsample()
        else:
            action = policy.mean
        if self._mask_ranges is not None:
            action = action.clamp_(self._mask_ranges[0, :], self._mask_ranges[1, :])
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

    def __init__(self, safe_actor, trigger_state = 0):
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
            normalize_values: bool = False,
            gamma: float = 0.99,
            lambda_: float = 0.95,
            ppo: bool = False,
            ppo_clip_coef: float = 0.25,
            value_clip: float = 1.0,
            updates_per_rollout: int = 1,
            pretrain: bool = False,

    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self._critic = critic
        self.critic_optimizer = critic_optimizer
        self.interventioner = interventioner

        self.updates_per_rollout = updates_per_rollout
        self.normalize_values = normalize_values
        self.value_mean = None
        self.value_std = None


        self.gamma = gamma
        self.lambda_ = lambda_

        self.ppo = ppo
        self.ppo_clip_coef = ppo_clip_coef
        self.value_clip = value_clip

        self.pretrain = pretrain

    def critic(self, state: torch.Tensor, unnormalize = False) -> torch.Tensor:
        value = self._critic(state)
        if unnormalize and self.normalize_values and \
                self.value_mean is not None and self.value_std is not None:
            value = value * max(self.value_std, 1e-8) + self.value_mean
        return value

    # Acting
    def act(self, state: np.ndarray, device: str) -> [np.ndarray, bool]:
        # check if state is safe
        if self.pretrain:
            action = self.interventioner.act(state)
            intervention = True
        else:
            if self.interventioner.check_safety(state):
                action =  self.actor.act(state, device)
                intervention = False
            else:
                action =  self.interventioner.act(state)
                intervention = True

        return action, intervention

    def update(self, batch):
        results = {"critic_loss":[],
                   "values_mean":[],
                   "values_std":[],
                   "targets_mean":[],
                    "targets_std":[],
                   "actor_loss":[],
                   "vt_error":[],
                   "log_probs":[],
                   "advantages":[],
                   "avg_policy_means": [],
                   "policy_stds": []}
        if self.ppo:
            with torch.no_grad():
                curr_log_probs = self.actor.log_prob(batch["obs"], batch["actions"])
        else:
            curr_log_probs = None
        for i in range(self.updates_per_rollout):

            critic_results = self.update_critic(batch)
            actor_results = self.update_actor(batch, curr_log_probs )


            results["critic_loss"].append(critic_results["critic_loss"])
            results["values_mean"].append(critic_results["values"].mean().item())
            results["values_std"].append(critic_results["values"].std().item())
            results["targets_mean"].append(critic_results["targets"].mean().item())
            results["targets_std"].append(critic_results["targets"].std().item())
            results["vt_error"].append((critic_results["values"] - critic_results["targets"]).mean().item())
            results["actor_loss"].append(actor_results["actor_loss"])
            results["log_probs"].append(actor_results["log_probs"].mean().item())
            results["advantages"].append(actor_results["advantages"].mean().item())
            results["avg_policy_means"].append(actor_results["policy_means"].mean().item())
            results["policy_stds"].append(actor_results["policy_stds"].mean().item())

        return results

    def update_critic(self, batch):
        # Compute the advantage
        target = self.compute_critic_target(batch)  # target can be normalized or unnormalized
        norm_values = self.critic(batch["obs"], unnormalize=False) # never fit to unnormalized values
        critic_loss = self.critic_loss(norm_values, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return {"critic_loss": critic_loss.item(),
                "values": norm_values,
                "targets": target}

    def compute_critic_target(self, batch):
        with torch.no_grad():
            values = self.critic(batch["obs"], unnormalize= True) # unnormalized critic output
            next_value = self.critic(batch["next_obs"][-1], unnormalize=True) # unnormalized critic output
            rewards = torch.Tensor(batch["rewards"]) # from batch
            advantages = self.compute_GAE(rewards, values, next_value, batch["dones"]) # unnormalized advantages
            target = advantages + values # unnormalized target
            if self.normalize_values:
                self.value_mean = target.mean().item()
                self.value_std = target.std().item()
                target = (target - self.value_mean) / max(self.value_std, 1e-8)  #normalized target
            return target

    def critic_loss(self, values, target):
        loss = torch.nn.functional.mse_loss(values, target)
        #loss = (1/2 * (values - target)**2).mean()
        return loss

    def fit_critic(self, batch, fit_epochs=100):
        metrics = {"critic_loss": [],
                   "values_mean": [],
                   "values_std": [],
                   "targets_mean": [],
                   "targets_std": [],
                   "deviation_mean": [],
                   "deviation_std": [],
                   }

        for i in range(fit_epochs):
            critic_results = self.update_critic(batch)
            metrics["critic_loss"].append(critic_results["critic_loss"])
            metrics["values_mean"].append(critic_results["values"].mean().item())
            metrics["values_std"].append(critic_results["values"].std().item())
            metrics["targets_mean"].append(critic_results["targets"].mean().item())
            metrics["targets_std"].append(critic_results["targets"].std().item())
            metrics["deviation_mean"].append((critic_results["values"] - critic_results["targets"]).abs().mean().item())
            metrics["deviation_std"].append((critic_results["values"] - critic_results["targets"]).abs().std().item())

        for key, value in metrics.items():
            metrics[key] = np.array(value)
        return metrics

    def update_actor(self, batch, curr_log_probs = None):
        if self.ppo:
            actor_results = self.ppo_actor_update(batch, curr_log_probs)
        else:
            actor_results = self.pg_actor_update(batch)
        return actor_results

    def pg_actor_update(self, batch):
        # Get advtantages
        with torch.no_grad():
            values = self.critic(batch["obs"], unnormalize= True)
            next_value = self.critic(batch["next_obs"][-1], unnormalize= True)
            rewards = torch.Tensor(batch["rewards"])
            advantages = self.compute_GAE(rewards, values, next_value, batch["dones"])
        # Get log probs
        log_probs, means, std = self.actor.log_prob(batch["obs"], batch["actions"], extra = True)
        # Compute actor loss
        actor_loss = self.actor_loss(advantages, log_probs)
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return {"actor_loss": actor_loss.item(),
                "log_probs": log_probs,
                "advantages": advantages,
                "policy_means": means,
                "policy_stds": std}
        # Updates

    def ppo_actor_update(self, batch, curr_log_probs):
        with torch.no_grad():
            values = self.critic(batch["obs"], unnormalize= True)
            next_value = self.critic(batch["next_obs"][-1], unnormalize= True)
            rewards = torch.Tensor(batch["rewards"])
            advantages = self.compute_GAE(rewards, values, next_value, batch["dones"])
        new_log_probs, means, std = self.actor.log_prob(batch["obs"], batch["actions"], extra = True)
        log_ratio = (new_log_probs - curr_log_probs)
        ratio = log_ratio.exp()
        with torch.no_grad():
            approx_kl = -log_ratio.mean()
            clip_frac = ((ratio - 1.0).abs() > self.ppo_clip_coef).float().mean().item()

        # compute loss
        pg_loss1 = -advantages * ratio # unclipped loss
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.ppo_clip_coef, 1.0 + self.ppo_clip_coef) # clipped loss
        actor_loss = torch.max(pg_loss1, pg_loss2).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return {"actor_loss": actor_loss.item(),
                "log_probs": new_log_probs,
                "advantages": advantages,
                "policy_means": means,
                "policy_stds": std}



    def actor_loss(self, advantages, log_probs):
        loss = (-advantages * log_probs).mean()
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


    def set_safe_threshold(self, threshold):
        self.interventioner.trigger_state = threshold

    # Saving and Loading
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
