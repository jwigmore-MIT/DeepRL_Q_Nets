from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
TensorBatch = List[torch.Tensor]

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
class MultiDiscreteActor(nn.Module):

    def __init__(self, state_dim, hidden_dim, nvec: np.ndarray):
        super().__init__()

        self.network = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU()
        )
        self.nvec = nvec
        self.actor_head = layer_init(nn.Linear(hidden_dim, np.sum(self.nvec)), std=0.01)
        self.train = True

    def _get_policy(self, state: torch.Tensor) -> torch.distributions.Distribution:
        "This method returns a list of distributions, one for each action dimension"
        hidden = self.network(state)
        logits = self.actor_head(hidden)
        split_logits = torch.split(logits, self.nvec.tolist(), dim=1) # :
        multi_cat = [torch.distributions.Categorical(logits=logits) for logits in split_logits]
        return multi_cat

    def log_prob(self, state: torch.Tensor, action: torch.Tensor, extra = False, do_sum = True) -> torch.Tensor:
        multi_cat = self._get_policy(state)
        action_mT = torch.transpose(action, 0, 1)
        log_probs = []
        entropy = []
        for a, categorical in zip(action_mT, multi_cat):
            log_probs.append(categorical.log_prob(a))
            entropy.append(categorical.entropy())

        logprob = torch.stack(log_probs)
        entropy = torch.stack(entropy)
        if do_sum:
            log_probs = logprob.sum(0)
        else:
            log_probs = logprob
        #logprob = torch.stack([categorical.log_prob(a.T) for a, categorical in zip(action_mT, multi_cat)])
        if extra:
            return {"log_probs":log_probs,
                    "entropy": entropy.mean(0)}# "probs": probs, }
        else:
            return log_probs

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        multi_cat = self._get_policy(state)
        action = self.sample_from_multicat(multi_cat)
        log_prob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_cat)])
        return action.T, log_prob.sum(0)

    def act(self, state: np.ndarray, device = "cpu") -> torch.Tensor:
        state_t = torch.tensor(state[None], dtype=torch.float32, device=device)
        multi_cat = self._get_policy(state_t)
        if self.train:
            action = self.sample_from_multicat(multi_cat)
        else:
            action = torch.stack([categorical.probs.argmax() for categorical in multi_cat])
        return action.T.cpu().data.numpy().flatten()

    def sample_from_multicat(self, multi_cat: List[torch.distributions.Categorical]) -> torch.Tensor:
        samples = []
        for categorical in multi_cat:
            # check if categorical.probs is empty
            if categorical.probs.nelement() == 0:
                samples.append(torch.tensor([0]))
            else:
                samples.append(categorical.sample())
        return torch.stack(samples)






class BetaActor(nn.Module):

    def __init__(self, state_dim: int,
        action_dim: int,
        hidden_dim: int,
        action_boundaries: [np.ndarray] = None,
        ):
        super().__init__()
        assert action_boundaries is not None
        self.action_space_low = torch.Tensor(action_boundaries[0,:])
        self.action_space_high = torch.Tensor(action_boundaries[1,:])

        self.affine = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.alpha_pre_softplus = nn.Linear(hidden_dim, action_dim)
        self.beta_pre_softplus = nn.Linear(hidden_dim, action_dim)
        self.softplus = nn.Softplus()

    def get_policy(self, state: torch.Tensor) -> torch.distributions.Distribution:
        x = self.affine(state)
        alpha = torch.add(self.softplus(self.alpha_pre_softplus(x)), 1.)
        beta = torch.add(self.softplus(self.beta_pre_softplus(x)), 1.)
        return torch.distributions.Beta(alpha, beta)
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy = self.get_policy(state)
        beta_action = policy.rsample()
        log_prob = policy.log_prob(beta_action).sum(-1, keepdim=True)
        action = self.scale_action(beta_action)

        return action, log_prob

    def act(self, state: torch.Tensor, device = "cpu") -> torch.Tensor:
        state_t = torch.tensor(state[None], dtype=torch.float32, device=device)
        policy = self.get_policy(state_t)
        if self.affine.training:
            beta_action = policy.rsample()
        else:
            beta_action = policy.mean
        action = self.scale_action(beta_action)
        return action.cpu().data.numpy().flatten()
    def log_prob(self, state: torch.Tensor, action: torch.Tensor, extra= False) -> torch.Tensor:
        #in this case the action will be scaled to the range of the action space
        policy = self.get_policy(state)
        beta_action = self.inv_scale_action(action) #unscale the action
        log_prob = policy.log_prob(beta_action).nan_to_num(neginf=1e-8 ).sum(-1, keepdim=True)
        if not extra:
            return log_prob
        else:
            policy_means = policy.mean
            policy_stds = policy.stddev[-1]
            return log_prob, policy_means, policy_stds

    def scale_action(self, beta_action: torch.Tensor) -> torch.Tensor:

        return beta_action * (self.action_space_high - self.action_space_low) + self.action_space_low

    def inv_scale_action(self, action: torch.Tensor) -> torch.Tensor:
        return ((action - self.action_space_low) / (self.action_space_high - self.action_space_low)).nan_to_num(0)
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
            kl_coef: float = 0.0, # Beta in PPO paper
            kl_target: float = None,
            intervention_penalty: float = 0.0,
            grad_clip: float = None,
            value_clip: float = 1.0,
            updates_per_rollout: int = 1,
            awac_lambda: float = 1.0,
            exp_adv_max: float = 100,


            pretrain: bool = False,
            normalized_states: Union[bool, str] = False, # False, "gym"

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
        self.kl_coef = kl_coef
        self.kl_target = kl_target
        self.grad_clip = grad_clip


        self._awac_lambda = awac_lambda
        self._exp_adv_max = exp_adv_max

        self.intervention_penalty = intervention_penalty

        self.normalized_states = normalized_states

        self.pretrain = pretrain

    def critic(self, state: torch.Tensor, unnormalize = False) -> torch.Tensor:
        value = self._critic(state)
        if unnormalize and self.normalize_values and \
                self.value_mean is not None and self.value_std is not None:
            value = value * max(self.value_std, 1e-8) + self.value_mean
        return value

    # Acting
    def act(self, state: np.ndarray, device: str, true_state = None) -> [np.ndarray, bool]:
        if true_state is None:
            true_state = state
        # check if state is safe
        if self.pretrain:
            action = self.interventioner.act(true_state)
            intervention = True
        else:
            if self.interventioner.check_safety(true_state):
                action =  self.actor.act(state, device)
                intervention = False
            else:
                action =  self.interventioner.act(true_state)
                intervention = True

        return action, intervention

    def update(self, batch):

        results = {}

        # results = {"critic_loss":[],
        #            "values_mean":[],
        #            "values_std":[],
        #            "targets_mean":[],
        #             "targets_std":[],
        #            "actor_loss":[],
        #            "vt_error":[],
        #            "log_probs":[],
        #            "advantages":[],
        #            "avg_policy_means": [],
        #            "policy_stds": []}

        if self.ppo:
            with torch.no_grad():
                curr_log_probs = self.actor.log_prob(batch["obs"], batch["actions"])
        else:
            curr_log_probs = None
        for i in range(self.updates_per_rollout):

            critic_results = self.update_critic(batch)
            actor_results = self.update_actor(batch, curr_log_probs)


            for key in critic_results.keys():
                if key not in results.keys():
                    results[key] = []
                else:
                    results[key].append(critic_results[key])
            for key in actor_results.keys():
                if key not in results.keys():
                    results[key] = []
                else:
                    results[key].append(actor_results[key])
            if "update_epochs" not in results.keys():
                results["update_epochs"] = [i+1]
            else:
                results["update_epochs"].append(i+1)

            if "stop_update" in actor_results.keys():
                if actor_results["stop_update"]:
                    break


        return results

    def update_critic(self, batch):
        # Compute the advantage
        target = self.compute_critic_target(batch)  # target can be normalized or unnormalized
        norm_values = self.critic(batch["obs"], unnormalize=False) # never fit to unnormalized values
        critic_loss = self.critic_loss(norm_values, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Compute "Critic Error" = (values - targets)
        critic_error = (norm_values - target)
        max_critic_error = critic_error.max().item()
        min_critic_error = critic_error.min().item()
        if abs(max_critic_error) > abs(min_critic_error):
            max_critic_dev = max_critic_error
        else:
            max_critic_dev = min_critic_error
        return {"critic_loss": critic_loss.item(),
                "max_critic_dev: ": max_critic_dev,
                "avg_critic_error: ": critic_error.abs().mean().item()}

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
        # metrics = {"critic_loss": [],
        #            "values_mean": [],
        #            "values_std": [],
        #            "targets_mean": [],
        #            "targets_std": [],
        #            "deviation_mean": [],
        #            "deviation_std": [],
        #            }
        metrics = {}

        for i in range(fit_epochs):
            critic_results = self.update_critic(batch)

            for key in critic_results.keys():
                if key not in metrics.keys():
                    metrics[key] = [critic_results[key]]
                else:
                    metrics[key].append(critic_results[key])
            # metrics["critic_loss"].append(critic_results["critic_loss"])
            # metrics["values_mean"].append(critic_results["values"].mean().item())
            # metrics["values_std"].append(critic_results["values"].std().item())
            # metrics["targets_mean"].append(critic_results["targets"].mean().item())
            # metrics["targets_std"].append(critic_results["targets"].std().item())
            # metrics["deviation_mean"].append((critic_results["values"] - critic_results["targets"]).abs().mean().item())
            # metrics["deviation_std"].append((critic_results["values"] - critic_results["targets"]).abs().std().item())

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
        #log_probs, means, std = self.actor.log_prob(batch["obs"], batch["actions"], extra = True)
        result = self.actor.log_prob(batch["obs"], batch["actions"], extra = True)
        # Compute actor loss
        actor_loss = self.actor_loss(advantages, result["log_probs"])
        for key in result.keys():
            result[key] = result[key].mean().item()
        result["actor_loss"] = actor_loss.item()

        #result["advantages"] = advantages
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return result
        # Updates

    def ppo_actor_update(self, batch, curr_log_probs):
        update = True
        with torch.no_grad():
            values = self.critic(batch["obs"], unnormalize= True)
            next_value = self.critic(batch["next_obs"][-1], unnormalize= True)
            rewards = torch.Tensor(batch["rewards"]) + self.intervention_penalty*torch.Tensor(batch["interventions"])
            advantages = self.compute_GAE(rewards, values, next_value, batch["dones"])
        result = self.actor.log_prob(batch["obs"], batch["actions"], extra = True)
        log_ratio = (result["log_probs"] - curr_log_probs)
        ratio = log_ratio.exp()

        with torch.no_grad():
            clip_frac = ((ratio - 1.0).abs() > self.ppo_clip_coef).float().mean().item()
        if self.kl_coef == 0:
            with torch.no_grad: approx_kl = -log_ratio.mean()
        else:
            approx_kl = -log_ratio.mean()
        if self.kl_target is not None:
            if approx_kl > self.kl_target:
                # if the KL is too high, we stop updating the policy
                update = False
                #result["stop_update"] = True


        # compute loss
        pg_loss1 = -advantages * ratio # unclipped loss
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.ppo_clip_coef, 1.0 + self.ppo_clip_coef) # clipped loss
        kl_loss = self.kl_coef * approx_kl
        actor_loss = torch.max(pg_loss1, pg_loss2).mean() + kl_loss
        if update:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_optimizer.step()
            pg_magnitude = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0) for p in list(self.actor.parameters())]), 2.0)

            #result["stop_update"] = False

        else:
            pg_magnitude = torch.Tensor([0])
        for key in result.keys():
            result[key] = result[key].mean().item()
        result["actor_loss"] = actor_loss.item()
        result["approx_kl"] = approx_kl.item()
        result["clip_frac"] = clip_frac
        result["actor_loss_unclipped"] = pg_loss1.mean().item()
        result["stop_update"] = not update
        result["pg_magnitude"] = pg_magnitude.item()
        return result



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


    def offline_update(self, batch):
        # update critic
        critic_results = self.offline_update_critic(batch)
        actor_results = self.offline_update_actor(batch)
        return critic_results, actor_results
    def offline_update_critic(self, batch):
        with torch.no_grad():
            next_values = self.critic(batch["next_obs"], unnormalize = True)
            v_target = torch.Tensor(batch["rewards"]) + self.gamma * next_values

        values = self.critic(batch["obs"], unnormalize = True)
        critic_loss = self.critic_loss(values, v_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return {"critic_loss": critic_loss.item()}

    def offline_update_actor(self, batch):
        with torch.no_grad():
            actions = self.actor(batch["obs"])
            values = self.critic(batch["obs"], actions, unnormalize = True)
            q_values = self.critic(batch["obs"], batch["actions"], unnormalize = True)
            advantages = q_values - values
            weights = torch.clamp_max(torch.exp(advantages / self._awac_lambda), self._exp_adv_max)
        result = self.actor.log_prob(batch["obs"], batch["actions"], extra = True, do_sum = False)
        actor_loss = (-result.log_probs * weights).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return {"actor_loss": actor_loss.item()}


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
