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

    def __init__(self, state_dim, hidden_dim, action_ranges: np.ndarray):
        super().__init__()

        self.network = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU()
        )
        self.action_ranges = action_ranges
        self.nvec = action_ranges[1,:] + 1
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







class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_value = self._mlp(torch.cat([state, action], dim=-1))
        return q_value

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)



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


class ProbabilisticInterventioner(nn.Module):
    """
    Allow for some probability of intervention in unsafe states near the boundary
    """
    def __init__(self, safe_actor, trigger_state = 0, omega = 1.0):
        super().__init__()
        self.safe_actor = safe_actor
        self.trigger_state = trigger_state
        self.omega = omega

    def check_safety(self, state):
        # In unsafe state, return False, otherwise True
        gap = self.trigger_state - np.sum(state)
        prob = min(1,np.exp(-self.omega * gap))  # probability of intervention
        if np.random.rand() < prob:
            return (False, prob)
        else:
            return (True, prob)


    def act(self, state, device = None):
        return self.safe_actor.act(state)



class SafeAWACAgent:
    """
    Combines (AWAC) Actor, (AWAC) Critic, and Intervention Policy
    """
    def __init__(
            self,
            # NN Settings
            actor: nn.Module,
            critic1: nn.Module,
            critic2: nn.Module,
            actor_optimizer: torch.optim.Optimizer,
            critic1_optimizer: torch.optim.Optimizer,
            critic2_optimizer: torch.optim.Optimizer,
            # Interventioner
            interventioner: nn.Module,
            # Updates
            pretrain: bool = False,
            updates_per_rollout: int = 1,
            grad_clip: float = None,
            # Normalization and discounting
            normalize_values: bool = False,
            normalized_states: Union[bool, str] = False,  # False, "gym"
            target_update_rate: float = 0.2,
            gamma: float = 0.99,
            # Critic Settings
            critic_tau: float = 0.005,
            value_clip: float = 1.0,
            # Actor settings
            ppo: bool = False,
            gae_lambda: float = 0.95,
            ppo_clip_coef: float = 0.25,
            kl_coef: float = 0.0, # Beta in PPO paper
            entropy_coef: float = 0.0,
            kl_target: float = None,
            intervention_penalty: float = 0.0,
            # AWAC Settings
            awac_lambda: float = 1.0,
            exp_adv_max: float = 100,




    ):
        # NN Settings
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self._critic1 = critic1
        self._critic2 = critic2
        self.critic1_optimizer = critic1_optimizer
        self.critic2_optimizer = critic2_optimizer
        # Interventioner
        self.interventioner = interventioner

        # Updates
        self.updates_per_rollout = updates_per_rollout
        self.normalize_values = normalize_values
        self.value_mean = None
        self.value_std = None
        self.target_update_rate = target_update_rate

        # Normalization and discounting
        self.gamma = gamma
        self.normalized_states = normalized_states
        self.pretrain = pretrain

        # Actor Updates
        self.ppo = ppo
        self.gae_lambda = gae_lambda
        self.ppo_clip_coef = ppo_clip_coef
        self.value_clip = value_clip
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        self.kl_target = kl_target
        self.grad_clip = grad_clip

        # AWAC Settings
        self._awac_lambda = awac_lambda
        self._exp_adv_max = exp_adv_max

        self.intervention_penalty = intervention_penalty


    def critic(self, state: torch.Tensor, action: torch.Tensor, unnormalize = False) -> torch.Tensor:
        # If
        value = torch.min(self._critic1(state, action), self._critic2(state, action))
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
            prob = 1
        else:
            safety_check = self.interventioner.check_safety(true_state)
            if isinstance(safety_check, tuple):
                is_safe, prob = safety_check
            else:
                is_safe = safety_check
                prob = 1- is_safe
            if is_safe:
                action =  self.actor.act(state, device)
                intervention = False
            else:
                action =  self.interventioner.act(true_state)
                intervention = True

        return action, intervention, prob

    def update(self, batch):

        results = {}
        # if live training get the log_prob of the batch

        with torch.no_grad():
            curr_log_prob = self.actor.log_prob(batch["obs"], batch["actions"])

        updates_per_rollout = self.updates_per_rollout
        for i in range(updates_per_rollout):


            critic_results = self.update_critic(batch)
            actor_results = self.update_actor(batch, curr_log_prob)


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
        # Compute critic loss
        results = self._critic_loss(batch)
        critic_loss = results["critic_loss"]
        # Update critic
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self._critic1.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self._critic2.parameters(), self.grad_clip)
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        return results

    def _critic_loss(self, batch):
        with torch.no_grad():
            next_actions, _ = self.actor(batch["next_obs"])

            q_next = self.critic(batch["next_obs"], next_actions)
            q_target = torch.Tensor(batch["rewards"]) + self.gamma  * q_next


        q1 = self._critic1(batch["obs"], batch["actions"])
        q2 = self._critic2(batch["obs"], batch["actions"])

        q1_loss = nn.functional.mse_loss(q1, q_target)
        q2_loss = nn.functional.mse_loss(q2, q_target)
        results = {}
        if not self.pretrain:
            with torch.no_grad():
                critic_error = (self.critic(batch["obs"], batch["actions"]) - q_target).abs()
                results["explained_variance"] = (1 - (critic_error.var() / q_target.var())).item()
                results["critic_error"] = critic_error.mean().item()

        loss = q1_loss + q2_loss
        results["critic_loss"] = loss
        return results

    def _actor_loss(self, batch, curr_log_prob):
        with torch.no_grad():
            pi_action, _ = self.actor(batch["obs"])
            v = self.critic(batch["obs"], pi_action)

            q = self.critic(batch["obs"], batch["actions"])
            adv = q - v
            weights = torch.clamp_max(
                torch.exp(adv / self._awac_lambda), self._exp_adv_max
            )
            unclamped_weights = torch.exp(adv / self._awac_lambda)
            next_pi_action, _ = self.actor(batch["next_obs"])
            next_v = self.critic(batch["next_obs"], next_pi_action)
            adv_gae = compute_GAE(torch.Tensor(batch["rewards"]), v, next_v, batch["dones"], gamma = self.gamma, gae_lambda = self.gae_lambda)
            adv_error = (adv_gae - adv)


        actor_results = self.actor.log_prob(batch["obs"], batch["actions"], extra = True)
        action_log_prob = actor_results["log_probs"]
        log_ratio = action_log_prob - curr_log_prob
        ratio = log_ratio.exp()
        with torch.no_grad():
            approx_kl = -log_ratio.mean()
        if True:
            loss = (-action_log_prob * weights).mean()
        else: # PPO Loss
            with torch.no_grad():
                clip_frac = ((ratio - 1.0).abs() > self.ppo_clip_coef).float().mean().item()
            pg_loss1 = -adv_gae * ratio
            pg_loss2 = -adv_gae * torch.clamp(ratio, 1.0 - self.ppo_clip_coef, 1.0 + self.ppo_clip_coef)
            loss = torch.max(pg_loss1, pg_loss2).mean()
        results = {
            "actor_loss": loss,
            "log_prob": action_log_prob.mean(),
            "approx_kl": approx_kl.item(),
            "entropy": actor_results["entropy"].mean().item(),
            "weights": weights.mean().item(),
            "unclamped_weights": unclamped_weights.mean().item(),
        }

        results["adv_error_mean"] = adv_error.mean().item()
        results["adv_error_std"] = adv_error.std().item()

        return results

    def update_actor(self, batch, curr_log_prob):
        # Compute actor loss
        results = self._actor_loss(batch, curr_log_prob)
        loss = results["actor_loss"]
        self.actor_optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()
        pg_magnitude = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0) for p in list(self.actor.parameters())]), 2.0)
        results["pg_magnitude"] = pg_magnitude.item()
        return results
    # Old


    def set_safe_threshold(self, threshold):
        self.interventioner.trigger_state = threshold

    # Saving and Loading
    def state_dict(self) -> Dict[str, Any]:
        # overloading this method to save interventioner as well
        return {
            "actor": self.actor.state_dict(),
            "critic1": self._critic1.state_dict(),
            "critic2": self._critic2.state_dict(),
            #"interventioner": self.interventioner
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        # also loads the interventioner
        self.actor.load_state_dict(state_dict["actor"])
        self._critic1.load_state_dict(state_dict["critic1"])
        self._critic2.load_state_dict(state_dict["critic2"])
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




def compute_GAE(rewards, values, next_values, dones, gamma = 0.99, gae_lambda = 0.95):
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
            adv[t] = last_gae_lambda = delta + gamma * gae_lambda * last_gae_lambda


    return adv


def GAE_test():
    rewards = torch.Tensor([1,2,3])
    values = torch.Tensor([4,5,6])
    next_values = torch.Tensor([5,6,7])
    dones = torch.Tensor([0,0,0])
    adv = compute_GAE(rewards, values, next_values, dones, gamma = 1.0, gae_lambda = 1.0)
    return adv


if __name__ == "__main__":
    adv = GAE_test()
