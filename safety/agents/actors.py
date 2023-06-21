import torch.nn as nn
import torch
from utils import layer_init
import numpy as np
from typing import List, Tuple

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


