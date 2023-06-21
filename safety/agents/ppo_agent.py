import torch.nn as nn
import torch
import numpy as np
from typing import Union

class PPOAgent:
    "Simple PPO Agent without intervention"

    def __init__(self,
                 actor: nn.Module,
                 critic: nn.Module,
                 actor_optimizer: torch.optim.Optimizer,
                 critic_optimizer: torch.optim.Optimizer,
                 interventioner: nn.Module,
                 normalize_values: bool = False,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 ppo: bool = False,
                 ppo_clip_coef: float = 0.25,
                 kl_coef: float = 0.0,  # Beta in PPO paper
                 entropy_coef: float = 0.0,
                 kl_target: float = None,
                 intervention_penalty: float = 0.0,
                 grad_clip: float = None,
                 value_clip: float = 1.0,
                 updates_per_rollout: int = 1,
                 awac_lambda: float = 1.0,
                 exp_adv_max: float = 100,
                 target_update_rate: float = 0.2,

                 pretrain: bool = False,
                 normalized_states: Union[bool, str] = False,  # False, "gym"

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
        self.target_update_rate = target_update_rate

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.ppo = ppo
        self.ppo_clip_coef = ppo_clip_coef
        self.value_clip = value_clip
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        self.kl_target = kl_target
        self.grad_clip = grad_clip

        self._awac_lambda = awac_lambda
        self._exp_adv_max = exp_adv_max

        self.intervention_penalty = intervention_penalty

        self.normalized_states = normalized_states

        self.pretrain = pretrain