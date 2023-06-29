import torch.nn as nn
import torch
import numpy as np
from typing import Union
from safety.agents.normalizers import Normalizer, CriticTargetScaler
from safety.agents.critics import Critic
import pickle
import torch.optim


class SACAgent:

    def __init__(self, actor,
                    actor_optim,
                    qf1,
                    qf1_target,
                    qf2,
                    qf2_target,
                    q_optim,
                    obs_normalizer,
                    alpha = 0.2,
                    alpha_auto_tune = False,
                    alpha_lr = 3e-4,
                    gamma = 0.95,
                    policy_update_freq = 2,
                    target_update_freq = 1,
                    noise_clip = 0.5,
                    tau = .005

                 ):
        self.actor = actor
        self.actor_optim = actor_optim
        self.qf1 = qf1
        self.qf1_target = qf1_target
        self.qf2 = qf2
        self.qf2_target = qf2_target
        self.q_optim = q_optim
        self.obs_normalizer = obs_normalizer
        if alpha_auto_tune:
            self.alpha_tuner = AlphaTuner(actor.action_dim, alpha_lr)
            self.alpha = self.alpha_tuner.alpha
        else:
            self.alpha_tuner = None
            self.alpha = alpha
        self.gamma = gamma
        self.policy_update_freq = policy_update_freq
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        self.noise_clip = noise_clip
        self.tau = tau



    def act(self, obs: np.ndarray, device: torch.device = None):
        if self.obs_normalizer is not None:
            nn_obs = self.obs_normalizer.normalize(obs, update=True)
            action = self.actor.act(obs, device)
            return action, nn_obs
        else:
            return self.actor.act(obs, device), obs

    def update(self, batch: dict):
        self.update_counter +=1
        results= {}

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(batch['next_nn_obs'])
            qf1_next_target = self.qf1_target(batch['next_nn_obs'], next_state_actions)
            qf2_next_target = self.qf2_target(batch['next_nn_obs'], next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            # one-step TD estimate of Q val
            next_q_value = batch["rewards"] + (1 - batch["dones"]) * self.gamma * min_qf_next_target.view(-1)

        qf1_a_values = self.qf1(batch['nn_obs'], batch['actions']).view(-1)
        qf2_a_values = self.qf2(batch['nn_obs'], batch['actions']).view(-1)

        qf1_loss = torch.nn.functional.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = torch.nn.functional.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optim.zero_grad()
        qf_loss.backward()
        self.q_optim.step()

        if self.update_counter % self.policy_update_freq == 0:
            for _ in range(self.policy_update_freq):
                actions, log_pi, means = self.actor.get_action(batch['nn_obs'])
                qf1_pi = self.qf1(batch['nn_obs'], actions)
                qf2_pi = self.qf2(batch['nn_obs'], actions)
                min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                self.actor_optim.zero_grad()
                policy_loss.backward()
                self.actor_optim.step()

                results["policy_loss"] = policy_loss.item()
                results["log_pi"] = log_pi.mean().item()

                if self.alpha_tuner is not None:
                    with torch.no_grad():
                        _, log_pi, _ = self.actor.get_action(batch['nn_obs'])
                    self.alpha = self.alpha_tuner.update_alpha(log_pi)

        if self.update_counter % self.target_update_freq == 0:
            self.soft_update(self.qf1_target, self.qf1)
            self.soft_update(self.qf2_target, self.qf2)

    def soft_update(self, target: nn.Module, source: nn.Module):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * source_param.data)


class AlphaTuner:
    def __init__(self, action_dim, q_lr):
        self.target_entropy = -torch.prod(torch.Tensor(action_dim)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp().item()
        self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=q_lr)

    def update_alpha(self, log_pi):

        alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy)).mean()
        self.a_optimizer.zero_grad()
        alpha_loss.backward()
        self.a_optimizer.step()
        self.alpha = self.log_alpha.exp().item()
        return self.alpha

