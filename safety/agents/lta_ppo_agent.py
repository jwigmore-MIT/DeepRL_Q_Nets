import torch.nn as nn
import torch
import numpy as np
from typing import Union
from safety.agents.normalizers import MovingNormalizer, CriticTargetScaler, FakeTargetScaler
from safety.agents.critics import Critic
import pickle
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class LTAPPOAgent:
    "Simple PPO Agent without intervention"

    def __init__(self,
                 actor: nn.Module,
                 critic: nn.Module,
                 actor_optimizer: torch.optim.Optimizer,
                 critic_optimizer: torch.optim.Optimizer,
                 obs_normalizer: MovingNormalizer = None,
                 target_scaler: CriticTargetScaler = None,
                 update_epochs: int = 10,
                 minibatches: int = 1,
                 shuffle_mb: bool = True,
                 recompute_adv: bool = True,
                 critic_first: bool = True,
                 alpha: float = 0.1,
                 nu: float = 1.0,
                 gamma: float = 1.0,
                 gae_lambda: float = 0.95,
                 clip_coef: float = 0.25,
                 kl_coef: float = 0.0,  # Beta in PPO paper
                 ent_coef: float = 0.0,
                 kl_target: float = None,
                 grad_clip: float = None,
                 value_clip: float = 1.0,
                 vf_coef: float = 0.5,
                 imit_coef: float = 0.0,
                 pg_coef: float = 1.0,
                 int_coef: float = 0.0,
                 norm_adv: bool = True,
                 pretrain_minibatches = 0,
                 pretrain_epochs = 0,
                 omega_norm: bool = False,
                 clip_adv = False,

                 ):
            # Actor
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.critic = critic
        self.critic_optimizer = critic_optimizer

        # Normalizers/Scalers
        self.obs_normalizer = obs_normalizer
        self.target_scaler = target_scaler


        # Update step parameters
        self.critic_first = critic_first
        self.recompute_adv = recompute_adv
        self.update_epochs = update_epochs
        self.minibatches = minibatches
        self.shuffle_mb = shuffle_mb
        self.norm_adv = norm_adv

        # For value/advantage estimation
        self.alpha = alpha # LTA estimate update rate
        self.nu = nu # Average Value Constraint Coefficient
        self.eta = None # LTA estimate
        self.omega = 1 # LTA variance estimate
        self.omega_norm = omega_norm # LTA variance estimate normalization
        self.b = 0
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.clip_coef = clip_coef

        self.kl_coef = kl_coef
        self.ent_coef = ent_coef
        self.kl_target = kl_target
        self.grad_clip = grad_clip
        self.value_clip = value_clip
        self.vf_coef = vf_coef
        self.imit_coef = imit_coef
        self.pg_coef  = pg_coef
        self.int_coef = int_coef

        self.pretrain_minibatches = pretrain_minibatches
        self.pretrain_epochs = pretrain_epochs





    def act(self, obs: np.ndarray, device: torch.device = None):
        # No need to check if state is safe
        # Still need to normalize the observation
        if self.obs_normalizer is not None:
            nn_obs = self.obs_normalizer.normalize(obs, update=True)
            action = self.actor.act(nn_obs, device)
            return action, nn_obs
        else:
            return self.actor.act(obs, device), obs

        # update the eta estimate
        self.update_b_eta(batch["rewards"], batch["nn_obs"])


    def update(self, batch: dict, pretrain = False):
        # Update the actor and critic networks using PPO algorithm
        results = {} # Dict to store results

        # get final next_nn_obs
        next_nn_obs = torch.Tensor(self.obs_normalizer.normalize(batch["next_obs"][-1].numpy(), update=False))

        # update the eta estimate
        self.update_b_eta(batch["rewards"], batch["nn_obs"])

        if not self.recompute_adv:
            with torch.no_grad():
                # modifies the rewards to include the interventions
                b_rewards = batch['rewards'] - self.int_coef * batch['interventions'] # rewards + interventions
                b_obs = batch['nn_obs'] # observations fed into nn
                b_log_probs = self.actor.log_prob(b_obs, batch['actions']) # log probs of actions taken in the batch
                b_actions = batch['actions'] # actions takin in batch
                #b_true_values = self.get_true_value(b_obs) # true values based on the current critic and target scaler // DONT NEED
                b_targets, b_advantages = self.compute_targets(b_rewards, batch['nn_obs'], next_nn_obs, batch['dones'])
                # targets should be normalized, advantages should not be normalized

        update_epochs = self.update_epochs if not pretrain else self.pretrain_epochs
        results = {}
        for i in range(update_epochs):
            # Compute the

            if self.recompute_adv:
                with torch.no_grad():
                    # modifies the rewards to include the interventions
                    b_rewards = batch['rewards'] - self.int_coef * batch['interventions'] # rewards + interventions
                    b_obs = batch['nn_obs'] # observations fed into nn
                    b_log_probs = self.actor.log_prob(b_obs, batch['actions']) # log probs of actions taken in the batch
                    b_actions = batch['actions'] # actions takin in batch
                    #b_true_values = self.get_true_value(b_obs) # true values based on the current critic and target scaler // DONT NEED
                    b_targets, b_advantages = self.compute_targets(b_rewards, batch['nn_obs'], next_nn_obs, batch['dones'])
                    # targets should be normalized, advantages should not be normalized

            # get indices
            b_inds = np.arange(len(batch['nn_obs']))
            if self.shuffle_mb:
                np.random.shuffle(b_inds)
            if not pretrain:
                minibatch_size = int(len(b_inds) // self.minibatches)
            else:
                minibatch_size = int(len(b_inds) // self.pretrain_minibatches)

            if pretrain:
                self.update_b_eta(batch["rewards"], batch["nn_obs"])

            for start in range(0, len(b_inds), minibatch_size):
                end = start+ minibatch_size
                mb_inds = b_inds[start:end]
                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_log_probs = b_log_probs[mb_inds]
                mb_targets = b_targets[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_interventions= batch['interventions'][mb_inds]
                # mb_true_values = b_true_values[mb_inds] # DONT NEED

                # update the critic
                if self.critic_first:
                    mb_critic_results = self.update_critic_mb(mb_obs, mb_targets)
                    if not pretrain:
                        mb_actor_results = self.update_actor_mb(mb_obs, mb_actions, mb_log_probs, mb_advantages, mb_interventions)
                else:
                    if not pretrain:
                        mb_actor_results = self.update_actor_mb(mb_obs, mb_actions, mb_log_probs, mb_advantages, mb_interventions)
                    mb_critic_results = self.update_critic_mb(mb_obs, mb_targets)


                # process results for wandb logging
                for key, value in mb_critic_results.items():
                    if key in results:
                        results[key].append(value)
                    else:
                        results[key] = [value]
                if not pretrain:
                    for key, value in mb_actor_results.items():
                        if key in results:
                            results[key].append(value)
                        else:
                            results[key] = [value]
        return results

    # === Critic Methods === #

    def get_true_value(self, nn_obs):
        if self.target_scaler is not None and self.target_scaler.target_mean is not None:
            nn_value = self.critic(nn_obs)
            value = self.target_scaler.scale(nn_value)
        else:
            value = self.critic(nn_obs)
        return value
    def get_nn_value(self, nn_obs: torch.Tensor):
        """
        Gets the value of the nn_obs from the internal critic
        """
        return self.critic(nn_obs)  #


    def unnormalize_value(self, nn_value: torch.Tensor):
        value = nn_value  # change this later
        return value

    def compute_targets(self, rewards, nn_obs, next_nn_obs, dones):
        """
        Returns target (normalized if target_scaler is not None) and advantages (not normalized in anyway)

        rewards: vector of rewards
        nn_obs: vector of nn observations
        next_nn_obs: final next nn obs
        dones: vector of dones
        """
        with torch.no_grad():
            # Get values of observations
            b_values = self.get_true_value(nn_obs)

            # get values of next observations
            b_next_value = self.get_true_value(next_nn_obs)

            rewards = torch.Tensor(rewards)

            # Compute advantages
            b_advantages = self.compute_GAE(rewards, b_values, b_next_value, dones)

            # Compute the "true" targets
            b_targets = b_advantages + b_values


            # Normalize the target
            if self.target_scaler is not None:
                self.target_scaler.update(b_targets)
                b_targets = self.target_scaler.normalize(b_targets)
        return b_targets, b_advantages

    def update_critic_mb(self, mb_obs, mb_targets):
        new_values = self.get_nn_value(mb_obs)
        new_values = new_values
        bias_factor = self.b*self.nu
        critic_loss = self.compute_critic_loss(new_values, mb_targets, bias_factor)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # === Logging === #
        # Compute errors
        critic_errors = (new_values - mb_targets)

        if self.target_scaler is not None:
            true_targets = self.target_scaler.scale(mb_targets)
            true_values = self.target_scaler.scale(new_values)
            true_errors = true_values - true_targets
            explained_variance = 1 - (torch.var(true_errors) / torch.var(true_targets))
        else:
            true_errors = critic_errors
            explained_variance = 1 - (torch.var(critic_errors) / torch.var(mb_targets))

        max_critic_error = true_errors.max().item()
        min_critic_error = true_errors.min().item()
        if abs(max_critic_error) > abs(min_critic_error):
            max_critic_dev = max_critic_error
        else:
            max_critic_dev = min_critic_error

        if self.target_scaler is not None:
            target_mean = self.target_scaler.target_mean
            target_std = self.target_scaler.target_std
        else:
            target_mean = None
            target_std = None
        return {"critic_loss": critic_loss.item(),
                "max_critic_dev: ": max_critic_dev,
                "avg_critic_error: ": critic_errors.abs().mean().item(),
                "avg_critic_true_error: ": true_errors.abs().mean().item(),
                "explained_variance": explained_variance.item(),
                "target_mean": target_mean,
                "target_std": target_std,
                "eta": self.eta,
                "omega": self.omega,
                "mb_targets": mb_targets.mean().item(),
                "mb_values": new_values.mean().item(),
                "bias_factor": bias_factor,
                "b": self.b}


    # def update_critic(self, batch: dict, b_targets: torch.Tensor):
    #     # Get current critic value estimates and compute error from targets based on this
    #     nn_values = self.get_nn_value(batch['n_obs'])
    #     critic_loss = self.compute_critic_loss(nn_values, b_targets)
    #     self.critic_optimizer.zero_grad()
    #     critic_loss.backward()
    #     self.critic_optimizer.step()
    #
    #     # === Logging === #
    #     # Compute errors
    #     critic_errors = (nn_values-b_targets)
    #
    #     if self.target_scaler is not None:
    #         true_targets = self.target_scaler.unnormalize(b_targets)
    #         true_values = self.target_scaler.unnormalize(nn_values)
    #         true_errors = true_values - true_targets
    #         explained_variance = 1 - (torch.var(true_errors) / torch.var(true_targets))
    #     else:
    #         true_errors = critic_errors
    #         explained_variance = 1 - (torch.var(critic_errors) / torch.var(b_targets))
    #
    #     max_critic_error = true_errors.max().item()
    #     min_critic_error = true_errors.min().item()
    #     if abs(max_critic_error) > abs(min_critic_error):
    #         max_critic_dev = max_critic_error
    #     else:
    #         max_critic_dev = min_critic_error
    #     return {"critic_loss": critic_loss.item(),
    #             "max_critic_dev: ": max_critic_dev,
    #             "avg_critic_error: ": critic_errors.abs().mean().item(),
    #             "avg_critic_true_error: ": true_errors.abs().mean().item(),
    #             "explained_variance": explained_variance.item(),
    #             "target_mean": self.target_scaler.target_mean,
    #             "target_std": self.target_scaler.target_std,
    #             "nn_values": nn_values.mean().item()}




    def compute_critic_loss(self, nn_values, b_targets, bias_factor):
        if self.clip_vloss:
            # clip the value loss
            unclipped_loss = 0.5 * (b_targets - bias_factor - nn_values).pow(2)
            clipped_diff = torch.clamp(b_targets - bias_factor - nn_values)
        #loss = torch.nn.functional.mse_loss(nn_values , b_targets)
        else:
            mean_loss = 0.5 * (b_targets - bias_factor - nn_values).pow(2).mean()
        return mean_loss


    def compute_GAE(self, rewards, values, next_val, dones = None):
        # Need to fix this and make sure there are no dones...
        with torch.no_grad():

            adv = torch.zeros_like(rewards)
            last_gae_lam = 0
            for t in reversed(range(len(adv))):
                if t == len(adv) - 1:
                    next_value = next_val
                else:
                    next_value = values[t + 1]
                delta = (rewards[t] - self.eta) / self.omega + self.gamma * next_value - values[t]
                #self.omega = 8
                adv[t] = last_gae_lam =  delta +  self.gamma * self.gae_lambda* last_gae_lam
        # if self.clip_adv:
        #     adv = torch.clamp(adv, -self.clip_adv, self.clip_adv)
        return adv



    def update_b_eta(self, rewards, nn_obs):
        # updating eta

        if self.alpha is None:
            self.eta = 0
            self.beta=0
            self.omega = 1
            return
        elif self.eta is None:
            self.eta = rewards.mean().item()
        else:
            self.eta = self.eta * (1-self.alpha) + rewards.mean().item() * (self.alpha)

        if self.omega is None and self.omega_norm:
            self.omega = (rewards-self.eta).std().item()
        elif self.omega_norm:
            #self.omega = self.omega * (1-self.alpha) + (rewards-self.eta).std().item() * (self.alpha)
            self.omega = (rewards).std().item()
        with torch.no_grad():
            # updating b
            values = self.get_true_value(nn_obs)  # nn_obs = batch["nn_obs"]

            self.b = self.b * (1 - self.alpha) + values.mean() * self.alpha




    # === Actor Methods === #

    def update_actor_mb(self, mb_obs, mb_actions, mb_log_probs, mb_advantages, mb_interventions):
        update = True
        result = self.actor.log_prob(mb_obs, mb_actions, extra=True)
        log_ratio = (result["log_probs"] - mb_log_probs)
        ratio = log_ratio.exp()

        with torch.no_grad():
            clip_frac = ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
        if self.kl_coef == 0:
            with torch.no_grad():
                approx_kl = -log_ratio.mean()
        else:
            approx_kl = -log_ratio.mean()
        # if approx_kl < 0:
        #     print("Negative KL detected")

        if self.kl_target is not None:
            if approx_kl.abs() > self.kl_target:
                # if the KL is too high, we stop updating the policy
                update = False
        mb_adv_mean = mb_advantages.mean().item()
        mb_adv_std = mb_advantages.std().item()
        if self.norm_adv:

            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)



        # policy loss
        # pg_loss1 = -mb_advantages  * ratio  # * (1-mb_interventions)# unclipped loss
        # pg_loss2 = -mb_advantages  * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) #* (1-mb_interventions) # clipped loss
        pg_loss1 = -mb_advantages * ratio * (1-mb_interventions) # unclipped loss
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * (1-mb_interventions) # clipped loss
        if self.actor._get_name() == "JSQDiscreteActor":
            mlp_actions = mb_actions[:,: mb_actions.shape[1]//2]
            imit_loss = ((result["actor_means"] - mlp_actions) * mb_interventions).pow(2)
        else:
            imit_loss = ((result["actor_means"] - mb_actions) * mb_interventions).pow(2)
        int_loss = mb_interventions.sum()

        kl_loss = self.kl_coef * approx_kl
        entropy_loss = -self.ent_coef * result["entropy"].mean()  # fix this
        actor_loss = self.pg_coef*torch.max(pg_loss1, pg_loss2).mean()  \
                     + self.imit_coef * imit_loss.mean()\
                     + kl_loss + entropy_loss #\
                        # + self.int_coef * int_loss # We might not actually want to do this because then we are double counting the loss due to interventions as they are included in the GAE estimation
        if actor_loss.abs().mean().item() > 1e4:
            print("actor_loss is too high")
            Exception("actor_loss is too high")
        if update:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_optimizer.step()
            pg_magnitude = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2.0) for p in list(self.actor.parameters())]), 2.0)

            # pg_variance = torch.var(
            #     torch.stack([p.grad.data.view(-1) for p in list(self.actor.parameters())]))
            # result["stop_update"] = False

        else:
            pg_magnitude = torch.Tensor([0])
        for key in result.keys():
            result[key] = result[key].mean().item()
        result["actor_loss"] = actor_loss.item()
        result["advantages"] = mb_advantages.mean().item()
        result["approx_kl"] = approx_kl.item()
        result["entropy_loss"] = entropy_loss.item()
        result["kl_loss"] = kl_loss.item()
        result["clip_frac"] = clip_frac
        result["actor_loss_unclipped"] = pg_loss1.mean().item()
        result["stop_update"] = float(not update)
        result["pg_magnitude"] = pg_magnitude.item()
        #result["pg_variance"] = pg_variance.item()
        result["imit_loss"] = self.imit_coef * imit_loss.mean().item()
        result["int_loss"] = self.int_coef * int_loss.item()
        result["mb_adv_std"] = mb_adv_std
        result["mb_adv_mean"] = mb_adv_mean


        return result



    # def update_actor(self, batch: dict, log_probs_old: torch.Tensor):
    #
    #     # flag of whether or not to end updates to the actor
    #     continue_updates = True
    #     results = {}
    #     with torch.no_grad():
    #         nn_values = self.get_nn_value(batch['n_obs'])
    #         b_values = self.unnormalize_value(nn_values)
    #
    #         nn_next_value = self.get_nn_value(batch['next_n_obs'][-1])
    #         b_next_value = self.unnormalize_value(nn_next_value)
    #
    #         # Compute advantages
    #         rewards = torch.Tensor(batch['rewards'])
    #         advantages = self.compute_GAE(rewards, b_values, b_next_value, batch['dones'])
    #
    #     result = self.actor.log_prob(batch["obs"], batch["actions"], extra=True)
    #     log_ratio = (result["log_probs"] - log_probs_old)
    #     ratio = log_ratio.exp()
    #
    #     with torch.no_grad():
    #         clip_frac = ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
    #     if self.kl_coef == 0:
    #         with torch.no_grad():
    #             approx_kl = -log_ratio.mean()
    #     else:
    #         approx_kl = -log_ratio.mean()
    #     if self.kl_target is not None:
    #         if approx_kl > self.kl_target:
    #             # if the KL is too high, we stop updating the policy
    #             update = False
    #             #result["stop_update"] = True
    #
    #
    #     # compute loss
    #     pg_loss1 = -advantages * ratio # unclipped loss
    #     pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) # clipped loss
    #     kl_loss = self.kl_coef * approx_kl
    #     entropy_loss = self.ent_coef * result["entropy"].mean() # fix this
    #     actor_loss = torch.max(pg_loss1, pg_loss2).mean() + kl_loss + entropy_loss
    #     if update:
    #         self.actor_optimizer.zero_grad()
    #         actor_loss.backward()
    #         if self.grad_clip is not None:
    #             torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
    #         self.actor_optimizer.step()
    #         pg_magnitude = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0) for p in list(self.actor.parameters())]), 2.0)
    #
    #         #result["stop_update"] = False
    #
    #     else:
    #         pg_magnitude = torch.Tensor([0])
    #     for key in result.keys():
    #         result[key] = result[key].mean().item()
    #     result["actor_loss"] = actor_loss.item()
    #     result["advantages"] = advantages.mean().item()
    #     result["approx_kl"] = approx_kl.item()
    #     result["entropy_loss"] = entropy_loss.item()
    #     result["kl_loss"] = kl_loss.item()
    #     result["clip_frac"] = clip_frac
    #     result["actor_loss_unclipped"] = pg_loss1.mean().item()
    #     result["stop_update"] = not update
    #     result["pg_magnitude"] = pg_magnitude.item()
    #     if pg_magnitude.item() > 1e6:
    #         print("pg_magnitude is too high")
    #         Exception("pg_magnitude is too high")
    #
    #     return result

    def get_log_prob(self, obs, action):
        return self.actor.log_prob(obs, action)

    def save_agent(self, save_path):
        # Save the actor, critic, target_scaler, obs_normalizer, agent_parameters, and the optimizer states
        pickle.dump(self.actor, open(save_path + "actor.pkl", "wb"))
        pickle.dump(self.critic, open(save_path + "critic.pkl", "wb"))
        pickle.dump(self.target_scaler, open(save_path + "target_scaler.pkl", "wb"))
        pickle.dump(self.obs_normalizer, open(save_path + "obs_normalizer.pkl", "wb"))
        pickle.dump(self.agent_parameters, open(save_path + "agent_parameters.pkl", "wb"))



