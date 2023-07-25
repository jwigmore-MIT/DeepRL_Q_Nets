import torch.nn as nn
import torch
import numpy as np
from typing import Union
from safety.agents.normalizers import MovingNormalizer, CriticTargetScaler
from safety.agents.critics import Critic
import pickle

class PPOAgent:
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
                 gamma: float = 0.99,
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
                 critic_error_threshold: float = np.infty

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
        self.update_epochs = update_epochs
        self.minibatches = minibatches


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
        self.agent_buffer = agent_buffer()

        self.critic_error_threshold = critic_error_threshold







    def act(self, obs: np.ndarray, device: torch.device = None):
        # No need to check if state is safe
        # Still need to normalize the observation
        if self.obs_normalizer is not None:
            nn_obs = self.obs_normalizer.normalize(obs, update=True)
            action = self.actor.act(nn_obs, device)
            return action, nn_obs
        else:
            return self.actor.act(obs, device), obs



    def update(self, batch: dict):
        # Update the actor and critic networks using PPO algorithm
        results = {} # Dict to store results

        # get final next_nn_obs
        next_nn_obs = torch.Tensor(self.obs_normalizer.normalize(batch["next_obs"][-1].numpy(), update=False))

        for i in range(self.update_epochs):
            # Compute the
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
            np.random.shuffle(b_inds)
            minibatch_size = int(len(b_inds) // self.minibatches)

            results = {}
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
                mb_critic_results = self.update_critic_mb(mb_obs, mb_targets)
                self.avg_critic_error = mb_critic_results["avg_critic_error"]
                mb_actor_results = self.update_actor_mb(mb_obs, mb_actions, mb_log_probs, mb_advantages, mb_interventions)

                # process results for wandb logging
                for key, value in mb_critic_results.items():
                    if key in results:
                        results[key].append(value)
                    else:
                        results[key] = [value]
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
        critic_loss = self.compute_critic_loss(new_values, mb_targets)
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
                "avg_critic_error": critic_errors.abs().mean().item(),
                "avg_critic_true_error": true_errors.abs().mean().item(),
                "explained_variance": explained_variance.item(),
                "target_mean": target_mean,
                "target_std": target_std,
                "mb_targets": mb_targets.mean().item(),
                "mb_values": new_values.mean().item(),
                }


    def update_critic(self, batch: dict, b_targets: torch.Tensor):
        # Get current critic value estimates and compute error from targets based on this
        nn_values = self.get_nn_value(batch['n_obs'])
        critic_loss = self.compute_critic_loss(nn_values, b_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # === Logging === #
        # Compute errors
        critic_errors = (nn_values-b_targets)

        if self.target_scaler is not None:
            true_targets = self.target_scaler.unnormalize(b_targets)
            true_values = self.target_scaler.unnormalize(nn_values)
            true_errors = true_values - true_targets
            explained_variance = 1 - (torch.var(true_errors) / torch.var(true_targets))
        else:
            true_errors = critic_errors
            explained_variance = 1 - (torch.var(critic_errors) / torch.var(b_targets))

        max_critic_error = true_errors.max().item()
        min_critic_error = true_errors.min().item()
        if abs(max_critic_error) > abs(min_critic_error):
            max_critic_dev = max_critic_error
        else:
            max_critic_dev = min_critic_error
        return {"critic_loss": critic_loss.item(),
                "max_critic_dev: ": max_critic_dev,
                "avg_critic_error: ": critic_errors.abs().mean().item(),
                "avg_critic_true_error: ": true_errors.abs().mean().item(),
                "explained_variance": explained_variance.item(),
                "target_mean": self.target_scaler.target_mean,
                "target_std": self.target_scaler.target_std}



    def compute_critic_loss(self, nn_values, b_targets):
        loss = torch.nn.functional.mse_loss(nn_values, b_targets)
        return loss


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
                delta = rewards[t] + self.gamma * next_value  - values[t]
                adv[t] = last_gae_lam =  delta +  self.gamma * self.gae_lambda* last_gae_lam
        return adv

    # === Actor Methods === #

    def update_actor_mb(self, mb_obs, mb_actions, mb_log_probs, mb_advantages, mb_interventions):
        if self.avg_critic_error < self.critic_error_threshold:
            update = True
            bad_critic = False
        else:
            update = False
            bad_critic = True
        result = self.actor.log_prob(mb_obs, mb_actions, extra=True)
        self.agent_buffer.add_data(result)
        log_ratio = (result["log_probs"] - mb_log_probs)
        ratio = log_ratio.exp()

        with torch.no_grad():
            clip_frac = ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
        if self.kl_coef == 0:
            with torch.no_grad():
                approx_kl = -log_ratio.mean()
                #approx_kl = ((ratio - 1) - log_ratio).mean()
        else:
            approx_kl = -log_ratio.mean()
            #approx_kl = ((ratio - 1) - log_ratio).mean()
        # if approx_kl < 0:
        #     print("Negative KL detected")

        if self.kl_target is not None:
            if approx_kl.abs() > self.kl_target:
                # if the KL is too high, we stop updating the policy
                update = False


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
                     + self.int_coef * int_loss\
                     + kl_loss + entropy_loss
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
        result["imit_loss"] = imit_loss.mean().item()
        result["int_loss"] = int_loss.item()
        result["bad_critic"] = bad_critic


        return result



    def update_actor(self, batch: dict, log_probs_old: torch.Tensor):

        # flag of whether or not to end updates to the actor
        continue_updates = True
        results = {}
        with torch.no_grad():
            nn_values = self.get_nn_value(batch['n_obs'])
            b_values = self.unnormalize_value(nn_values)

            nn_next_value = self.get_nn_value(batch['next_n_obs'][-1])
            b_next_value = self.unnormalize_value(nn_next_value)

            # Compute advantages
            rewards = torch.Tensor(batch['rewards'])
            advantages = self.compute_GAE(rewards, b_values, b_next_value, batch['dones'])

        result = self.actor.log_prob(batch["obs"], batch["actions"], extra=True)
        log_ratio = (result["log_probs"] - log_probs_old)
        ratio = log_ratio.exp()

        with torch.no_grad():
            clip_frac = ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
        if self.kl_coef == 0:
            with torch.no_grad():
                approx_kl = -log_ratio.mean()
        else:
            approx_kl = -log_ratio.mean()
        if self.kl_target is not None:
            if approx_kl > self.kl_target:
                # if the KL is too high, we stop updating the policy
                update = False
                #result["stop_update"] = True


        # compute loss
        pg_loss1 = -advantages * ratio # unclipped loss
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) # clipped loss
        kl_loss = self.kl_coef * approx_kl
        entropy_loss = self.ent_coef * result["entropy"].mean() # fix this
        actor_loss = torch.max(pg_loss1, pg_loss2).mean() + kl_loss + entropy_loss
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
        result["advantages"] = advantages.mean().item()
        result["approx_kl"] = approx_kl.item()
        result["entropy_loss"] = entropy_loss.item()
        result["kl_loss"] = kl_loss.item()
        result["clip_frac"] = clip_frac
        result["actor_loss_unclipped"] = pg_loss1.mean().item()
        result["stop_update"] = not update
        result["pg_magnitude"] = pg_magnitude.item()
        if pg_magnitude.item() > 1e6:
            print("pg_magnitude is too high")
            Exception("pg_magnitude is too high")

        return result

    def get_log_prob(self, obs, action):
        return self.actor.log_prob(obs, action)

    def save_agent(self, save_path):
        # Save the actor, critic, target_scaler, obs_normalizer, agent_parameters, and the optimizer states
        pickle.dump(self.actor, open(save_path + "actor.pkl", "wb"))
        pickle.dump(self.critic, open(save_path + "critic.pkl", "wb"))
        pickle.dump(self.target_scaler, open(save_path + "target_scaler.pkl", "wb"))
        pickle.dump(self.obs_normalizer, open(save_path + "obs_normalizer.pkl", "wb"))
        pickle.dump(self.agent_parameters, open(save_path + "agent_parameters.pkl", "wb"))




class agent_buffer:

    def __init__(self):
        self._pointer = 0
        self.max_size = 1000000
        self.buffer = {}

    def add_data(self, dict_):
        "dict_ is a dictionary of Tensors"
        try:
            for key, value in dict_.items():
                # convert the data from a tensor to a numpy array
                data = value.detach().numpy()
                n_data = data.shape[0]
                if len(data.shape) == 1:
                    data = data.reshape((n_data, 1))

                l_data = data.shape[1]
                # if the key is not an attribute of the class, add it as an attribute
                if not hasattr(self, key):
                    init_data = np.zeros((self.max_size, l_data))
                    setattr(self, key, init_data)

                item = getattr(self, key)
                item[self._pointer: self._pointer + n_data] = data
            self._pointer += n_data
        except ValueError:
            print("ValueError")
            print(dict_)
            print(data)
            print(n_data)
            print(self._pointer)
            print(self.max_size)
            print(key)
            print(value)
            print(item)
            print(self.__dict__)


