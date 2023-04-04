# Import libraries
import os
import time
import wandb
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import logging
logging.getLogger().setLevel(logging.INFO)

tqdm_config = {
    "dynamic_ncols": True,
    "ascii": True,
}



# Import custom methods
from environment_init import make_MCMH_env
from agent_init import Agent
from testers import agent_test

class StopTrainingOnNoImprovement:

    def __init__(self, num_updates, check_n):
        self.score_vector = np.zeros(num_updates)
        self.index = 0
        self.check_n = check_n
        self.no_improvement_steps = 0
        self.stopped = False

    def update(self, score):
        self.score_vector[self.index] = score
        if self.index < self.check_n:
            self.index += 1
            return False

        if self.score_vector[self.index] == self.score_vector[self.index-1]:
            self.no_improvement_steps += 1
        else:
            self.no_improvement_steps = 0

        if self.no_improvement_steps >= self.check_n:
            print("!"*30)
            print(f"No Improvement in last {self.self.no_improvement_steps} updates")
            print(f"Ending training early (after {self.index+1} updates)")
            self.stopped = True
            return True
        else:
            return False




def load_agent(agent, artifact):
    model_weight_dir = artifact.download()
    model_dict = {}
    for x in os.listdir(model_weight_dir):
        if x.endswith('.pt'):
            model_dict = torch.load(os.path.join(model_weight_dir, x))
    agent.load_state_dict(state_dict=model_dict)

def train_agent(env_para, train_args, test_args, run, checkpoint_saver, artifact = None, sweep = False):



    ## Setup tensorboard
    writer = SummaryWriter(run.dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(train_args).items()])),
        )

    ## Set the seed of random, np.random, and torch
    # TRY NOT TO MODIFY: seeding
    random.seed(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)
    torch.backends.cudnn.deterministic = train_args.torch_deterministic

    ## Select the device
    device = torch.device("cuda" if torch.cuda.is_available() and train_args.cuda else "cpu")

    # Initialize the environments
    envs = gym.vector.SyncVectorEnv(
        [make_MCMH_env(env_para,
                       max_steps = train_args.num_steps_per_reset,
                       time_scaled = train_args.time_scaled,
                       moving_average= train_args.moving_average,
                       no_state_penalty= train_args.no_state_penalty,
                       min_reward = train_args.min_reward,
                       delivered_rewards = train_args.delivered_rewards,
                       horizon_scaled = train_args.horizon_scaled,
                       ) for i in range(train_args.num_envs)]
    )

    # Initialize agents and pass agents (nn.module) to device
    agent = Agent(envs).to(device)

    if artifact is not None:
        load_agent(agent, artifact)


    # Watch the gradient
    wandb.watch(agent, log_freq = 100)

    ## Initialize ADAM optimizer
    #optimizer = optim.Adam(agent.parameters(), lr=train_args.actor_learning_rate, eps=1e-5)
    optimizer = optim.Adam([
        {'params': agent.actor_logstd, 'lr': train_args.actor_learning_rate},
        {'params': agent.actor_mean.parameters(), 'lr': train_args.actor_learning_rate},
        {'params': agent.critic.parameters(), 'lr': train_args.critic_learning_rate},
    ], eps = 1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((train_args.num_steps_per_rollout, train_args.num_envs) + envs.single_observation_space.shape).to(device)
    uc_actions = torch.zeros((train_args.num_steps_per_rollout, train_args.num_envs) + envs.single_action_space.shape).to(device) #UNCLIPPED actions
    logprobs = torch.zeros((train_args.num_steps_per_rollout, train_args.num_envs)).to(device)
    rewards = torch.zeros((train_args.num_steps_per_rollout, train_args.num_envs)).to(device)
    backlogs = torch.zeros((train_args.num_steps_per_rollout, 1)).to(device)
    dones = torch.zeros((train_args.num_steps_per_rollout, train_args.num_envs)).to(device)
    values = torch.zeros((train_args.num_steps_per_rollout, train_args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    train_steps = 0
    update_counter = 0
    mini_batch_counter = 0
    reset_counter = 0
    rollout_counter = 0
    wandb.define_metric("counters/global_steps")
    wandb.define_metric("counters/train_steps")
    wandb.define_metric("counters/rollouts")
    wandb.define_metric("counters/training_episodes")
    wandb.define_metric("counters/mb_updates")
    wandb.define_metric("counters/updates")
    wandb.define_metric("train/*", step_metric="counters/training_episodes")
    wandb.define_metric("train/mean_backlog", step_metric = "counters/train_steps")
    wandb.define_metric("losses/*", step_metric="counters/rollouts")
    wandb.define_metric("misc/*", step_metric = "counters/rollouts")


    start_time = time.time()
    next_obs, _ = envs.reset(seed=train_args.seed)  ## TODO: see what reset() normally returns
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(train_args.num_envs).to(device)
    num_updates = train_args.total_timesteps // train_args.batch_size  # We don't specify the number of updates
    best_LTA = -np.inf
    Stopper = StopTrainingOnNoImprovement(num_updates,train_args.no_improv_thresh)
    manual_stop = False

    while not manual_stop:
        try:
            pbar = tqdm(range(num_updates+1))
            for update in pbar:
                ## LEARNING RATE SCHEDULING
                if True:
                    if train_args.actor_lr_decay == "linear":
                        frac = 1.0 - (update - 1.0) / num_updates
                        lrnow = frac * train_args.actor_lr_decay_rate * train_args.actor_learning_rate
                        optimizer.param_groups[0]["lr"] = lrnow # updating actor_logstd learning_rate
                        optimizer.param_groups[1]["lr"] = lrnow # updating actor_mean learning_rate
                    elif train_args.actor_lr_decay == "exponential":
                        lrnow = train_args.actor_learning_rate * np.exp(-train_args.actor_lr_decay_rate * update)
                        optimizer.param_groups[0]["lr"] = lrnow
                        optimizer.param_groups[1]["lr"] = lrnow
                    if train_args.critic_lr_decay == "linear":
                        frac = 1.0 - (update - 1.0) / num_updates
                        lrnow = frac * train_args.critic_lr_decay_rate * train_args.critic_learning_rate
                        optimizer.param_groups[2]["lr"] = lrnow  # updating critic learning_rate
                    elif train_args.critic_lr_decay == "exponential":
                        lrnow = train_args.critic_learning_rate * np.exp(-train_args.critic_lr_decay_rate * update)
                        optimizer.param_groups[2]["lr"] = lrnow
                if False: # Really applies to shared learning rate
                    if train_args.actor_lr_decay == "linear":
                        frac = 1.0 - (update - 1.0) / num_updates
                        lrnow = frac * train_args.actor_lr_decay_rate * train_args.actor_learning_rate
                        optimizer.param_groups[0]["lr"] = lrnow # updating actor_logstd learning_rate

                ## (1) COLLECT ROLLOUT
                '''
                STEP (1): COLLECT ROLLLOUT
                    For all environments, we collect a rollout of length train_args.num_steps, and record for each (env,step)
                    a. obs - state which is input into the policy and value networks
                    b. dones - if the environment was terminated/truncated
                    c. uc_action - UNCLIPPED action output by policy function
                    d. logpob - log probability of taking said action i.e. log \pi(a_t|s_t)
                    e. value - value network output i.e. v(s_t)
                    f. reward - observed rewards 
                '''
                # Note: each `step` corresponds to a step in all parallel environments run simultaneously
                # Parameter in train_args json is "num_steps_per_env" and the num_steps used below is num_steps_per_env * the number of envs

                for step in range(0, train_args.num_steps_per_rollout): # num_steps: max number of steps PER ROLLOUT
                    global_step += 1 * train_args.num_envs
                    train_steps += 1
                    wandb.log({"counters/train_steps":train_steps})
                    wandb.log({"counters/global_steps":global_step})
                    obs[step] = next_obs
                    dones[step] = next_done

                    # ALGO LOGIC: action logic
                    with torch.no_grad():
                        uc_action, logprob, _, value = agent.get_action_and_value(next_obs)
                        values[step] = value.flatten()
                    uc_actions[step] = uc_action
                    logprobs[step] = logprob

                    # TRY NOT TO MODIFY: execute the game and log data.
                    next_obs, reward, terminated, truncated, infos = envs.step(uc_action.cpu().numpy())
                    done = np.logical_or(terminated, truncated)
                    rewards[step] = torch.tensor(reward).to(device).view(-1)
                    next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)


                    if "final_info" in infos:
                        reset_counter += 1
                        wandb.log({"counters/training_episodes": reset_counter})
                        sum_eps_LTA_returns = 0
                        sum_eps_returns = 0
                        sum_eps_length = 0
                        sum_backlog = 0
                        sum_delivered = 0
                        n_eps = 0
                        for info in infos["final_info"]:
                            # Skip the envs that are not done
                            if info is None:
                                continue
                            n_eps += 1
                            sum_backlog += info["backlog"]
                            sum_delivered += info["delivered"]
                            sum_eps_returns += info["episode"]["r"]  # Raw return
                            sum_eps_LTA_returns += info['episode']['r'] / info['episode'][
                                'l']  # returns averaged by the episode length
                            sum_eps_length += info["episode"]["l"]  #
                            # wandb.log({
                            #     "train/episodic_return": info["episode"]["r"],
                            #     "train/episodic_length": info["episode"]["l"],
                            #     "train/episodic_average": average_eps_reward,
                            # })
                        avg_backlog = sum_backlog / n_eps
                        avg_delivered = sum_delivered / n_eps
                        avg_eps_return = sum_eps_returns / n_eps
                        avg_LTA_return = sum_eps_LTA_returns / n_eps
                        avg_eps_length = sum_eps_length / n_eps

                        wandb.log({
                            "train/mean_eps_return": avg_eps_return,
                            "train/avg_LTA_return": avg_LTA_return,
                            "train/avg_eps_backlog": avg_backlog,
                            "train/mean_backlog" : avg_backlog,
                            "train/avg_eps_delivered" : avg_delivered,
                            "train/avg_eps_length": avg_eps_length
                        })

                        best_scores, info_string = checkpoint_saver(agent, reset_counter, avg_LTA_return[0])

                        if avg_LTA_return > best_LTA:  # check if the average LTA reward is greater than the previous best
                            new_best = True
                            torch.save(agent.state_dict(), checkpoint_saver.dirpath + f"manual_save.pt")
                            best_LTA = avg_LTA_return
                            wandb.log({"train/best_LTA": best_LTA})
                    # (Above) if "final_info" in infos -> i.e. the episode has ended, env reset, and done = True
                    else:
                        backlogs[step] = torch.Tensor(infos["backlog"].mean()).to(device)
                        wandb.log({"train/delivered": infos["delivered"].mean()})
                        wandb.log({"train/mean_backlog": backlogs[step].to("cpu").numpy()})
                        continue



                # A ROLLOUT HAS BEEN COLLECTED

                rollout_counter += 1
                wandb.log({"counters/rollouts": rollout_counter})


                """
                STEP (2): LEARNING PHASE COMPUTE LOSSES AND GRADIENTS FROM ROLLOUT DATA
                """

                with torch.no_grad():
                    # bootstrap value if not done
                    next_value = agent.get_value(next_obs).reshape(1, -1)
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    ''' (2a) Advantage Loop with Bootstrapping
                        if a sub-environment is not terminated nor truncated, 
                        PPO estimates the value of the next state in this sub-environment as the value target.
                        Recall: done is true if the trajectory ended AFTER a step
                    '''
                    for t in reversed(range(train_args.num_steps_per_rollout)):
                        if t == train_args.num_steps_per_rollout - 1: # i.e. the final step in the rollout
                            nextnonterminal = 1.0 - next_done   # 0 if the final step was terminal, 1 otherwise
                            nextvalues = next_value             # nextvalues = critic estimate of the final state
                        else:   # if the step is not the final one
                            nextnonterminal = 1.0 - dones[t + 1] # check if the next step was terminal
                            nextvalues = values[t + 1]           # next_values = critic estimate of the next step

                        delta = rewards[t] + train_args.gamma * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + train_args.gamma * train_args.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + values

                ## Prep data into flat batches
                b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
                b_logprobs = logprobs.reshape(-1)
                b_actions = uc_actions.reshape((-1,) + envs.single_action_space.shape)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1) #
                b_values = values.reshape(-1)
                b_inds = np.arange(train_args.batch_size)

                # Optimizing the policy and value network
                clipfracs = []


                for epoch in range(train_args.updates_per_rollout):
                    update_counter +=1
                    wandb.log({"counters/updates": update_counter})
                    # Shuffle the batch indices
                    np.random.shuffle(b_inds)

                    # Loop through 'start' indices -> (0, minibatch_size, 2*minibatch_size, ...)
                    for start in range(0, train_args.batch_size, train_args.minibatch_size):
                        mini_batch_counter += 1
                        wandb.log({"counters/mb_updates": mini_batch_counter})
                        # Get end index
                        end = start + train_args.minibatch_size

                        # All minibatch indices
                        mb_inds = b_inds[start:end]

                        # Pass minibatch observations and actions into the policy and value networks
                        # Computes log probability, entropy, and value estimate for the (s,a) pairs
                        _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])

                        # Compute log-ratio and ration based on the old policy (b_logprobs[mb_inds])) \
                        # and any updates to the policy network since obtaining the data (newlogprob)
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        # Approximate the kl divergence between the old_policy and updated policy, and
                        # compute the fraction of training data that triggered clipping ratios for the policy network
                        with torch.no_grad():
                            # calculate approx_kl http://joschu.net/blog/kl-approx.html
                            old_approx_kl = (-logratio).mean()  #k1 in above blog
                            approx_kl = ((ratio - 1) - logratio).mean() #k2 in above blog
                            clipfracs += [((ratio - 1.0).abs() > train_args.clip_coef).float().mean().item()]

                        # Get minibatch advantages and normalize if required
                        mb_advantages = b_advantages[mb_inds]
                        if train_args.norm_adv:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio # unclipped loss
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - train_args.clip_coef, 1 + train_args.clip_coef) # clipped loss
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean() # take the maximum of the negative losses (i.e. the minimum)

                        # Value loss
                        newvalue = newvalue.view(-1)
                        if train_args.clip_vloss:
                            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                            v_clipped = b_values[mb_inds] + torch.clamp(
                                newvalue - b_values[mb_inds],
                                -train_args.clip_coef,
                                train_args.clip_coef,
                                )
                            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()
                        else:
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                        entropy_loss = entropy.mean()
                        loss = pg_loss - train_args.ent_coef * entropy_loss + v_loss * train_args.vf_coef

                        optimizer.zero_grad()
                        loss.backward()
                        if train_args.max_grad_norm> 0:
                            nn.utils.clip_grad_norm_(agent.parameters(), train_args.max_grad_norm)
                        optimizer.step()

                    if eval(train_args.target_kl) is not None:
                        if approx_kl > train_args.target_kl:
                            break
                """
                y_pred: output of value network
                y_true: true value function
                """
                y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                var_y = np.var(y_true)
                diff_y = y_true-y_pred
                abs_diff_y = np.abs(diff_y)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                wandb.log({
                    "train/actor_learning_rate": optimizer.param_groups[0]["lr"],
                    #"train/critic_learing_rate": optimizer.param_groups[2]["lr"],
                    "train/SPS": int(train_steps / (time.time() - start_time)),
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    # "losses/policy_loss1": pg_loss1.mean().item(),
                    # "losses/policy_loss2": pg_loss2.mean().item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/approx_kl1": old_approx_kl.item(),
                    "losses/approx_kl2": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "misc/abs_diff_y": abs_diff_y.mean(),
                    "misc/y_pred": y_pred.mean(),
                    "misc/y_measured": y_true.mean(),
                    "misc/explained_variance": explained_var,
                    "misc/var_y_true": var_y

                })


                # writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                # writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
                # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                # writer.add_scalar("losses/explained_variance", explained_var, global_step)
                # # print("SPS:", int(global_step / (time.time() - start_time)))
                # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            manual_stop = True
            print(f"Training concluded after {update} updates")
        except KeyboardInterrupt:
            print('\n')
            print("!" * 30)
            print(f"Manually stopping after {update} updates")
            manual_stop = True


    print("="*30)
    print("TESTING BEST AGENT FROM CURRENT TRAINING RUN")
    agent.load_state_dict(torch.load(checkpoint_saver.dirpath + f"manual_save.pt"))
    test_outputs = agent_test(run, agent, env_para, test_args, store_history=True)
    envs.close()
    writer.close()
    return {"Stopper": Stopper,
            "Agent": agent,
            "test_outputs": test_outputs}
