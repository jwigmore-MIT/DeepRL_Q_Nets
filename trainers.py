# Import libraries

import time
import wandb
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import logging
logging.getLogger().setLevel(logging.INFO)



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






def train_agent(env_para, train_args, test_args, run, checkpoint_saver):



    ## Setup tensorboard
    writer = SummaryWriter(run.dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(train_args).items()])),
        )

    ## Set the seed of random, np.random, and torch
    # TRY NOT TO MODIFY: seeding
    # random.seed(train_args.seed)
    # np.random.seed(train_args.seed)
    # torch.manual_seed(train_args.seed)
    torch.backends.cudnn.deterministic = train_args.torch_deterministic

    ## Select the device
    device = torch.device("cuda" if torch.cuda.is_available() and train_args.cuda else "cpu")

    # Initialize the environments
    envs = gym.vector.SyncVectorEnv(
        [make_MCMH_env(env_para, max_steps = train_args.num_steps) for i in range(train_args.num_envs)]
    )

    # Initialize agents and pass agents (nn.module) to device
    agent = Agent(envs).to(device)

    # Watch the gradient
    wandb.watch(agent, log_freq = 100)

    ## Initialize ADAM optimizer
    optimizer = optim.Adam(agent.parameters(), lr=train_args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((train_args.num_steps, train_args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((train_args.num_steps, train_args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((train_args.num_steps, train_args.num_envs)).to(device)
    rewards = torch.zeros((train_args.num_steps, train_args.num_envs)).to(device)
    dones = torch.zeros((train_args.num_steps, train_args.num_envs)).to(device)
    values = torch.zeros((train_args.num_steps, train_args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
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
            for update in range(1, num_updates + 1):
                # try:
                #     stdin = sys.stdin.read()
                #     if "\n" in stdin or "\r" in stdin:
                #         print(f"Received User Input - Ending training after {update} updates")
                #         break
                # except IOError:
                #     pass
                # Annealing the rate if instructed to do so.
                if train_args.anneal_lr:
                    frac = 1.0 - (update - 1.0) / num_updates
                    lrnow = frac * train_args.learning_rate
                    optimizer.param_groups[0]["lr"] = lrnow
                # Note: each `step` corresponds to a step in all parallel environments run simultaneously
                for step in range(0, train_args.num_steps): # num_steps: number of steps PER ROLLOUT
                    global_step += 1 * train_args.num_envs
                    obs[step] = next_obs
                    dones[step] = next_done

                    # ALGO LOGIC: action logic
                    with torch.no_grad():
                        action, logprob, _, value = agent.get_action_and_value(next_obs)
                        values[step] = value.flatten()
                    actions[step] = action
                    logprobs[step] = logprob

                    # TRY NOT TO MODIFY: execute the game and log data.
                    next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
                    done = np.logical_or(terminated, truncated)
                    rewards[step] = torch.tensor(reward).to(device).view(-1)
                    next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

                    # Only print when at least 1 env is done
                    if "final_info" not in infos:
                        continue


                    sum_avg_eps_rewards = 0
                    sum_eps_returns = 0
                    for info in infos["final_info"]:
                        # Skip the envs that are not done
                        if info is None:
                            continue
                        average_eps_reward = info['episode']['r' ] /info['episode']['l']
                        sum_avg_eps_rewards += average_eps_reward
                        sum_eps_returns += info["episode"]["r"]
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        writer.add_scalar("charts/episodic_average", average_eps_reward , global_step)
                    avg_LTA_reward = sum_avg_eps_rewards /infos['_final_info'].sum()
                    avg_eps_return = sum_avg_eps_rewards /infos['_final_info'].sum()

                    checkpoint_saver(agent, update ,avg_LTA_reward[0])

                    if avg_LTA_reward > best_LTA: # check if the average LTA reward is greater than the previous best
                        print(f"FOUND NEW BEST POLICY: Making a manual save")
                        # torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")
                        torch.save(agent.state_dict(), checkpoint_saver.dirpath + f"manual_save.pt")
                        best_LTA = avg_LTA_reward
                        #if abs(best_LTA )< 7:
                        #    test_rewards, test_history = agent_test(run, agent, env_para, test_args, store_history= True)
                    print(
                        f"global_step={global_step}, avg_episodic_return={avg_eps_return}, average_LTA_reward = {avg_LTA_reward}, current_best = {best_LTA}")
                    stop = Stopper.update(avg_LTA_reward)

                # bootstrap value if not done
                with torch.no_grad():
                    next_value = agent.get_value(next_obs).reshape(1, -1)
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    for t in reversed(range(train_args.num_steps)):
                        if t == train_args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = rewards[t] + train_args.gamma * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + train_args.gamma * train_args.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + values

                # flatten the batch
                b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
                b_logprobs = logprobs.reshape(-1)
                b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = values.reshape(-1)

                # Optimizing the policy and value network
                b_inds = np.arange(train_args.batch_size)
                clipfracs = []
                for epoch in range(train_args.update_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, train_args.batch_size, train_args.minibatch_size):
                        end = start + train_args.minibatch_size
                        mb_inds = b_inds[start:end]

                        _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            # calculate approx_kl http://joschu.net/blog/kl-approx.html
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [((ratio - 1.0).abs() > train_args.clip_coef).float().mean().item()]

                        mb_advantages = b_advantages[mb_inds]
                        if train_args.norm_adv:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - train_args.clip_coef, 1 + train_args.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

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
                        nn.utils.clip_grad_norm_(agent.parameters(), train_args.max_grad_norm)
                        optimizer.step()

                    if eval(train_args.target_kl) is not None:
                        if approx_kl > train_args.target_kl:
                            break

                y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
                writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                writer.add_scalar("losses/explained_variance", explained_var, global_step)
                # print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        except KeyboardInterrupt:
            print('\n')
            print("!" * 30)
            print(f"Manually stopping after {update} updates")
            manual_stop = True


    print("="*30)
    print("TESTING BEST AGENT FROM CURRENT TRAINING RUN")
    agent.load_state_dict(torch.load(checkpoint_saver.dirpath + f"manual_save.pt"))
    test_rewards, test_history = agent_test(run, agent, env_para, test_args, store_history=True)
    envs.close()
    writer.close()
    return {"Stopper": Stopper,
            "Agent": agent}
