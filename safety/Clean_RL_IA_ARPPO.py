""" Clean RL Implementation of Intervention Assisted Average Reward PPO
Difference from ARPPO:
    1. Before taking a step, checks to see if state is safe
    2. If state is safe, take step as normal
    3. If state is unsafe, take step with intervention
    4. Log whether or not safe was safe


"""

"""
TODO: Test with current implementations of ServerAllocation and ServerAssignment
1. Check masking functionality
2. Rerun baselines
"""




# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# my imports

from safety.utils import clean_rl_ppo_parse_config
from tqdm import tqdm
from safety.clean_rl_utils import observation_checker, parse_args_or_config, generate_clean_rl_env, apply_obs_wrapper, apply_reward_wrapper

np.seterr(all='raise')



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, temperature=1.0, learn_temperature=False, hidden_size=64, hidden_depth=2, actor_hidden_dims = None, critic_hidden_dims = None, mask_value = -1e8):
        super().__init__()
        self.mask_value = mask_value
        self.learn_temperature = learn_temperature
        if learn_temperature == "DNN":
            # state based temperature learning
            self.temperature = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                nn.Sigmoid(),
                layer_init(nn.Linear(64, 1), std=1.0, bias_const=temperature),
                nn.Sigmoid(),
            )
        elif learn_temperature:
            self.temperature = nn.Parameter(torch.ones(1)*temperature)
        else:
            self.temperature = temperature
        self.deterministic = False



        # Critic
        if critic_hidden_dims is None:
            critic_hidden_depth = hidden_depth
            critic_hidden_size = hidden_size
        else:
            critic_hidden_depth = critic_hidden_dims[0]
            critic_hidden_size = critic_hidden_dims[1]

        in_dim = np.array(envs.single_observation_space.shape).prod()
        critic_layers = []
        for i in range(critic_hidden_depth):
            critic_layers.append(layer_init(nn.Linear(in_dim, critic_hidden_size)))
            critic_layers.append(nn.Tanh())
            in_dim = critic_hidden_size
        critic_layers.append(layer_init(nn.Linear(in_dim, 1), std=1.0))
        self.critic = nn.Sequential(*critic_layers)
       # Actor
        if actor_hidden_dims is None:
            actor_hidden_depth = hidden_depth
            actor_hidden_size = hidden_size
        else:
            actor_hidden_depth = actor_hidden_dims[0]
            actor_hidden_size = actor_hidden_dims[1]
        in_dim = np.array(envs.single_observation_space.shape).prod()
        out_dim = envs.single_action_space.n
        actor_layers = []
        for i in range(actor_hidden_depth):
            actor_layers.append(layer_init(nn.Linear(in_dim, actor_hidden_size)))
            actor_layers.append(nn.Tanh())
            in_dim = actor_hidden_size
        actor_layers.append(layer_init(nn.Linear(in_dim, out_dim), std=0.01))
        self.actor = nn.Sequential(*actor_layers)


    def get_temperature(self, x):
        if self.learn_temperature == "DNN":
            return self.temperature(x).clamp(0.1, 10.0)
        elif self.learn_temperature:
            return self.temperature.clamp(0.1, 10.0)
        else:
            return self.temperature
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, mask = None):

        logits = self.actor(x)/self.get_temperature(x)
        if mask is not None:
            # convert mask to Tensor and match shape of logits
            if isinstance(mask, torch.Tensor):
                mask = mask.reshape(logits.shape).bool()
            else:
                mask = torch.Tensor(mask).to(logits.device).reshape(logits.shape).bool()
            logits[mask] = self.mask_value
        probs = Categorical(logits=logits)
        if action is None:
            if self.deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

def eval_model(agent, args, train_step = 0, test = False, pbar = None):
    if pbar is not None:
        if not test:
            pbar.set_description(f"Running Eval - Train Step {train_step}")
        else:
            pbar.set_description(f"Running Test - Train Step {train_step}")

    """Evaluate the agent"""
    # Set new seed for each eval
    seed = args.seed + train_step
    # Generate the eval environment
    eval_env = generate_clean_rl_env(args)()

    # Set agent to eval mode
    agent.eval()
    steps = args.eval_steps if not test else args.test_steps
    # Initialize variables
    eval_obs = torch.zeros((steps, 1) + eval_env.observation_space.shape).to(args.device)
    eval_actions = torch.zeros((steps, 1) + eval_env.action_space.shape).to(args.device)
    eval_interventions = torch.zeros((steps)).to(args.device)
    eval_rewards = torch.zeros((steps)).to(args.device)
    eval_backlogs = np.zeros((steps))
    agent.eval()
    next_obs_array, next_info = eval_env.reset(seed = seed)
    next_obs = torch.Tensor(next_obs_array).to(device)
    eval_sum_backlogs = 0
    for t in range(steps):
        observation_checker(next_obs)
        eval_obs[t] = next_obs

        if eval_env.get_backlog() > args.int_thresh:  # minus one to account for the source packet

            np_action = eval_env.get_stable_action(args.stable_policy)
            action = torch.Tensor([np_action])
            # action = torch.Tensor(np.argmin(buffers)).to(device)
            with torch.no_grad():
                _, log_prob, _, value = agent.get_action_and_value(next_obs.to(device), action.to(device))
                eval_interventions[t] = torch.Tensor([1]).to(device)
        else:
            # ALGO LOGIC: action logic
            with torch.no_grad():
                if args.apply_mask:
                    mask = eval_env.get_mask()
                else:
                    mask = None
                action, logprob, _, value = agent.get_action_and_value(next_obs, mask = mask)
                eval_interventions[t] = torch.Tensor([0]).to(device)
        eval_actions[t] = action

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, terminated, truncated, info = eval_env.step(action.item())
        done = np.array(terminated | truncated)
        eval_rewards[t] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
        eval_backlogs[t] = info['backlog']
        eval_sum_backlogs += info['backlog']
        if t > 0 and t % 100 == 0:
            if t >= args.window_size:
                window_averaged_backlog = np.mean(
                    eval_backlogs[t - args.window_size:t])
            else:
                window_averaged_backlog = np.mean(eval_backlogs[:t])
            lta_backlogs = np.cumsum(eval_backlogs[:t]) / np.arange(1, t + 1)
            wb_prefix = f"eval/eval_{train_step}" if not test else "test"
            wandb.log({f"{wb_prefix}/lta_backlogs": lta_backlogs[-1],
                       f"{wb_prefix}/window_averaged_backlog": window_averaged_backlog,
                       f"eval/eval_step": t})
    wandb.log({f"eval/eval_lta_backlog": lta_backlogs[-1],
               f"eval/global_steps": train_step,})

    return lta_backlogs[-1]






if __name__ == "__main__":
    config_file = os.path.relpath("clean_rl/ServerAllocation/M8/M8A1-O_IA_AR_PPO.yaml")


    args = parse_args_or_config(config_file)
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    args.total_timesteps = int(args.total_timesteps)

    run_name = f"[{args.policy_name}] {args.env_name} - {int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [generate_clean_rl_env(args, normalize= False) for i in range(args.num_envs)]
    )


    if hasattr(args, "critic_hidden_dims") and hasattr(args, "actor_hidden_dims") \
            and args.critic_hidden_dims is not None and args.actor_hidden_dims is not None:
        critic_hidden_dims = args.critic_hidden_dims
        actor_hidden_dims = args.actor_hidden_dims
        hidden_size = None
        hidden_depth = None
    elif hasattr(args, 'hidden_size'):
        hidden_size = args.hidden_size
        hidden_depth = args.hidden_depth
        critic_hidden_dims = None
        actor_hidden_dims = None
    else:
        hidden_size = 64
        hidden_depth = 2
        critic_hidden_dims = None
        actor_hidden_dims = None
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    if hasattr(args, 'temperature'):
        temperature = args.temperature
        learn_temperature = args.learn_temperature if hasattr(args, 'learn_temperature') else False
    else:
        temperature = 1.0
        learn_temperature = False

    agent = Agent(envs, temperature = args.temperature, learn_temperature=learn_temperature,
                  hidden_size=hidden_size, hidden_depth = hidden_depth,
                  actor_hidden_dims=actor_hidden_dims, critic_hidden_dims= critic_hidden_dims).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    masks = torch.zeros((args.num_steps, args.num_envs) + (envs.action_space.nvec[0],)).to(device)
    interventions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    backlogs = np.zeros((args.num_steps, args.num_envs))
    total_backlogs = np.zeros((args.total_timesteps, args.num_envs))
    mean_diffs = np.zeros((args.total_timesteps, args.num_envs))
    last_ks = np.zeros((args.total_timesteps, args.num_envs))

    sum_backlogs = 0
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs_array, next_info = envs.reset()
    next_obs = torch.Tensor(next_obs_array).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    #pbar = tqdm(range(num_updates), ncols=80, desc="Training Episodes")

    # Average Reward Variables
    eta = 0
    beta = 0
    elapsed_eval_time = 0

    # Running mean
    last_k = []
    k = args.k
    drift_threshold = args.drift_threshold
    # rollout the stable policy until reaching steady state
    if args.stable_rollout:

        terminate = False
        max_rollout_steps = args.max_stable_rollout
        # sr = stable rollout
        sr_states = np.zeros((max_rollout_steps, args.num_envs) + envs.single_observation_space.shape)
        sr_actions = np.zeros((max_rollout_steps, args.num_envs) + envs.single_action_space.shape)
        sr_rewards = np.zeros((max_rollout_steps, args.num_envs))
        sr_backlogs = np.zeros((max_rollout_steps, args.num_envs))
        sr_steps = 0
        sr_pbar = tqdm(total = max_rollout_steps, ncols=80, desc="Stable Rollout")
        while not terminate:
            sr_steps+=1
            if sr_steps > max_rollout_steps:
                break
            else:
                action = envs.call("get_stable_action", args.stable_policy)
                #torch_action  = torch.Tensor([action]).to(device).int()

                next_obs_array, reward, terminated, truncated, info = envs.step(action)

                sr_states[sr_steps] = next_obs_array
                sr_actions[sr_steps] = action
                sr_rewards[sr_steps] = reward
                sr_backlogs[sr_steps] = info['backlog'].item()
                # check if steady state is reached
                if sr_steps > 10000:
                    if np.mean(sr_backlogs[sr_steps-1000:sr_steps]) == np.mean(sr_backlogs[sr_steps-2000:sr_steps-1000]):
                        terminate = True
                if sr_steps % args.num_steps == 0 or terminate:
                    #sr_pbar.set_description(f"Stable Rollout - Step {sr_steps}")
                    sr_pbar.update(n = args.num_steps)
                    lta_backlogs = np.cumsum(sr_backlogs[:sr_steps])/np.arange(1, sr_steps+1)
                    log_dict = {"sr_steps": sr_steps,
                                "sr_backlog": np.mean(sr_backlogs[sr_steps-args.num_steps:sr_steps]),
                                "sr_lta_backlog": lta_backlogs[-1]}
                    wandb.log(log_dict)

        if True:
            import matplotlib.pyplot as plt
            sr_lta_backlogs = np.cumsum(sr_backlogs[:sr_steps])/np.arange(1, sr_steps+1)
            plt.plot(sr_backlogs[:sr_steps])
            plt.plot(sr_lta_backlogs)
            plt.show()
        max_state = np.max(sr_states, axis=0)
        max_buffer_state = max_state[:,:envs.envs[0].unwrapped.n_queues]
        mean_state = np.mean(sr_states[sr_steps-1000:sr_steps], axis=0)
        mean_buffer_state = mean_state[:,:envs.envs[0].unwrapped.n_queues]
        if args.intervention_type == "substate":
            if args.intervention_threshold_type == "max":
                substate_threshold = max_buffer_state * args.intervention_threshold_coef
                print(f"Max substate: {max_buffer_state}")
                print(f"Intervention threshold coefficient: {args.intervention_threshold_coef}")
                print(f"Substate threshold: {substate_threshold}")
            elif args.intervention_threshold_type == "mean":
                substate_threshold = mean_buffer_state*args.intervention_threshold_coef
                print(f"Mean substate: {mean_buffer_state}")
                print(f"Intervention threshold coefficient: {args.intervention_threshold_coef}")
                print(f"Substate threshold: {substate_threshold}")
        elif args.intervention_type == "backlog":
            if args.intervention_threshold_type == "max":
                backlog_threshold = args.intervention_threshold_coef * np.max(sr_backlogs[sr_steps-1000:sr_steps]) # take the max of the last 1000 steps
                print(f"Max backlog: {np.max(sr_backlogs[sr_steps-1000:sr_steps])}")
                print(f"Intervention threshold coefficient: {args.intervention_threshold_coef}")
                print(f"Backlog threshold: {backlog_threshold}")
            elif args.intervention_threshold_type == "mean":
                backlog_threshold = args.intervention_threshold_coef * np.mean(sr_backlogs[sr_steps-1000:sr_steps]) # take the mean of the last 1000 steps
                print(f"Mean backlog: {np.mean(sr_backlogs[sr_steps-1000:sr_steps])}")
                print(f"Intervention threshold coefficient: {args.intervention_threshold_coef}")
                print(f"Backlog threshold: {backlog_threshold}")



        # need to get max_states before normalization
        # Apply normalization based on max_state

        buffer_norm_factor = max_buffer_state.sum()*1.5
        envs.envs[0] = apply_obs_wrapper(envs.envs[0], args, buffer_norm_factor)
        envs.envs[0] = apply_reward_wrapper(envs.envs[0], args)
        # take one more step to get normalized next_obs information
        action = envs.call("get_stable_action", args.stable_policy)
        # torch_action  = torch.Tensor([action]).to(device).int()
        next_obs_array, reward, terminated, truncated, info = envs.step(action)
        next_obs = torch.Tensor(next_obs_array).to(device)

    else:
        if args.intervention_type == "backlog":
            backlog_threshold = args.fixed_backlog_threshold
        elif args.intervention_type == "substate":
            substate_threshold = args.fixed_substate_threshold

    pbar = tqdm(total = args.total_timesteps, ncols=80, desc="Training Steps", position=0, leave=True)

    for update in range(num_updates):
        pbar.update(n = args.num_steps)
        # Cutoff learning based on number of steps


        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        if global_step > args.learning_steps:
            lr_now = 0
            optimizer.param_groups[0]["lr"] = lr_now
            pbar.set_description("Rolling out final policy")
            agent.deterministic = True
        # for debuggin advantage based intervention
        temperatures = np.zeros(args.num_steps)
        # Generate Trajectory
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            observation_checker(next_obs)
            obs[step] = next_obs
            dones[step] = next_done
            if args.intervention_type == "backlog": # threshold based intervention
                if envs.call("get_backlog")[0] > backlog_threshold:
                    reward_penalty = - args.intervention_penalty
                    np_action = envs.call("get_stable_action", args.stable_policy)[0]
                    action = torch.Tensor([np_action]).to(device).int()
                    with torch.no_grad():
                        _, logprob, _, value = agent.get_action_and_value(next_obs, action)
                        values[step] = value.flatten()
                        interventions[step] = torch.Tensor([1]).to(device)
                else:
                    reward_penalty = 0
                    with torch.no_grad():
                        if args.apply_mask:
                            mask = envs.call("get_mask")[0]
                        else:
                            mask = None
                        action, logprob, _, value = agent.get_action_and_value(next_obs, mask = mask)
                        values[step] = value.flatten()
                        interventions[step] = torch.Tensor([0]).to(device)
            elif args.intervention_type == "substate": # max_state based intervention
                if (envs.call("get_buffers") > substate_threshold).any():
                    reward_penalty = - args.intervention_penalty
                    np_action = envs.call("get_stable_action", args.stable_policy)[0]
                    action = torch.Tensor([np_action]).to(device).int()
                    with torch.no_grad():
                        _, logprob, _, value = agent.get_action_and_value(next_obs, action)
                        values[step] = value.flatten()
                        interventions[step] = torch.Tensor([1]).to(device)
                else:
                    reward_penalty = 0
                    with torch.no_grad():
                        if args.apply_mask:
                            mask = envs.call("get_mask")[0]
                        else:
                            mask = None
                        action, logprob, _, value = agent.get_action_and_value(next_obs, mask=mask)
                        values[step] = value.flatten()
                        interventions[step] = torch.Tensor([0]).to(device)
            #minus one to account for the source packet
            elif False: #drift based intervention
                # take the mean of the difference of the last 10 steps
                if last_k.__len__()< 2:
                    mean_diff = 0
                else:
                    mean_diff = np.mean(np.diff(last_k))


                # if global_step > 10:
                #     mean_diff = np.diff(total_backlogs[global_step-100:global_step]).mean()
                # elif global_step <= 1:
                #     mean_diff = 0
                # else:
                #     mean_diff = np.diff(total_backlogs[:global_step+1]).mean()
                if mean_diff >  drift_threshold: # positive growth on average
                    reward_penalty = - args.intervention_penalty
                    np_action = envs.call("get_stable_action", args.stable_policy)[0]
                    action = torch.Tensor([np_action]).to(device).int()
                    with torch.no_grad():
                        _, logprob, _, value = agent.get_action_and_value(next_obs, action)
                        values[step] = value.flatten()
                        interventions[step] = torch.Tensor([1]).to(device)
                else: # negative or even growth on average
                    reward_penalty = 0
                    with torch.no_grad():
                        if args.apply_mask:
                            mask = envs.call("get_mask")[0]
                        else:
                            mask = None
                        action, logprob, _, value = agent.get_action_and_value(next_obs, mask=mask)
                        values[step] = value.flatten()
                        interventions[step] = torch.Tensor([0]).to(device)
                        if args.apply_mask: masks[step] = torch.Tensor(mask).to(device)
                mean_diffs[global_step] = mean_diff
                last_k.append(backlogs[step].item())

            temperatures[step] = agent.get_temperature(next_obs)
            actions[step] = action
            #last_ks[global_step] = np.mean(last_k)
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = terminated | truncated
            rewards[step] = torch.tensor(reward+reward_penalty).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            backlogs[step] = info['backlog'][0]
            sum_backlogs += info['backlog'][0]
            # if last_k.__len__() > k:
            #     last_k.pop(0)
            # if step % 64 == 0:
            #     writer.add_scalar("mean_diff", mean_diff, global_step)

        # Compute and log time-average backlog to write
        time_averaged_backlog = sum_backlogs /global_step

        writer.add_scalar("time_averaged_backlog", time_averaged_backlog, global_step)
        ## Keep track of total backlogs ##
        total_backlogs[global_step - args.num_steps:global_step] = backlogs

        # Compute and log window-average backlog to writer
        if hasattr(args, 'window_size') and global_step > args.window_size:
            window_backlogs = total_backlogs[global_step - args.window_size:global_step]

            window_average_backlog = np.mean(window_backlogs)
            writer.add_scalar("window_average_backlog", window_average_backlog, global_step)
        else: window_average_backlog = None

        # Eval
        if args.do_eval:
            elapsed_eval_time += args.num_steps
            if elapsed_eval_time >= args.eval_freq:
                elapsed_eval_time = 0
                eval_LTA_Backlog = eval_model(agent, args, train_step = global_step, test = False, pbar= pbar)
                #make sure agent is in train mode
                agent.train()
                pbar.set_description("Training Episodes")



        # Update Average Reward Variables
        eta = (1-args.alpha)*eta + args.alpha*rewards.mean()
        beta = (1-args.alpha)*beta + args.alpha*values.mean()
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] - eta + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_masks = masks.reshape((-1,) + (envs.action_space.nvec[0],))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_inter = interventions.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        entropies = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            # Recording mb_stats
            mb_stats = {"mb_adv_mean": [], "mb_adv_std": []}
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                if args.on_policy:
                    mb_interventions = b_inter[mb_inds]
                else:
                    mb_interventions = torch.zeros_like(b_inter[mb_inds])

                mb_non_interventions = (1-mb_interventions).sum()
                if mb_non_interventions > 0:
                    pg_inds = (1-mb_interventions).bool()
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds], b_masks[mb_inds])

                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio[pg_inds]).mean()
                        approx_kl = (((ratio - 1) - logratio)[pg_inds]).mean()
                        clipfracs += [(((ratio - 1.0)[pg_inds]).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages2 = b_advantages[mb_inds]*(1-mb_interventions)
                    mb_stats["mb_adv_mean"].append(mb_advantages.mean().item())
                    mb_stats["mb_adv_std"].append(mb_advantages.std().item())
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -(mb_advantages * ratio)[pg_inds]
                    pg_loss1b = -mb_advantages*ratio*(1-mb_interventions)
                    pg_loss2 = -(mb_advantages* torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef))[pg_inds]
                    pg_loss2b = -mb_advantages*(1-mb_interventions) * torch.clamp(ratio*(1-mb_interventions), 1 - args.clip_coef, 1 + args.clip_coef)

                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()*(1-mb_interventions).float().mean()
                    pg_lossb = torch.max(pg_loss1b, pg_loss2b).mean()
                    entropy_loss = entropy[pg_inds].mean()
                    entropies += [entropy[pg_inds].mean().item()]


                else:
                    pg_loss = entropy_loss = torch.zeros(1).to(device)
                    pg_loss1 = pg_loss2 = entropy = torch.empty(0)
                    old_approx_kl = approx_kl = torch.zeros(1).to(device)
                    clipfracs += [0]
                    # still need new_value for value losses
                    _, _, _, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds], b_masks[mb_inds])
                    mb_advantages = b_advantages[mb_inds]
                    mb_stats["mb_adv_mean"].append(mb_advantages.mean().item())
                    mb_stats["mb_adv_std"].append(mb_advantages.std().item())
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                if clipfracs[-1] > 1:
                    # throw error
                    raise ValueError("Clipping was more than 100%")
                if entropies.__len__() <1:
                    entropy_4log= np.nan
                else:
                    entropy_4log = np.mean(entropies)
                # Value loss
                newvalue = newvalue.view(-1)
                bias_factor = args.nu*beta
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds] - bias_factor) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds] - bias_factor,
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]-bias_factor) ** 2).mean()

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()



            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            error = np.abs(y_pred - y_true)
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if args.track:
                log_dict = {
                    "update_info/learning_rate": optimizer.param_groups[0]["lr"],
                    "update_info/temperature": temperatures.mean(),
                    "update_info/entropy_loss": entropy_4log,
                    "update_info/approx_abs_kl": approx_kl.abs().item(),
                    "update_info/clipfrac": np.mean(clipfracs),
                    "update_info/loss": loss.item(),
                    "update_info/clipped_pg_loss": pg_loss.item(),
                    "update_info/abs_pg_loss": pg_loss.abs().item(),
                    "update_info/unclipped_pg_loss": pg_loss1.mean().item(),
                    "update_info/value_loss": v_loss.item(),
                    "update_info/avg_value": b_values.mean().item(),
                    "update_info/critic_error": error.mean(),
                    "update_info/advantages": advantages.mean().item(),
                    "update_info/explained_variance": explained_var,
                    "update_info/old_approx_kl": old_approx_kl.item(),
                    "update_info/mb_adv_mean": np.mean(mb_stats["mb_adv_mean"]),
                    "update_info/mb_adv_std": np.mean(mb_stats["mb_adv_std"]),
                    "ARPPO_info/eta": eta,
                    "ARPPO_info/beta": beta,
                    "ARPPO_info/bias_factor": bias_factor,

                    "rollout/backlog": np.mean(backlogs[:global_step]),
                    "rollout/time_averaged_backlog": time_averaged_backlog,
                    "rollout/window_average_backlog": window_average_backlog,
                    "rollout/rewards": np.mean(rewards.cpu().numpy()),
                    "rollout/episode": update,
                    "rollout/intervention_rate": interventions.mean().item(),
                    # "rollout/mean_diff": mean_diff,
                    # "rollout/last_k_mean": np.mean(last_k),
                    "update_info/update": update,
                    "global_step": global_step,

                }
                wandb.log(log_dict)
                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break



    # Test the trained policy
    if args.test:
        eval_model(agent, args, train_step = global_step, test = True)
        # args.test_steps = 100000
        # args.test_log_interval = 1000
        #
        # test_obs = torch.zeros((args.test_steps,1) + envs.single_observation_space.shape).to(device)
        # test_actions = torch.zeros((args.test_steps,1) + envs.single_action_space.shape).to(device)
        # test_interventions = torch.zeros((args.test_steps)).to(device)
        # test_rewards = torch.zeros((args.test_steps)).to(device)
        # test_backlogs = np.zeros((args.test_steps))
        # total_backlogs = np.zeros((args.test_steps))
        # envs.reset()
        # agent.eval()
        #
        # next_obs_array, next_info = envs.reset()
        # next_obs = torch.Tensor(next_obs_array).to(device)
        #
        # test_sum_backlogs = 0
        # pbar = tqdm(range(args.test_steps), ncols=80, desc="Test Episode")
        # for t in pbar:
        #     observation_checker(next_obs)
        #     test_obs[t] = next_obs
        #
        #     if envs.get_attr("get_backlog")[0] > args.int_thresh:  # minus one to account for the source packet
        #         buffers = envs.get_attr("get_obs")[0]
        #         np_action = envs.call("get_stable_action",args.stable_policy)[0]
        #         action = torch.Tensor([np_action])
        #         # action = torch.Tensor(np.argmin(buffers)).to(device)
        #         with torch.no_grad():
        #             _, log_prob, _, value = agent.get_action_and_value(next_obs.to(device), action.to(device))
        #             test_interventions[t] = torch.Tensor([1]).to(device)
        #     else:
        #         # ALGO LOGIC: action logic
        #         with torch.no_grad():
        #             action, logprob, _, value = agent.get_action_and_value(next_obs)
        #             test_interventions[t] = torch.Tensor([0]).to(device)
        #     test_actions[t] = action
        #
        #     # TRY NOT TO MODIFY: execute the game and log data.
        #     next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy().astype(int))
        #     done = terminated | truncated
        #     test_rewards[t] = torch.tensor(reward).to(device).view(-1)
        #     next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
        #     test_backlogs[t] = info['backlog'][0]
        #     test_sum_backlogs += info['backlog'][0]
        #     if t > 0 and t % args.test_log_interval == 0:
        #         if t >= args.window_size:
        #             window_averaged_backlog = np.mean(
        #                 test_backlogs[t - args.window_size:t])
        #         else:
        #             window_averaged_backlog = np.mean(test_backlogs[:t])
        #         lta_backlogs = np.cumsum(test_backlogs[:t]) / np.arange(1, t + 1)
        #         wandb.log({"test/lta_backlogs": lta_backlogs[-1],
        #                    "test/window_averaged_backlog": window_averaged_backlog,
        #                    "test_step": t})




    envs.close()
    writer.close()
