"""
Intervention Aided Online Policy Gradient Algorithm
"""
from agent import Actor, Critic, SafeAgent, Interventioner
from config import Config
import wandb
import torch
import numpy as np
import pandas as pd
import gymnasium as gym
import uuid
from tqdm import tqdm
from copy import deepcopy
import os

from param_extractors import parse_env_json
from buffers import Buffer
from Environments.MultiClassMultiHop import MultiClassMultiHop
from NonDRLPolicies.Backpressure import MCMHBackPressurePolicy
from wrappers import wrap_env

def gen_rollout(env, agent, length, device = "cpu", frac = "", show_progress = True):

    # Initialize temporary storage
    obs = np.zeros([length, env.observation_space.shape[0]])
    next_obs = np.zeros([length, env.observation_space.shape[0]])
    rewards = np.zeros([length, 1])
    terminals = np.zeros([length, 1])
    interventions = np.zeros([length, 1])
    timeouts = np.zeros([length, 1])
    actions = np.zeros([length, env.action_space.shape[0]])
    flows = np.zeros([length, env.action_space.shape[0]])
    arrivals = np.zeros([length, env.flat_qspace_size])

    # Get the current state of the environment
    next_ob = env.get_f_state()

    if show_progress:
        pbar = tqdm(range(length), desc=f"Generating Rollout {frac}")
    else:
        pbar = range(length)

    # Perform Rollout
    for t in pbar:
        obs[t] = next_ob
        actions[t], interventions[t] = agent.act(obs[t], device)
        next_obs[t], rewards[t], terminals[t], timeouts[t], info = env.step(actions[t])
        if "final_info" in info:
            # info won't contain flows nor arrivals
            pass
        else:
            flows[t] = info['flows'][-1]
            arrivals[t] = info['arrivals'][-1]
        next_ob = next_obs[t]
    terminals[t] = 1
    return {
        "obs": obs,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
        "next_obs": next_obs,
        "flows": flows,
        "arrivals": arrivals,
        "interventions": interventions
    }

def get_reward_stats(reward_vec):
    sum_average_rewards = np.array(
        [np.cumsum(reward_vec[:,i], axis = 0)/np.arange(1,reward_vec.shape[0]+1)
         for i in range(reward_vec.shape[1])]).T

    time_averaged_rewards = sum_average_rewards.mean(axis = 1)
    time_averaged_errors = sum_average_rewards.std(axis=1)
    return time_averaged_rewards, time_averaged_errors
def log_rollouts(rollout, log_df = None,  policy_name = "policy", glob = "test", hidden = False):
    if log_df is None:
        log_df = pd.DataFrame(rollout["rewards"], columns= ["rewards"])
        wandb.define_metric(f"{glob}/step")
        wandb.define_metric(f"{glob}/LTA - {policy_name}", summary = "mean", step_metric = f"{glob}/step", hidden = hidden)
        start_step = 0
    else:
        new_df = pd.DataFrame(rollout["rewards"], columns=["rewards"])
        # extend log_df with new_df
        log_df = pd.concat([log_df, new_df], axis = 0).reset_index(drop = True)
        start_step = log_df.shape[0] - new_df.shape[0]
    log_df["LTA_Rewards"], log_df["LTA_Error"] = get_reward_stats(np.array(log_df["rewards"]).reshape(-1,1))
    for i in range(start_step, log_df.shape[0]):
        log_dict = {
            f"{glob}/step": i,
            f"{glob}/LTA - {policy_name}": log_df["LTA_Rewards"].loc[i],
        }
        wandb.log(log_dict)
    eps_LTA_reward = log_df["LTA_Rewards"].iloc[-1]
    return log_df, eps_LTA_reward
def run_iaopg(config, agent, env, buffer):

    horizon = config.iaopg.horizon
    num_rollouts = config.iaopg.num_rollouts
    rollout_length = config.iaopg.rollout_length
    best_checkpoint = None
    eps_LTA_reward = -np.inf
    log_def = None
    pbar = tqdm(range(num_rollouts),ncols=80, desc="Training")
    t = 0

    for eps in pbar:
        rollout = gen_rollout(env, agent,rollout_length , frac = f"{eps + 1}/{num_rollouts}")
        buffer.add_transitions(data = rollout)
        # log rollout
        log_df, eps_LTA_reward = log_rollouts(rollout, log_df = log_def, policy_name = "IAOPG", glob = "train")
        # update agent
        update_result = agent.update(buffer.get_last_rollout())
        update_result.update({"eps": eps})
        update_result.update({"eval_LTA_reward": eps_LTA_reward})

        if best_checkpoint is not None or eps_LTA_reward > best_checkpoint["LTA_reward"]:
            agent.load_checkpoint(best_checkpoint)
            best_checkpoint = {
                "eps": eps,
                "LTA_reward": eps_LTA_reward,
                "agent": deepcopy(agent),
            }
            # Save best checkpoint
            if config.checkpoints_path is not None:
                torch.save(
                    best_checkpoint["agent"].state_dict(),
                    os.path.join(config.checkpoints_path, "online_best.pt"),)

        if eps % config.run_settings.save_freq == 0:
            if config.checkpoints_path is not None:
                torch.save(
                    agent.state_dict(),
                    os.path.join(config.checkpoints_path, f"eps_{eps}.pt")
                )
        wandb.log(update_result)

    return agent, best_checkpoint, buffer
def wandb_init(config) -> None:
    """Initialize wandb."""
    run = wandb.init(
        config=vars(config),
        project=config.wandb.project,
        group=config.wandb.group,
        name=config.wandb.name,
        id=str(uuid.uuid4()),
    )

RUN_SETTINGS = {}

if __name__ == "__main__":

    print(f"Running with settings: {RUN_SETTINGS}")

    ## Get configuration parameters
    config = Config()
    # set each attribute of config.run based on RUN_SETTINGS
    for k, v in RUN_SETTINGS.items(): setattr(config.run, k, v)
    config.print_all()

    # initialize environment
    parse_env_json(config.env.env_json_path, config)
    base_env = MultiClassMultiHop(config = config)
    env = wrap_env(base_env, state_mean=None, state_std=None)

    # initialize replay buffer
    buffer = Buffer(config.env.flat_state_dim, config.env.flat_action_dim, config.buffer_size, config.device)

    # initialize actor, critic, optimizers, and agent
    actor = Actor(config.env.flat_state_dim, config.env.flat_action_dim, config.agent.actor_hidden_dim)
    actor.to(config.device)
    critic = Critic(config.env.flat_state_dim, config.agent.critic_hidden_dim)
    critic.to(config.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.agent.learning_rate)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=config.agent.learning_rate)

    # Initialize Intervention Actor
    safe_actor = MCMHBackPressurePolicy(env, M = True)
    trigger_state = config.iaopg.trigger_state
    interventioner = Interventioner(safe_actor, trigger_state)


    # Initialize Intervention Agent
    agent = SafeAgent(actor, critic, actor_optim, critic_optim, interventioner, config.agent.gamma)

    # Initialize wandb
    wandb_init(config)

    # run IAOPG
    run_iaopg(config, agent, env, buffer)



