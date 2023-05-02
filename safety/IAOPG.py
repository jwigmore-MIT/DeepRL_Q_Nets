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
import roller

from param_extractors import parse_env_json
from buffers import Buffer
from Environments.MultiClassMultiHop import MultiClassMultiHop
from NonDRLPolicies.Backpressure import MCMHBackPressurePolicy
from wrappers import wrap_env
from wandb_funcs import wandb_init

def log_fit_metrics(fit_metrics):
    length = fit_metrics["critic_loss"].shape[0]
    for i in range(length):
        wandb.log({
            "fit/critic_loss": fit_metrics["critic_loss"][i],
            "fit/values": fit_metrics["values"][i],
            "fit/targets": fit_metrics["targets"][i],
            "fit/deviation:": fit_metrics["deviation"][i],
            "fit/fit_steps": i
        })

def run_iaopg(config, agent, env, buffer):

    horizon = config.iaopg.horizon
    num_rollouts = config.iaopg.num_rollouts
    rollout_length = config.iaopg.rollout_length
    best_checkpoint = None
    eps_LTA_reward = -np.inf
    log_def = None
    pbar = tqdm(range(num_rollouts),ncols=80, desc="Training")
    t = 0
    wandb.watch(agent.actor, log_freq = 1)
    wandb.watch(agent.critic, log_freq = 1)

    for eps in pbar:
        rollout = roller.gen_rollout(env, agent,rollout_length , frac = f"{eps + 1}/{num_rollouts}", show_progress=False)
        buffer.add_transitions(data = rollout)
        # log rollout
        log_df, eps_LTA_reward = roller.log_rollouts(rollout, log_df = log_def, policy_name = "IAOPG", glob = "online")
        # update agent
        if eps < 1:
            batch = buffer.get_last_rollout()
            fit_metrics = agent.fit_critic(batch, fit_epochs=config.iaopg.fit_epochs)
            log_fit_metrics(fit_metrics)
            #pbar.update("Fitting Critic to first rollout")

        batch = buffer.get_last_rollout()
        update_result = agent.update(batch)
        update_result.update({"eps": eps})
        update_result.update({"eval_LTA_reward": eps_LTA_reward})

        if best_checkpoint is None or eps_LTA_reward > best_checkpoint["LTA_reward"]:
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



# test for value function estimation



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
    env = wrap_env(base_env, self_normalize_obs = config.env.self_normalize_obs, reward_min = -40)

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



