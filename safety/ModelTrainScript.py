
# Pyrallis Imports
import pyrallis
from typing import Any, Dict, List, Optional, Tuple, Union

# General Imports
import os
import numpy as np
import torch
from copy import deepcopy
import gymnasium as gym
import wandb
from tqdm import tqdm

# Custom imports
from safety.buffers import Buffer
from NonDRLPolicies.Backpressure import MCMHBackPressurePolicy
from safety.roller import gen_rollout, log_rollouts
from safety.wandb_funcs import wandb_init, save_agent_wandb
from safety.loggers import log_rollouts, log_update_metrics, log_rollout_summary
from safety.utils import *
from Environments.MCMH_tools import generate_env
from safety.agents.actors import init_actor
from safety.agents.critics import Critic
from safety.agents.ppo_agent import PPOAgent
from safety.agents.normalizers import Normalizer, CriticTargetScaler
from safety.agents.safe_agents import init_safe_agent



if __name__ == "__main__":

    # === Init Config === #
    config_file = "PPO-Gaussian-Env2a.yaml"
    config_file = "SafePPO-Gaussian-Env1b.yaml"
    config = parse_config(config_file)

    # === Init Environment === #
    env = generate_env(config)

    # === Get Action Range Information === #
    mid, action_ranges = get_action_ranges(env)
    # if config.agent.actor.type == "TanGaussian":
    #     config.agent.actor.kwargs.action_mids = mid

    # set seed
    set_seed(config.seed, env, config.deterministic_torch)

    # init buffer
    buffer = Buffer(config.env.flat_state_dim, config.env.flat_action_dim, config.buffer.size, config.device)

    # initialize actor and critic
    actor = init_actor(config, mid, action_ranges)
    actor.to(config.device)
    critic = Critic(config.env.flat_state_dim, config.agent.critic.hidden_dim)
    critic.to(config.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.agent.actor.learning_rate)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=config.agent.critic.learning_rate)

    # initialize obs_normalizer and target_scaler
    obs_normalizer = Normalizer(config.env.flat_state_dim, config.normalizers.obs.eps)
    target_scaler = CriticTargetScaler(config.env.flat_state_dim, config.normalizers.target.update_rate, config.normalizers.target.eps)



    # Initialize Neural Agent
    agent = PPOAgent(actor, critic, actor_optim, critic_optim, obs_normalizer = obs_normalizer, target_scaler = target_scaler,
                      **config.agent.kwargs.toDict())

    if hasattr(config.agent, "safety"):
        agent = init_safe_agent(config.agent.safety, agent, env)



    # init wandb
    wandb_init(config)
    history = None
    pbar = tqdm(range(config.train.num_episodes), ncols=80, desc="Training Episodes")
    artifact = wandb.Artifact(config.artifact_name, type="agent")
    for eps in pbar:
        rollout = gen_rollout(env, agent, length = config.train.batch_size, show_progress=False, reset = True)
        buffer.add_transitions(rollout)
        rollout = process_rollout(rollout, agent)
        eps_lta_reward = log_rollout_summary(rollout, eps, glob="rollout_summary")
        batch = buffer.get_last_rollout()
        update_metrics = agent.update(batch)
        log_update_metrics(update_metrics, eps, glob="update_metrics", type = "minibatches")

    save_agent(agent, config.save_dir, mod = "_final")
    save_config(config, config.save_dir)

    env.reset()
    rollout = gen_rollout(env, agent, length=1000)
    test_history, test_lta_reward = log_rollouts(rollout, glob="test")
    log_rollout_summary(rollout, 0, glob="test")

    wandb.finish()


