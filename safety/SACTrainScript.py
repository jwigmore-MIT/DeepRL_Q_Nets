
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
from safety.roller import gen_rollout, log_rollouts, gen_step
from safety.wandb_funcs import wandb_init, save_agent_wandb
from safety.loggers import log_rollouts, log_update_metrics, log_rollout_summary
from safety.utils import *
from Environments.MCMH_tools import generate_env
from safety.agents.actors import init_actor
from safety.agents.critics import init_critic
from safety.agents.ppo_agent import PPOAgent
from safety.agents.normalizers import MovingNormalizer, CriticTargetScaler
from safety.agents.safe_agents import init_safe_agent
from safety.agents.sac_agent import SACAgent

"""
Need to add SAC configuration file and then start testing and debugging
Then move onto the safety stuff
"""

if __name__ == "__main__":

    # === Init Config === #
    config_file = "PPO-Gaussian.yaml"
    config_file = "SafePPO-Gaussian.yaml"
    config = parse_config(config_file)

    # === Init Environment === #
    env = generate_env(config, max_steps=config.env.max_steps)

    # === Get Action Range Information === #
    mid, action_ranges = get_action_ranges(env)

    # set seed
    set_seed(config.seed, env, config.deterministic_torch)

    # init buffer
    buffer = Buffer(config.env.flat_state_dim, config.env.flat_action_dim, config.buffer.size, config.device)

    # initialize actor and critic
    actor = init_actor(config, mid, action_ranges)
    actor.to(config.device)
    qf1, qf2, qf1_target, qf2_target = init_critic(config)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.actor.learning_rate)
    q_optim = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=config.critic.learning_rate)
    # initialize obs_normalizer and target_scaler
    obs_normalizer = MovingNormalizer(config.env.flat_state_dim, config.normalizers.obs.eps)
    #target_scaler = CriticTargetScaler(config.env.flat_state_dim, config.normalizers.target.update_rate, config.normalizers.target.eps)



    # Initialize Neural Agent
    agent = SACAgent(actor,
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
                    tau = .005)

    if hasattr(config.agent, "safety"):
        agent = init_safe_agent(config.agent.safety, agent, env)



    # init wandb
    wandb_init(config)
    history = None
    pbar = tqdm(range(config.train.num_steps), ncols=80, desc="Training Steps")
    artifact = wandb.Artifact(config.artifact_name, type="agent")
    random = True
    for t in pbar:
        step = gen_step(env, agent, random = random)
        buffer.add_transitions(step)
        if t >= config.train.learning_starts:
            random = False
            batch = buffer.sample()
            update_metrics = agent.update(batch)

            if t>= config.train.log_interval:
                log_update_metrics(update_metrics, t, glob="update_metrics", type = "steps")



    save_agent(agent, config.save_dir, mod = "_final")
    save_config(config, config.save_dir)

    env.reset()
    rollout = gen_rollout(env, agent, length=1000)
    test_history, test_lta_reward = log_rollouts(rollout, glob="test")
    log_rollout_summary(rollout, 0, glob="test")

    wandb.finish()


