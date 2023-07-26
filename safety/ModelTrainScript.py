
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
from safety.wandb_funcs import wandb_init, save_agent_wandb, load_agent_wandb
from safety.loggers import log_rollouts, log_update_metrics, log_rollout_summary
from safety.utils import *
from Environments.MCMH_tools import generate_env
from safety.agents.actors import init_actor
from safety.agents.critics import Critic
from safety.agents.ppo_agent import PPOAgent
from safety.agents.lta_ppo_agent import LTAPPOAgent
from safety.agents.normalizers import MovingNormalizer, CriticTargetScaler, FixedNormalizer, MovingNormalizer2
from safety.agents.safe_agents import init_safe_agent



if __name__ == "__main__":

    # === Init Config === #
    #config_file = "PPO-Gaussian-Env1b.yaml"
    #config_file = "SafePPO-Gaussian-Env1b.yaml"
    #config_file = "continuing/SafeLTAPPO-Gaussian-Env1b.yaml"
    #config_file = "PPO-TanGaussian-Env1b.yaml"
    #config_file = "SafePPO-TanGaussian-Env1b.yaml"
    config_file = "continuing/SafeLTAPPO-Discrete-JSQN2S3.yaml"
    #config_file = "noncontinuing/SafePPO-Discrete-JSQN4.yaml"
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
    actor, actor_optim = init_actor(config, mid, action_ranges)
    actor.to(config.device)
    critic = Critic(config.env.flat_state_dim, **config.agent.critic.kwargs.toDict())
    critic.to(config.device)
    #actor_optim = torch.optim.Adam(actor.parameters(), lr=config.agent.actor.learning_rate)

    critic_optim = torch.optim.Adam(critic.parameters(), lr=config.agent.critic.learning_rate)

    # initialize obs_normalizer and target_scaler
    #obs_normalizer = MovingNormalizer(config.env.flat_state_dim, config.normalizers.obs.eps)
    obs_normalizer = FixedNormalizer(config.env.flat_state_dim, config.normalizers.obs.norm_factor)
    #obs_normalizer = MovingNormalizer2(config.env.flat_state_dim, config.normalizers.obs.eps, buffer_size = 128, beta = 0.2)
    if hasattr(config.normalizers, "target") and config.normalizers.target.update_rate is not None:
        target_scaler = CriticTargetScaler(config.env.flat_state_dim, config.normalizers.target.update_rate, config.normalizers.target.eps)
    else:
        target_scaler = None
    # if config.agent.lta_agent:
    #     target_scaler = None


    # Initialize Neural Agent
    if hasattr(config.agent, "lta_agent") and config.agent.lta_agent:
        agent = LTAPPOAgent(actor, critic, actor_optim, critic_optim, obs_normalizer = obs_normalizer, target_scaler = target_scaler,
                          **config.agent.kwargs.toDict())
    else:
        agent = PPOAgent(actor, critic, actor_optim, critic_optim, obs_normalizer = obs_normalizer, target_scaler = target_scaler,
                          **config.agent.kwargs.toDict())

    if hasattr(config.agent, "safety"):
        agent = init_safe_agent(config.agent.safety, agent, env)



    # init wandb
    wandb_init(config)
    history = None
    pbar = tqdm(range(config.train.num_episodes), ncols=80, desc="Training Episodes")
    artifact = wandb.Artifact(config.artifact_name, type="agent")
    best_lta_backlog = np.inf
    history = None
    if config.train.pretrain_steps > 0:
        rollout = gen_rollout(env, agent, length = config.train.pretrain_steps, show_progress=False, reset = config.train.reset)
        buffer.add_transitions(rollout)
        rollout = process_rollout(rollout, agent)
        eps_lta_backlog = log_rollout_summary(rollout, 0, glob="pretrain")
        history, _ = log_rollouts(rollout, glob="Live Rollout", history=history)
        batch = buffer.get_last_rollout()
        update_metrics = agent.update(batch, pretrain = True)
        log_update_metrics(update_metrics, 0, glob="pretrain", type = "minibatches")

    for eps in pbar:
        rollout = gen_rollout(env, agent, length = config.train.batch_size, show_progress=False, reset = config.train.reset)
        buffer.add_transitions(rollout)
        rollout = process_rollout(rollout, agent)
        eps_lta_backlog = log_rollout_summary(rollout, eps, glob="rollout_summary")
        if eps_lta_backlog < best_lta_backlog:
            best_lta_backlog = eps_lta_backlog
            if eps >= 50:
                save_agent_wandb(agent, mod = "_best")
        if not config.train.reset:
            history, _ = log_rollouts(rollout, glob="Live Rollout", history=history)
        batch = buffer.get_last_rollout()
        update_metrics = agent.update(batch)
        log_update_metrics(update_metrics, eps, glob="update_metrics", type = "minibatches")

    #save_agent(agent, config.save_dir, mod = "_final")
    #save_config(config, config.save_dir)

    env.reset()
    agent.change_mode(mode = "test")
    rollout = gen_rollout(env, agent, length=50000)
    rollout = process_rollout(rollout, agent)
    test_history, test_lta_reward = log_rollouts(rollout, glob="test")
    log_rollout_summary(rollout, 0, glob="test")

    wandb.finish()


