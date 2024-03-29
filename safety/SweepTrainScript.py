
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
from safety.agents.lta_ppo_agent import LTAPPOAgent
from safety.agents.normalizers import MovingNormalizer, CriticTargetScaler, FixedNormalizer, MovingNormalizer2
from safety.agents.safe_agents import init_safe_agent


sweep_config = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {
        'name': 'lta_backlog',
        'goal': 'minimize'
    },
    'parameters': {
        'actor_lr': {'max': 1e-3, 'min': 1e-5},
        'critic_lr': {'max': 1e-3, 'min': 1e-5},
        'alpha': {'max': 0.5, 'min': 0.0},
        'nu': {'values': [1, 0.5, 0.25, 0.1, 0]},
        'gae_lambda': {'max': 0.99, 'min': 0.25},
        'kl_target': {'max': 1, 'min': 0.001},
        'batch_size': {'values': [128, 256, 512]}
    }
}


def run_sweep(sweep_config, base_config_file):
    # get init config
    config = parse_config(base_config_file)

    # Modify config for sweep
    config.agent.actor.learning_rate = sweep_config["actor_lr"]
    config.agent.critic.learning_rate = sweep_config["critic_lr"]
    config.agent.kwargs.alpha = sweep_config["alpha"]
    config.agent.kwargs.nu = sweep_config["nu"]
    config.agent.kwargs.gae_lambda = sweep_config["gae_lambda"]
    config.agent.kwargs.kl_target = sweep_config["kl_target"]
    config.train.batch_size = sweep_config["batch_size"]



    return lta_backlog

def run_trial(config):

def main():
    wandb.init(project='my-first-sweep')
    lta_reward = run_sweep(wandb.config)
    wandb.log({'lta_reward': lta_reward})


if __name__ == "__main__":

    # === Init Config === #
    #config_file = "PPO-Gaussian-Env1b.yaml"
    #config_file = "SafePPO-Gaussian-Env1b.yaml"
    #config_file = "continuing/SafeLTAPPO-Gaussian-Env1b.yaml"
    #config_file = "PPO-TanGaussian-Env1b.yaml"
    #config_file = "SafePPO-TanGaussian-Env1b.yaml"
    config_file = "continuing/SafeLTAPPO-Discrete-JSQN4.yaml"
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
    target_scaler = CriticTargetScaler(config.env.flat_state_dim, config.normalizers.target.update_rate, config.normalizers.target.eps)
    if config.agent.lta_agent:
        target_scaler = None


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
    history = None
    for eps in pbar:
        rollout = gen_rollout(env, agent, length = config.train.batch_size, show_progress=False, reset = config.train.reset)
        buffer.add_transitions(rollout)
        rollout = process_rollout(rollout, agent)
        eps_lta_reward = log_rollout_summary(rollout, eps, glob="rollout_summary")
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


