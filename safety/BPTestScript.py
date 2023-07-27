
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
import pickle

# Custom imports
from safety.buffers import Buffer
from NonDRLPolicies.Backpressure import MCMHBackPressurePolicy
from safety.roller import gen_rollout, log_rollouts
from safety.wandb_funcs import wandb_init, load_agent_wandb
from safety.loggers import log_rollouts, log_update_metrics, log_rollout_summary
from safety.utils import parse_config, get_action_ranges, set_seed, process_rollout
from Environments.MCMH_tools import generate_env
from safety.agents.actors import init_actor
from safety.agents.critics import Critic
from safety.agents.ppo_agent import PPOAgent
from safety.agents.normalizers import MovingNormalizer, CriticTargetScaler
from safety.utils import visualize_buffer



if __name__ == "__main__":

    # === Init Config === #
    config_file = "Backpressure/BPR_Test1.yaml"
    config = parse_config(config_file, run_type="TEST")

    # === Init Environment === #
    env = generate_env(config)

    # ===Load Agent ===#
    agent = MCMHBackPressurePolicy(env, **config.agent.BP_args.toDict())

    # init wandb
    wandb_init(config)


    env.reset()
    test_history = None

    pbar = tqdm(range(config.eval.n_eval_episodes), desc=f"Testing episodes")
    for eps in pbar:
        rollout = gen_rollout(env, agent, length=config.eval.episode_length, reset=False, show_progress=False)
        test_history, test_lta_reward = log_rollouts(rollout, glob="test", history=test_history)
        log_rollout_summary(rollout, eps, glob="test")

    wandb.finish()


