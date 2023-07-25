
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



if __name__ == "__main__":

    # === Init Config === #
    config_file = "noncontinuing/SafePPO-Discrete-JSQN4AS.yaml"
    config = parse_config(config_file, run_type="TEST")

    # === Init Environment === #
    env = generate_env(config)

    # ===Load Agent ===#
    load_path = "../safety/wandb/run-20230721_154836-5c67902c-95c7-462d-becc-cc80de06d2e7/files/agent_best.pkl"
    agent = pickle.load(open(load_path, "rb"))
    agent.force_nn = True
    agent.change_mode(mode = "test")
    # init wandb
    wandb_init(config)

    test_multiple = False
    view_multiple = False
    env.reset(seed = config.seed)
    if test_multiple:
        for eps in range(10):
            rollout = gen_rollout(env, agent, length=1000)
            if view_multiple:
                test_history, test_lta_reward = log_rollouts(rollout, policy_name=f"Test eps{eps}", glob="test")
            log_rollout_summary(rollout, eps, glob="test")
            env.reset(seed = config.seed)
    else:
        rollout = gen_rollout(env, agent, length=10000)
        test_history, test_lta_reward = log_rollouts(rollout, glob="test")
        log_rollout_summary(rollout, 0, glob="test")
        env.reset(seed=config.seed)
    wandb.finish()


