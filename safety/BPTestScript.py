
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
from safety.agents.normalizers import Normalizer, CriticTargetScaler



if __name__ == "__main__":

    # === Init Config === #
    config_file = "Backpressure/BP_Test1.yaml"
    config = parse_config(config_file, run_type="TEST")

    # === Init Environment === #
    env = generate_env(config)

    # ===Load Agent ===#
    agent = MCMHBackPressurePolicy(env, M = True)

    # init wandb
    wandb_init(config)


    env.reset()
    rollout = gen_rollout(env, agent, length=2000)
    test_history, test_lta_reward = log_rollouts(rollout, glob="test")
    log_rollout_summary(rollout, 0, glob="test")

    wandb.finish()


