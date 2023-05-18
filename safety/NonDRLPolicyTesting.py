
# Pyrallis Imports
import pyrallis
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict, dataclass, field

# General Imports
import os
import numpy as np
import torch
from copy import deepcopy
import random
import gymnasium as gym

# Custom imports
from param_extractors import parse_env_json
from safety.buffers import Buffer
from Environments.MultiClassMultiHop import MultiClassMultiHop
from safety.wrappers import flatten_env, standard_reward_wrapper, normalize_obs_wrapper
from safety.agent import Actor, Critic, SafeAgent, Interventioner, BetaActor, MultiDiscreteActor
import wandb

from tqdm import tqdm


@dataclass
class AgentConfig:
    learning_rate: float = 8e-3
    gamma: float = 0.99
    lambda_: float = 0.95
    actor_hidden_dim: int = 64
    critic_hidden_dim: int = 64

@dataclass
class EnvConfig:
    env_json_path: str = "../JSON/Environment/Diamond2.json"
    flat_state_dim: int = None
    flat_action_dim: int = None
    self_normalize_obs: bool = False
    def __post_init__(self):
        self.name = os.path.splitext(os.path.basename(self.env_json_path))[0]

@dataclass
class RunSettings:
    save_freq: int = 1

@dataclass
class TestingConfig:
    rollout_length: int = 1000
    policy = "Backpressure" # "Backpressure




@dataclass
class WandBConfig:
    project: str = "KeepItSimple"
    group: str = "Baseline"
    name: str = "Diamond2-Backpressure"
    checkpoints_path: Optional[str] = None
@dataclass
class LoggerConfig:
    include: List[str] =  field(default_factory=lambda: ["all"])
    type: str = "final" # "final", "all"
@dataclass
class Config:
    device: str = "cpu"
    checkpoints_path: str = "Saved_Models"
    seed: int = 5031997
    deterministic_torch: bool = True


    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    run_settings: RunSettings = field(default_factory=RunSettings)
    testing: TestingConfig = field(default_factory=TestingConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    def print_all(self):
        print("-" * 80)
        print(f"Config: \n{pyrallis.dump(self)}")
        print("-" * 80)
        print("-" * 80)




def process_prerollout(prerollout, threshold_ratio = 1, rollout_length= 100, standardize_reward = False, normalize_obs = False):
    results = {}
    rollout = deepcopy(prerollout)
    #rollout_length = rollout["rewards"].shape[0]
    results["min_reward"] = rollout["rewards"].min()
    results["mean_obs"] = rollout["obs"].mean(axis = 0)
    results["std_obs"] = rollout["obs"].std(axis = 0)
    # Calculate safety threshold
    results["q_threshold"] = max(rollout["backlogs"]) * threshold_ratio
    # Normalize Rewards to be between (-1 and 1)/rollout_length
    if standardize_reward:
        rollout["rewards"] = (rollout["rewards"] - results["min_reward"]/2) / (-results["min_reward"]/2) + 1e-8
    if normalize_obs:
        rollout["obs"] = (rollout["obs"] - results["mean_obs"]) / (results["std_obs"] + 1e-8)

    return rollout, results

def process_rollout(rollout, agent):
    with torch.no_grad():
        rollout["action_prob"] = np.exp(agent.actor.log_prob(torch.Tensor(rollout["obs"]), torch.Tensor(rollout["actions"])).detach().cpu().numpy())
        #rollout["v_values"] = agent.critic.forward(torch.tensor(rollout["obs"])).detach().cpu().numpy()
    return rollout

def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):

    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def create_bias_and_mask(flat_env):
    high = flat_env.action_space.high
    low =  flat_env.action_space.low

    bias = (high+low)/2

    mask_ranges = np.array([low, high])
    return bias, mask_ranges

if __name__ == "__main__":
    from NonDRLPolicies.Backpressure import MCMHBackPressurePolicy
    from NonDRLPolicies.StaticPolicies import DiamondOptimalPolicy
    from safety.roller import gen_rollout, log_rollouts
    from safety.wandb_funcs import wandb_init, log_history
    from safety.loggers import log_rollouts, log_pretrain_metrics, log_update_metrics, log_rollout_summary

    config = Config()

    # init env
    parse_env_json(config.env.env_json_path, config)
    base_env = MultiClassMultiHop(config=config)
    env = flatten_env(base_env)
    bias, mask_ranges = create_bias_and_mask(env)

    # set seed
    set_seed(config.seed, env, config.deterministic_torch)

    # init buffer

    nvec = mask_ranges[1,:] + 1
    actor = MultiDiscreteActor(config.env.flat_state_dim, config.agent.actor_hidden_dim, nvec)
    actor.to(config.device)
    critic = Critic(config.env.flat_state_dim, config.agent.critic_hidden_dim)
    critic.to(config.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.agent.learning_rate)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=config.agent.learning_rate)

    # Initialize Intervention Actor
    if config.testing.policy == "Optimal":
        agent = DiamondOptimalPolicy(env)
    elif config.testing.policy == "Backpressure":
        agent = MCMHBackPressurePolicy(env, M=True)


    # init wandb
    wandb_init(config)


    rollout_length = config.testing.rollout_length
    rollout = gen_rollout(env, agent, length=rollout_length, show_progress=True, safe_agent=False)

    history, eps_lta_reward = log_rollouts(rollout, history=None, policy_name=config.testing.policy, glob="Non-DRL Baseline",
                                           include=config.logger.include)
    agent.pretrain = False



    wandb.finish()

