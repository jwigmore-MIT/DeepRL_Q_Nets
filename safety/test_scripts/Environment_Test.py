
# Pyrallis Imports
import pyrallis
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict, dataclass, field

# General Imports
import os
import numpy as np
import torch

# Custom imports
from param_extractors import parse_env_json
from safety.buffers import Buffer
from Environments.MultiClassMultiHop import MultiClassMultiHop
from NonDRLPolicies.Backpressure import MCMHBackPressurePolicy
from safety.wrappers import wrap_env
from safety.agent import Actor, Critic, SafeAgent, Interventioner
import wandb

from tqdm import tqdm


@dataclass
class AgentConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    actor_hidden_dim: int = 64
    critic_hidden_dim: int = 64

@dataclass
class EnvConfig:
    env_json_path: str = "../../JSON/Environment/OneHop1a.json"
    flat_state_dim: int = None
    flat_action_dim: int = None
    self_normalize_obs: bool = False
    def __post_init__(self):
        self.name = os.path.splitext(os.path.basename(self.env_json_path))[0]

@dataclass
class RunSettings:
    save_freq: int = 1

@dataclass
class IAOPGConfig:
    rollout_length: int = 100
    horizon:int = 200
    trigger_state: int = 30


    def __post_init__(self):
        self.num_rollouts = self.horizon // self.rollout_length

@dataclass
class WandBConfig:
    project: str = "EnvironmentTest"
    group: str = "OneHop"
    name: str = "BP-OneHop"
    checkpoints_path: Optional[str] = "../Saved_Models/BP/"
@dataclass
class LoggerConfig:
    include: List[str] =  field(default_factory=lambda: ["all"])
@dataclass
class Config:
    device: str = "cpu"
    buffer_size: int = 2_000_000  # Replay Buffer
    checkpoints_path: str = "Saved_Models"

    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    run_settings: RunSettings = field(default_factory=RunSettings)
    iaopg: IAOPGConfig = field(default_factory=IAOPGConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    def print_all(self):
        print("-" * 80)
        print(f"Config: \n{pyrallis.dump(self)}")
        print("-" * 80)
        print("-" * 80)


if __name__ == "__main__":
    from NonDRLPolicies.Backpressure import MCMHBackPressurePolicy
    from safety.roller import gen_rollout, log_rollouts
    from safety.wandb_funcs import wandb_init

    config = Config()

    # init env
    parse_env_json(config.env.env_json_path, config)
    base_env = MultiClassMultiHop(config=config)
    env = wrap_env(base_env, self_normalize_obs=config.env.self_normalize_obs, reward_min=-40)

    # init buffer
    buffer = Buffer(config.env.flat_state_dim, config.env.flat_action_dim, config.buffer_size, config.device)

    # initialize actor, critic, optimizers, and agent
    actor = Actor(config.env.flat_state_dim, config.env.flat_action_dim, config.agent.actor_hidden_dim)
    actor.to(config.device)
    critic = Critic(config.env.flat_state_dim, config.agent.critic_hidden_dim)
    critic.to(config.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.agent.learning_rate)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=config.agent.learning_rate)

    # Initialize Intervention Actor
    safe_actor = MCMHBackPressurePolicy(env, M=True)
    trigger_state = 0 # So BP is always used
    interventioner = Interventioner(safe_actor, trigger_state)

    # Initialize Intervention Agent
    agent = SafeAgent(actor, critic, actor_optim, critic_optim, interventioner, config.agent.gamma)

    # init wandb
    wandb_init(config)

    num_rollouts = config.iaopg.num_rollouts
    rollout_length = config.iaopg.rollout_length
    pbar = tqdm(range(num_rollouts), ncols=80, desc="Environment Rollouts with BP")
    history = None
    for eps in pbar:
        rollout = gen_rollout(env, agent, length=rollout_length, show_progress=False)
        buffer.add_transitions(rollout)
        history, eps_lta_reward = log_rollouts(rollout, history = history,  policy_name= "BP", glob = "live", include =  config.logger.include)

    wandb.finish()

