
# Pyrallis Imports
import pyrallis
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict, dataclass, field

# General Imports
import os
import numpy as np
import torch
from copy import deepcopy

# Custom imports
from param_extractors import parse_env_json
from safety.buffers import Buffer
from Environments.MultiClassMultiHop import MultiClassMultiHop
from NonDRLPolicies.Backpressure import MCMHBackPressurePolicy
from safety.wrappers import wrap_env, reward_wrapper
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
    num_pretrain_rollouts = 1
    horizon:int = 10000
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

def process_prerollout(prerollout, threshold_ratio = 1, rollout_length= 100):
    rollout = deepcopy(prerollout)
    #rollout_length = rollout["rewards"].shape[0]
    min_reward = rollout["rewards"].min()
    q_threshold = -min_reward * threshold_ratio
    # Normalize Rewards to be between (-1 and 1)/rollout_length
    rollout["rewards"] = (rollout["rewards"] - min_reward/2) / (-min_reward/2) + 1e-8 / rollout_length
    return rollout, q_threshold, min_reward

def process_rollout(rollout, agent):
    with torch.no_grad():
        rollout["action_prob"] = np.exp(agent.actor.log_prob(torch.Tensor(rollout["obs"]), torch.Tensor(rollout["actions"])).detach().cpu().numpy())
        #rollout["v_values"] = agent.critic.forward(torch.tensor(rollout["obs"])).detach().cpu().numpy()
    return rollout



if __name__ == "__main__":
    from NonDRLPolicies.Backpressure import MCMHBackPressurePolicy
    from safety.roller import gen_rollout, log_rollouts
    from safety.wandb_funcs import wandb_init
    from safety.loggers import log_rollouts, log_pretrain_metrics

    config = Config()

    # init env
    parse_env_json(config.env.env_json_path, config)
    base_env = MultiClassMultiHop(config=config)
    env = wrap_env(base_env)

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

    num_rollouts = config.iaopg.num_rollouts - config.iaopg.num_pretrain_rollouts
    rollout_length = config.iaopg.rollout_length
    pretrain_rollout_length = rollout_length * config.iaopg.num_pretrain_rollouts
    history = None
    # Pre-train
    agent.pretrain = True
    prerollout = gen_rollout(env, agent, length=pretrain_rollout_length, show_progress=False)
    rollout, q_threshold, min_reward = process_prerollout(prerollout)
    env = reward_wrapper(env, min_reward, rollout_length)
    agent.set_safe_threshold(q_threshold)

    history, eps_lta_reward = log_rollouts(rollout, history = history,  policy_name= "BP", glob = "pretrain", include =  config.logger.include)
    buffer.add_transitions(rollout)
    batch = buffer.get_last_rollout()
    pretrain_metrics = agent.fit_critic(batch, fit_epochs=1000)
    log_pretrain_metrics(pretrain_metrics)
    # Start real training
    pbar = tqdm(range(num_rollouts), ncols=80, desc="Environment Rollouts with BP")
    agent.pretrain = False
    for eps in pbar:
        rollout = gen_rollout(env, agent, length=rollout_length, show_progress=False)
        buffer.add_transitions(rollout)
        rollout = process_rollout(rollout, agent)
        history, eps_lta_reward = log_rollouts(rollout, history = history,  policy_name= "IAOPG", glob = "live", include =  config.logger.include)
        batch = buffer.get_last_rollout()
        update_metrics = agent.update(batch)
        update_metrics.update({"eps": eps})
        wandb.log(update_metrics)
    wandb.finish()

