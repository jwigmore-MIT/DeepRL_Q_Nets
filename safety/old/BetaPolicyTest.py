
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
from NonDRLPolicies.Backpressure import MCMHBackPressurePolicy
from safety.wrappers import flatten_env, standard_reward_wrapper, normalize_obs_wrapper
from safety.agent import Actor, Critic, SafeAgent, Interventioner, BetaActor
import wandb

from tqdm import tqdm


@dataclass
class AgentConfig:
    learning_rate: float = 3e-5
    gamma: float = 0.99
    lambda_: float = 0.95
    actor_hidden_dim: int = 64
    critic_hidden_dim: int = 64

@dataclass
class EnvConfig:
    env_json_path: str = "../JSON/Environment/Env1b.json"
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

    horizon:int = 10000
    trigger_state: int = None
    updates_per_rollout: int = 10 #10

    # Pretraining
    num_pretrain_rollouts = 1 # Number of rollouts to collect
    pretrain_fit_epochs = 100 # Number of epochs to fit the critic to the pretrain data
    threshold_ratio = 1 # what percentage of the max cumulative state encountered should be the safety threshold

    # Modifications
    standardize_reward: bool = False
    normalize_obs: Union[bool, str] = False # "custom", "gym"
    normalize_values: bool = True

    ppo: bool = True
    ppo_clip_coef: float = 0.2

    init_std: float = 1.5


    def __post_init__(self):
        self.num_rollouts = self.horizon // self.rollout_length

@dataclass
class WandBConfig:
    project: str = "InterventionAssistedOnlinePolicyGradient"
    group: str = "PPOBeta"
    name: str = "Env1b-PPOBeta"
    checkpoints_path: Optional[str] = None
@dataclass
class LoggerConfig:
    include: List[str] =  field(default_factory=lambda: ["all"])
@dataclass
class Config:
    device: str = "cpu"
    buffer_size: int = 2_000_000  # Replay Buffer
    checkpoints_path: str = "Saved_Models"
    seed: int = 5031997
    deterministic_torch: bool = True


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

    def update_trigger_state(self, trigger_state):
        self.iaopg.trigger_state = trigger_state
        if wandb.run is not None:
            wandb.run.config.update({"iaopg": self.iaopg}, allow_val_change=True)


def process_prerollout(prerollout, threshold_ratio = 1, rollout_length= 100, standardize_reward = False, normalize_obs = False):
    results = {}
    rollout = deepcopy(prerollout)
    #rollout_length = rollout["rewards"].shape[0]
    results["min_reward"] = rollout["rewards"].min()
    results["mean_obs"] = rollout["obs"].mean(axis = 0)
    results["std_obs"] = rollout["obs"].std(axis = 0)
    # Calculate safety threshold
    results["q_threshold"] = -results["min_reward"] * threshold_ratio
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
    from safety.roller import gen_rollout, log_rollouts
    from safety.wandb_funcs import wandb_init
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
    buffer = Buffer(config.env.flat_state_dim, config.env.flat_action_dim, config.buffer_size, config.device)

    # initialize actor, critic, optimizers, and agent
    # actor = Actor(config.env.flat_state_dim, config.env.flat_action_dim, config.agent.actor_hidden_dim,
    #               init_std=config.iaopg.init_std,
    #               min_actions= env.action_space.low,
    #               max_actions= env.action_space.high,
    #               bias = bias,
    #               mask_ranges = mask_ranges)
    actor = BetaActor(config.env.flat_state_dim, config.env.flat_action_dim, config.agent.actor_hidden_dim, mask_ranges)
    actor.to(config.device)
    critic = Critic(config.env.flat_state_dim, config.agent.critic_hidden_dim)
    critic.to(config.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.agent.learning_rate)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=config.agent.learning_rate)

    # Initialize Intervention Actor
    safe_actor = MCMHBackPressurePolicy(env, M=True)
    interventioner = Interventioner(safe_actor, trigger_state = config.iaopg.trigger_state)

    # Initialize Intervention Agent
    agent = SafeAgent(actor, critic, actor_optim, critic_optim, interventioner,
                      gamma = config.agent.gamma,
                      normalize_values = config.iaopg.normalize_values,
                      lambda_ = config.agent.lambda_,
                      updates_per_rollout=config.iaopg.updates_per_rollout,
                      ppo_clip_coef=config.iaopg.ppo_clip_coef,
                      ppo = config.iaopg.ppo)

    # init wandb
    wandb_init(config)

    num_rollouts = config.iaopg.num_rollouts
    rollout_length = config.iaopg.rollout_length
    history = None
    # Pre-train
    # Start real training
    pbar = tqdm(range(num_rollouts), ncols=80, desc="Environment Rollouts with BP")
    agent.pretrain = False
    for eps in pbar:
        if eps < config.iaopg.num_pretrain_rollouts:
            agent.pretrain = True # so we use the safe actor in pretraining
            pretrain_rollout = gen_rollout(env, agent, length=rollout_length, show_progress=False)
            rollout, pretrain_results = process_prerollout(pretrain_rollout,
                                                    threshold_ratio = config.iaopg.threshold_ratio,
                                                    rollout_length=rollout_length,
                                                    standardize_reward = config.iaopg.standardize_reward)
            if True:
                config.update_trigger_state(pretrain_results["q_threshold"])
                agent.set_safe_threshold(pretrain_results["q_threshold"])
            # Wrap env based on pretrain if needed
            if config.iaopg.standardize_reward:
                env = standard_reward_wrapper(env, pretrain_results["min_reward"])
            if config.iaopg.normalize_obs:
                env = normalize_obs_wrapper(env, pretrain_results["mean_obs"], pretrain_results["std_obs"])
            history, eps_lta_reward = log_rollouts(rollout, history=history, policy_name="Pretrain", glob="pretrain",
                                                   include=config.logger.include)
            buffer.add_transitions(rollout)
            batch = buffer.get_last_rollout()
            pretrain_metrics = agent.fit_critic(batch, fit_epochs=config.iaopg.pretrain_fit_epochs)
            log_pretrain_metrics(pretrain_metrics)
            agent.pretrain = False
        else:
            rollout = gen_rollout(env, agent, length=rollout_length, show_progress=False)
            buffer.add_transitions(rollout)
            rollout = process_rollout(rollout, agent)
            history, eps_lta_reward = log_rollouts(rollout, history = history,  policy_name= "IAOPG", glob = "live", include =  config.logger.include)
            log_rollout_summary(rollout,eps, glob = "rollout_summary")
            batch = buffer.get_last_rollout()
            update_metrics = agent.update(batch)
            log_update_metrics(update_metrics, eps)

    # Test final agent
    env.reset()
    rollout = gen_rollout(env, agent, length = 1000)
    test_history, test_lta_reward = log_rollouts(rollout, history = None, policy_name="Final Agent", glob = "test")
    log_rollout_summary(rollout, 0, glob = "test")

    wandb.finish()

