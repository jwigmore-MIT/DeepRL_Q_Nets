
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
from safety.agent import Actor, Critic, SafeAgent, Interventioner, BetaActor, MultiDiscreteActor, ProbabilisticInterventioner
import wandb

from tqdm import tqdm


@dataclass
class AgentConfig:


    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    actor_hidden_dim: int = 64
    critic_hidden_dim: int = 64

@dataclass
class EnvConfig:
    env_json_path: str = "../JSON/Environment/OnePacket/SixNodeOP.json"
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

    horizon:int = 1000000
    trigger_state: int = 60
    updates_per_rollout: int = 10 #10

    # Pretraining
    num_pretrain_rollouts: int = 0 # Number of rollouts to collect
    pretrain_fit_epochs: int = 10# Number of epochs to fit the critic to the pretrain data
    threshold_ratio: float = 2.0 # what percentage of the max cumulative state encountered should be the safety threshold
    interventioner: str = "Deterministic" # "Deterministic", "Probabilistic", "None
    inter_lambda: float = 2 # if greater than 0, the lambda term for the probabalistic interventioner

    # Modifications
    standardize_reward: bool = False
    normalize_obs: Union[bool, str] = "gym" # "custom", "gym"
    normalize_values: bool = True
    target_update_rate: float = 0.2
    norm_states: bool = False # don't use

    ppo: bool = True
    ppo_clip_coef: float = 0.2
    kl_coef: float = 1
    entropy_coef: float = 5
    kl_target: float = 4
    grad_clip: float = 10

    intervention_penalty: float = 0


    init_std: float = 1.5


    def __post_init__(self):
        self.num_rollouts = self.horizon // self.rollout_length

@dataclass
class WandBConfig:
    project: str = "KeepItSimple"
    group: str = "SafeActor"
    name: str = "SixNodeOP-IAOPG"
    checkpoints_path: Optional[str] = None
@dataclass
class LoggerConfig:
    include: List[str] =  field(default_factory=lambda: ["all"])
    type: str = "final" # "final", "all"

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

def get_action_ranges(flat_env):
    """
    Creates a mid vector i.e. the middle of the action space and an action_ranges vector i.e. the range of the action space
     - mid can be used for gaussian policy parameterizations to bias the policy towards initially trying to take an
       action in the middle of the action space
     - action_ranges can be used to truncate a gaussian policy, or set the range of actions for a Beta or multi-discrete policy
    """
    high = flat_env.action_space.high
    low =  flat_env.action_space.low
    mid = (high+low)/2

    action_ranges = np.array([low, high])
    return mid, action_ranges

if __name__ == "__main__":
    from NonDRLPolicies.Backpressure import MCMHBackPressurePolicy
    from safety.roller import gen_rollout, log_rollouts
    from safety.wandb_funcs import wandb_init, log_history
    from safety.loggers import log_rollouts, log_pretrain_metrics, log_update_metrics, log_rollout_summary

    config = Config()

    # init env
    parse_env_json(config.env.env_json_path, config)
    base_env = MultiClassMultiHop(config=config)
    env = flatten_env(base_env)
    if config.iaopg.normalize_obs == "gym":
        env = gym.wrappers.NormalizeObservation(env)
    mid, action_ranges = get_action_ranges(env)

    # set seed
    set_seed(config.seed, env, config.deterministic_torch)

    # init buffer
    buffer = Buffer(config.env.flat_state_dim, config.env.flat_action_dim, config.buffer_size, config.iaopg.norm_states, config.device)


    actor = MultiDiscreteActor(config.env.flat_state_dim, config.agent.actor_hidden_dim, action_ranges)
    actor.to(config.device)
    critic = Critic(config.env.flat_state_dim, config.agent.critic_hidden_dim)
    critic.to(config.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.agent.learning_rate)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=config.agent.learning_rate)

    # Initialize Intervention Actor
    safe_actor = MCMHBackPressurePolicy(env, M=True)
    if config.iaopg.interventioner == "Probabilistic":
        interventioner = ProbabilisticInterventioner(safe_actor, trigger_state = config.iaopg.trigger_state, lambda_ = config.iaopg.inter_lambda)
    elif config.iaopg.interventioner == "Deterministic":
        interventioner = Interventioner(safe_actor, trigger_state = config.iaopg.trigger_state)

    # Initialize Intervention Agent
    agent = SafeAgent(actor, critic, actor_optim, critic_optim, interventioner,
                      gamma = config.agent.gamma,
                      normalize_values = config.iaopg.normalize_values,
                      target_update_rate=config.iaopg.target_update_rate,
                      gae_lambda = config.agent.gae_lambda,
                      updates_per_rollout=config.iaopg.updates_per_rollout,
                      ppo_clip_coef=config.iaopg.ppo_clip_coef,
                      ppo = config.iaopg.ppo,
                      entropy_coef=config.iaopg.entropy_coef,
                      kl_coef=config.iaopg.kl_coef,
                      kl_target=config.iaopg.kl_target,
                      intervention_penalty=config.iaopg.intervention_penalty,
                      normalized_states = config.iaopg.normalize_obs,
                      grad_clip=config.iaopg.grad_clip,)

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
            # if config.iaopg.standardize_reward:
            #     env = standard_reward_wrapper(env, pretrain_results["min_reward"])
            # if config.iaopg.normalize_obs:
            #     env = normalize_obs_wrapper(env, pretrain_results["mean_obs"], pretrain_results["std_obs"])
            history, eps_lta_reward = log_rollouts(rollout, history=history, policy_name="Pretrain", glob="pretrain",
                                                   include=config.logger.include)
            buffer.add_transitions(rollout)
            batch = buffer.get_last_rollout()
            if config.iaopg.pretrain_fit_epochs > 0:
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
            log_update_metrics(update_metrics, eps, type = config.logger.type)

    # Test final agent
    env.reset()
    rollout = gen_rollout(env, agent, length = 1000)
    test_history, test_lta_reward = log_rollouts(rollout, history = None, policy_name="Final Agent", glob = "test")
    log_rollout_summary(rollout, 0, glob = "test")
    log_history(history)

    wandb.finish()

