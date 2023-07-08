
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
import pickle
from datetime import datetime

# Custom imports
from param_extractors import parse_env_json
from safety.buffers import Buffer
from Environments.MultiClassMultiHop import MultiClassMultiHop
from NonDRLPolicies.Backpressure import MCMHBackPressurePolicy
from safety.wrappers import flatten_env, standard_reward_wrapper, normalize_obs_wrapper
from safety.AWAC_Agent import Critic, SafeAWACAgent, Interventioner, MultiDiscreteActor, ProbabilisticInterventioner
import wandb

from tqdm import tqdm

OFFLINE_TRAIN = True
ONLINE_TRAIN = True
@dataclass
class AgentConfig:
    # Optimizer Settings
    learning_rate: float = 1e-4

    # Discounting rewards and advantages
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # NN Architecture
    actor_hidden_dim: int = 64
    critic_hidden_dim: int = 64

    # Safety Settings
    trigger_state: int = 20
    threshold_ratio: float = 2.0 # what percentage of the max cumulative state encountered should be the safety threshold

    #Update Settings
    updates_per_rollout: int = 10 #10
    ppo: bool = True
    ppo_clip_coef: float = 0.2
    kl_coef: float = 0.0
    entropy_coef: float = 0
    kl_target: float = 1.0
    grad_clip: float = None

    # Intervention Settings
    interventioner: str = "Deterministic" # "Deterministic", "Probabilistic", "None
    omega: float = 0.1 # intervention probability coefficient for Probabilistic Interventioner

    # Normalization Settings
    standardize_reward: bool = False
    normalize_obs: Union[bool, str] = "gym"  # "custom", "gym"
    normalize_values: bool = False
    target_update_rate: float = 0.2

    # AWAC Settings
    awac_lambda: float = 1.0  # lambda parameter for AWAC
    exp_adv_max: float = 100.0  # max value for the exponentiated advantage




@dataclass
class OnlineTrainConfig:
    rollout_length: int = 100
    horizon: int = 100000
    model_save_dir = "Saved_Agents/Online_Trained"  # where to save the model
    model_save_freq = 10 # how often to save the model in terms of rollouts
    batch_size = 100 # batch size for the online training
    def __post_init__(self):
        self.num_rollouts = self.horizon // self.rollout_length


@dataclass
class OfflineTrainConfig:
    offline_training_epochs: int = 100  # number of epochs to train on the offline data
    offline_batch_size: int = 100  # batch size for the offline training
    eval_freq = 10  # how often to evaluate the offline training
    eval_rollouts = 1  # how many rollouts to use for the evaluation
    eval_length = 1000  # how long each rollout should be for the evaluation
    model_save_dir = "Saved_Agents/Offline_Trained"  # where to save the model
    model_load_path = None #"Saved_Agents/Offline_Trained/Diamond2R05-AWAC_22_05_16_44/offline_agent_99.pt" #"Saved_Agents/Offline_Trained/AWAC_22_05_13_33_epoch900.pt"
@dataclass
class OfflineDataConfig:
    dataset = "Load" # "SafeActor", "Load", (Need to implement Random and BPM)
    save_path: str = "Offline_data/Diamond2R05" #None
    load_path: str = "Offline_data/Diamond2R05_22_05_16_24" #None
    num_rollouts: int = 1 # Number of rollouts to collect
    rollout_length: int = 10000

    def __post_init__(self):
        # get current datetime
        now = datetime.now()
        # dd/mm/YY_H:M:S
        dt_string = now.strftime("%d_%m_%H_%M")
        self.save_path = self.save_path + f"_{dt_string}"





@dataclass
class EnvConfig:
    env_json_path: str = "../JSON/Environment/Diamond2R05.json"
    flat_state_dim: int = None
    flat_action_dim: int = None
    self_normalize_obs: bool = False
    def __post_init__(self):
        self.name = os.path.splitext(os.path.basename(self.env_json_path))[0]

@dataclass
class RunSettings:
    save_freq: int = 1

@dataclass
class WandBConfig:
    project: str = "SafeAWAC"
    group: str = "SafeActor"
    name: str = "Diamond2R05-AWAC"
    checkpoints_path: Optional[str] = None

    def __post_init__(self):
        # Get datetime as string
        now = datetime.now()
        # dd/mm/YY_H:M:S
        dt_string = now.strftime("%d_%m_%H_%M")
        self.name = self.name + f"_{dt_string}"
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
    offline_data: OfflineDataConfig = field(default_factory=OfflineDataConfig)
    offline_training: OfflineTrainConfig = field(default_factory=OfflineTrainConfig)
    online_training: OnlineTrainConfig = field(default_factory=OnlineTrainConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    run_settings: RunSettings = field(default_factory=RunSettings)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    def print_all(self):
        print("-" * 80)
        print(f"Config: \n{pyrallis.dump(self)}")
        print("-" * 80)
        print("-" * 80)

    def update_trigger_state(self, trigger_state):
        self.agent.trigger_state = trigger_state
        if wandb.run is not None:
            wandb.run.config.update({"agent": self.agent}, allow_val_change=True)

    def __post_init__(self):
        # Make directories for saved agents
        offline_model_path = os.path.join(self.offline_training.model_save_dir, self.wandb.name)
        if not os.path.exists(offline_model_path):
            os.makedirs(offline_model_path)
            self.offline_training.model_save_path = offline_model_path
        online_model_path = os.path.join(self.online_training.model_save_dir, self.wandb.name)
        if not os.path.exists(online_model_path):
            os.makedirs(online_model_path)
            self.online_training.model_save_path = online_model_path

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

    # Initialize the environment
    parse_env_json(config.env.env_json_path, config)
    base_env = MultiClassMultiHop(config=config)
    env = flatten_env(base_env)
    if config.agent.normalize_obs == "gym":
        env = gym.wrappers.NormalizeObservation(env)
    mid, action_ranges = get_action_ranges(env)

    # set seed
    set_seed(config.seed, env, config.deterministic_torch)

    # init buffer
    buffer = Buffer(config.env.flat_state_dim, config.env.flat_action_dim, config.buffer_size, config.device)

    # Initialize Actor and Critics
    actor = MultiDiscreteActor(config.env.flat_state_dim, config.agent.actor_hidden_dim, action_ranges)
    actor.to(config.device)
    critic1 = Critic(config.env.flat_state_dim, config.env.flat_action_dim, config.agent.critic_hidden_dim)
    critic1.to(config.device)
    critic2 = Critic(config.env.flat_state_dim, config.env.flat_action_dim, config.agent.critic_hidden_dim)
    critic2.to(config.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.agent.learning_rate)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=config.agent.learning_rate)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=config.agent.learning_rate)

    # Initialize Intervention Actor
    safe_actor = MCMHBackPressurePolicy(env, M=True)
    if config.agent.interventioner == "Probabilistic":
        interventioner = ProbabilisticInterventioner(safe_actor, trigger_state = config.agent.trigger_state, lambda_ = config.agent.omega)
    elif config.agent.interventioner == "Deterministic":
        interventioner = Interventioner(safe_actor, trigger_state = config.agent.trigger_state)

    # Initialize Intervention Agent
    agent = SafeAWACAgent(actor, critic1, critic2, actor_optim, critic1_optim, critic2_optim, interventioner,
                      gamma = config.agent.gamma,
                      normalize_values = False,
                      target_update_rate= 0,
                      updates_per_rollout=config.agent.updates_per_rollout,
                      ppo_clip_coef=config.agent.ppo_clip_coef,
                      ppo = config.agent.ppo,
                      entropy_coef=config.agent.entropy_coef,
                      kl_coef=config.agent.kl_coef,
                      kl_target=config.agent.kl_target,
                      normalized_states = config.agent.normalize_obs,
                      grad_clip=config.agent.grad_clip,
                      awac_lambda=config.agent.awac_lambda,
                      exp_adv_max=config.agent.exp_adv_max,)

    # init wandb
    if config.offline_training.model_load_path is not None:
        print(f"LOADING MODEL FROM {config.offline_training.model_load_path}")
        agent.load_state_dict(torch.load(config.offline_training.model_load_path))

    wandb_init(config)


    # Pre-train Dataset
    if config.offline_data.dataset == "SafeActor":
        # Generate Backpressure Dataset according to config.pretrain settings
        agent.pretrain = True
        rollout = gen_rollout(env, agent, length=config.offline_data.rollout_length, pbar_desc="Generating Offline Dataset" ,show_progress=True)
        pickle.dump(rollout, open(config.offline_data.save_path, "wb"))


    else:
        # Load the dataset from the path specified in config.pretrain.dataset
        rollout = pickle.load(open(config.offline_data.load_path, "rb"))
    buffer.add_transitions(rollout)
    # Log rollouts
    offline_history, offline_lta_reward = log_rollouts(rollout, history=None, policy_name="Offline", glob="Offline",
                                          include=config.logger.include)
    if OFFLINE_TRAIN:
        agent.pretrain = True
    # Perform Offline Training
        for epoch in tqdm(range(config.offline_training.offline_training_epochs), desc="Offline Training"):
            batch = buffer.sample(config.offline_training.offline_batch_size)
            update_metrics = agent.update(batch)
            log_update_metrics(update_metrics, epoch, type = config.logger.type, glob = "OfflineUpdate")
            # Offline Evaluation
            if epoch % config.offline_training.eval_freq == 0 or epoch == config.offline_training.offline_training_epochs - 1:
                eval_env = deepcopy(env)
                eval_env.reset()
                eval_rollout = gen_rollout(eval_env, agent, length = config.offline_training.eval_length, show_progress=False)
                eval_history, eval_lta_reward = log_rollouts(eval_rollout, history = None, policy_name=f"Offline Agent {epoch}", glob = "OfflineAgentEval")
                log_rollout_summary(eval_rollout, epoch, glob = "OfflineAgentEval")
                # Save Agent
                torch.save(agent.state_dict(), os.path.join(config.offline_training.model_save_path, f"offline_agent_{epoch}.pt"))

        # Test offline trained agent
        env.reset()
        agent.pretrain = False
        rollout = gen_rollout(env, agent, length = 1000)
        test_history, test_lta_reward = log_rollouts(rollout, history = None, policy_name="Offline Agent", glob = "OfflineAgentTest")
        log_rollout_summary(rollout, 0, glob = "OfflineAgentTest")
    # Online Training
    env.reset()
    agent.pretrain = False
    online_history = None
    for eps in tqdm(range(config.online_training.num_rollouts), desc="Online Training"):
        rollout = gen_rollout(env, agent, length = config.online_training.rollout_length, show_progress=False)
        buffer.add_transitions(rollout)
        rollout = process_rollout(rollout, agent)
        history, eps_lta_reward = log_rollouts(rollout, history = online_history,  policy_name= "OnlineAgent", glob = "Online(live)", include =  config.logger.include)
        log_rollout_summary(rollout,eps, glob = "Online Rollout Summary")
        batch = buffer.sample(config.online_training.batch_size)
        update_metrics = agent.update(batch)
        log_update_metrics(update_metrics, eps, type = config.logger.type, glob = "OnlineUpdate")
        if config.online_training.model_save_freq is not None and eps % config.online_training.model_save_freq == 0\
            or eps == config.online_training.num_rollouts - 1:
            torch.save(agent.state_dict(), os.path.join(config.online_training.model_save_path,  f"online_agent_{eps}"))
    # Test final agent
    env.reset()
    rollout = gen_rollout(env, agent, length = 1000)
    test_history, test_lta_reward = log_rollouts(rollout, history = None, policy_name="Final Agent", glob = "Final Agent Test")
    log_rollout_summary(rollout, 0, glob = "Final Agent Test")
    #log_history(online_history)


    # num_rollouts = config.agent.num_rollouts
    # rollout_length = config.agent.rollout_length
    # history = None
    # # Pre-train
    # # Start real training
    # pbar = tqdm(range(num_rollouts), ncols=80, desc="Environment Rollouts with BP")
    # agent.pretrain = False
    # for eps in pbar:
    #     if eps < config.pretrain.num_rollouts:
    #         agent.pretrain = True # so we use the safe actor in pretraining
    #         pretrain_rollout = gen_rollout(env, agent, length=rollout_length, show_progress=False)
    #         rollout, pretrain_results = process_prerollout(pretrain_rollout,
    #                                                 threshold_ratio = config.agent.threshold_ratio,
    #                                                 rollout_length=rollout_length,
    #                                                 standardize_reward = config.agent.standardize_reward)
    #         if True:
    #             config.update_trigger_state(pretrain_results["q_threshold"])
    #             agent.set_safe_threshold(pretrain_results["q_threshold"])
    #         history, eps_lta_reward = log_rollouts(rollout, history=history, policy_name="Pretrain", glob="pretrain",
    #                                                include=config.logger.include)
    #         buffer.add_transitions(rollout)
    #         batch = buffer.get_last_rollout()
    #         if config.pretrain.num_updates > 0:
    #             pretrain_metrics = agent.fit_critic(batch, fit_epochs=config.pretrain.num_updates)
    #             log_pretrain_metrics(pretrain_metrics)
    #         agent.pretrain = False
    #     else:
    #         rollout = gen_rollout(env, agent, length=rollout_length, show_progress=False)
    #         buffer.add_transitions(rollout)
    #         rollout = process_rollout(rollout, agent)
    #         history, eps_lta_reward = log_rollouts(rollout, history = history,  policy_name= "agent", glob = "live", include =  config.logger.include)
    #         log_rollout_summary(rollout,eps, glob = "rollout_summary")
    #         batch = buffer.get_last_rollout()
    #         update_metrics = agent.update(batch)
    #         log_update_metrics(update_metrics, eps, type = config.logger.type)
    #
    # # Test final agent
    # env.reset()
    # rollout = gen_rollout(env, agent, length = 1000)
    # test_history, test_lta_reward = log_rollouts(rollout, history = None, policy_name="Final Agent", glob = "test")
    # log_rollout_summary(rollout, 0, glob = "test")
    # log_history(history)

    wandb.finish()

