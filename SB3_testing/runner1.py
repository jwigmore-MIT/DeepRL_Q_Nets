import os
from param_extractors import parse_env_json
from Environments.MultiClassMultiHop import MultiClassMultiHop
from NonDRLPolicies.Backpressure import MCMHBackPressurePolicy
import gymnasium as gym
import numpy as np
from copy import deepcopy

# Pyrallis Imports
import pyrallis
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict, dataclass, field

@dataclass
class EnvConfig:
    env_json_path: str = "../JSON/Environment/Env1/Env1a.json"
    flat_state_dim: int = None
    flat_action_dim: int = None
    def __post_init__(self):
        self.name = os.path.splitext(os.path.basename(self.env_json_path))[0]



@dataclass
class Config:
    seed: int = 5031997

    env: EnvConfig = field(default_factory=EnvConfig)

    def print_all(self):
        print("-" * 80)
        print(f"Config: \n{pyrallis.dump(self)}")
        print("-" * 80)
        print("-" * 80)


class FlatActionWrapper(gym.ActionWrapper):
    """
    This action wrapper maps flattened actions <nd.array> back to dictionary
    actions of which the Base environment understands
    """

    def __init__(self, MCMH_env):
        super(FlatActionWrapper, self).__init__(MCMH_env)
        self.action_space = self.flatten_action_space()

    def action(self, action: np.ndarray):
        return self.unflatten_action(action)


class StepLoggingWrapper(gym.Wrapper):
    "Custom wrapper to log some of the outputs of the step function"

    def __init__(self, env, log_keys = ["backlog"], filename = "log.csv"):
        super(StepLoggingWrapper, self).__init__(env)
        self.log_keys = log_keys
        self.filename = filename
        self.eps = 0
        self.log ={self.eps: {}}
        for key in self.log_keys:
            self.log[self.eps][key] = []



    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        for key in self.log_keys:
            self.log[self.eps][key].extend(info[key])
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.eps += 1
        self.log[self.eps] = {}
        for key in self.log_keys:
            self.log[self.eps][key] = []
        return self.env.reset(**kwargs)

    def save_log(self):
        import csv
        with open(self.filename, 'w') as f:
            writer = csv.writer(f)
            for key, value in self.log.items():
                writer.writerow([key, value])



def generate_env(config: Config, monitor_settings = None, backpressure = False):
    env = MultiClassMultiHop(config=config)
    env = gym.wrappers.time_limit.TimeLimit(env, max_episode_steps=100)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if monitor_settings is not None:
        env = Monitor(env, **monitor_settings)
    env = FlatActionWrapper(env)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.ClipAction(env)
    if not backpressure:
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.NormalizeReward(env)
    env = StepLoggingWrapper(env)
    return env

if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.ppo import MlpPolicy
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.results_plotter import load_results, ts2xy
    import wandb
    from wandb.integration.sb3 import WandbCallback

    # initialize configuration
    config = Config()
    parse_env_json(config.env.env_json_path, config)

    # init wandb
    run = wandb.init(
        project="sb3",
        config= vars(config),
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )

    # Initialize environement
    env1 = generate_env(config, monitor_settings = {"filename": "training", "info_keywords": ("backlog",)})

    dumb_model = PPO(MlpPolicy, env1,
                     n_steps = 100,
                     verbose=0,
                     device = "cpu")

    # Backpressure Performance
    bp_env = generate_env(config, monitor_settings = {"filename": "backpressure", "info_keywords": ("backlog",)}, backpressure = True)
    backpressure_policy = MCMHBackPressurePolicy(bp_env, M = True)
    mean_reward, std_reward = evaluate_policy(backpressure_policy, bp_env, n_eval_episodes=10)

    # Untrained Performance
    # eval_env = deepcopy(env)
    # eval_env = Monitor(eval_env, filename = "untrained", info_keywords= ("backlog",))
    # mean_reward, std_reward = evaluate_policy(dumb_model, eval_env, n_eval_episodes=100)
    # print("Untrained Performance")
    # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


    # Train
    # train_env = generate_env(config, monitor_settings = {"filename": "training", "info_keywords": ("backlog",)})
    # model = PPO(MlpPolicy, train_env, n_steps = 100, verbose=0, device = "cpu", tensorboard_log= "log_dir")
    # # copy weights of dumb_model to model
    # model.set_parameters(dumb_model.get_parameters())
    #
    # model.learn(total_timesteps=100000, progress_bar= True,  tb_log_name="first_run", callback = WandbCallback())
    # #
    # # # # Trained Performance
    # eval_env = generate_env(config, monitor_settings = {"filename": "trained", "info_keywords": ("backlog",)})
    # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
    #
    # print("Trained Performance")
    # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")



