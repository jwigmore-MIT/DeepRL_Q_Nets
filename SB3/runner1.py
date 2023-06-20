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
                     n_steps = 128,
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



