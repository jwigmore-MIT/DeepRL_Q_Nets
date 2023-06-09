from SB3.config.base import Config, AgentConfig
from Environments.MCMH_tools import generate_env
from SB3.utils import generate_agent, load_agent, EvalWandbLogger
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO

import pyrallis
import wandb
import yaml
import os
from collections import namedtuple
from munch import Munch
from stable_baselines3.common.evaluation import evaluate_policy
from datetime import datetime

def parse_config(config_file_name: str) -> Config:
    """
    Parses the config file
    """
    working_dir = os.path.dirname(os.path.abspath(__file__)) # directory root i.e. DeepRL_Q_Nets/SB3
    config_file = os.path.join(working_dir, "config", config_file_name) # full path to config file
    # parse the config file
    with open(config_file, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = Munch.fromDict(config_dict) # convert to Munch object
    # Extract env name
    config.env.env_name = config.env.env_json_path.split("/")[-1].split(".")[0]
    # Create run name
    config.run_name = config.agent.policy_name + "_" + config.env.env_name + "_" + "TEST" + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")

    # Set root directory
    config.root_dir = os.path.dirname(working_dir)
    # check if the policy is backpressure
    config.BP = True if config.agent.policy_name.__contains__("Backpressure") else False
    # check to save models
    if config.save_models:
        config.save_dir = os.path.join(working_dir, "saved_models", config.run_name)
        os.makedirs(config.save_dir, exist_ok=True)
    return config

# === Config === #
config_file_name = "PPO1.yaml"
config = parse_config(config_file_name)



# === Environment Generation === #
env = generate_env(config, monitor_settings = config.monitor.toDict(), backpressure = config.BP)



# === Loading Agent Generation === #
agent = load_agent(config, env)


# === Logger Initialization === #
run = wandb.init(
        project= config.wandb.project,
        config= vars(config),
        sync_tensorboard=False,  # auto-upload sb3's tensorboard metrics
        name= config.run_name,
    )




# === Testing === #
eval_env = generate_env(config, max_steps = config.env.time_limit, monitor_settings = config.monitor.toDict(), backpressure = config.BP)
EvalWandbLogger = EvalWandbLogger()
mean_reward, std_reward = evaluate_policy(agent, eval_env, callback = EvalWandbLogger._on_step, **config.eval.toDict())
EvalWandbLogger.write_log()


wandb.finish()


