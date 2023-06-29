import os
import yaml
from datetime import datetime
from munch import Munch
import numpy as np
import torch
import random
import pickle


def parse_cleanrl_config(config_file_name: str, run_type = "TRAIN"):
    working_dir = os.path.dirname(os.path.abspath(__file__))  # directory root i.e. DeepRL_Q_Nets/SB3
    config_file = os.path.join(working_dir, "config_files", config_file_name)  # full path to config file
    # parse the config file
    with open(config_file, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    # convert to Munch object
    config = Munch.fromDict(config_dict)
    config.env_id = config.env.env_json_path.split("/")[-1].split(".")[0]
    config.root_dir = os.path.dirname(working_dir)

    return config

def parse_config(config_file_name: str, run_type = "TRAIN"):
    """
    Parses the config file
    """
    working_dir = os.path.dirname(os.path.abspath(__file__)) # directory root i.e. DeepRL_Q_Nets/SB3
    config_file = os.path.join(working_dir, "config_files", config_file_name) # full path to config file
    # parse the config file
    with open(config_file, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    # convert to Munch object
    config = Munch.fromDict(config_dict)

    # Extract env name
    config.env.env_name = config.env.env_json_path.split("/")[-1].split(".")[0]

    # Create run name
    config.run_name = config.agent.policy_name + "_" + config.env.env_name + "_"  + run_type + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")

    # Artifact Name
    config.artifact_name = config.agent.policy_name + "_" + config.env.env_name + "_" + "agent"

    # Agent Save directory

    # Set root directory
    config.root_dir = os.path.dirname(working_dir)

    # check if the policy is backpressure
    config.BP = True if config.agent.policy_name.__contains__("Backpressure") else False

    # check to save models
    if config.save_models and run_type == "TRAIN":
        config.save_dir = os.path.join(working_dir, "saved_models", config.run_name)
        os.makedirs(config.save_dir, exist_ok=True)


    return config


def process_rollout(rollout, agent):
    with torch.no_grad():
        rollout["action_prob"] = np.exp(agent.get_log_prob(torch.Tensor(rollout["nn_obs"]), torch.Tensor(rollout["actions"])).detach().cpu().numpy())
        #np.exp(agent.actor.log_prob(torch.Tensor(rollout["obs"]), torch.Tensor(rollout["actions"])).detach().cpu().numpy())
        #rollout["v_values"] = agent.critic.forward(torch.tensor(rollout["obs"])).detach().cpu().numpy()
    return rollout

def set_seed(
    seed: int, env= None, deterministic_torch: bool = False
):

    os.environ["PYTHONHASHSEED"] = str(seed)
    env.reset(seed = seed)
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



def save_agent(agent, save_dir, mod = ""):
    """
    Saves the agent
    """
    pickle.dump(agent, open(save_dir + f"/agent{mod}.pkl", "wb"))

def save_config(config, save_dir):
    """
    Saves the config
    """
    yaml.dump(config, open(save_dir + "/config.yaml", "w"))