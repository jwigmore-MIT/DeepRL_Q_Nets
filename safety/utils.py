import os
import yaml
from datetime import datetime
from munch import Munch
import numpy as np
import torch
import random
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
from plotly.subplots import make_subplots
import sys

# def parse_cleanrl_config(config_file_name: str, run_type = "TRAIN"):
#     working_dir = os.path.dirname(os.path.abspath(__file__))  # directory root i.e. DeepRL_Q_Nets/SB3
#     config_file = os.path.join(working_dir, "config_files", config_file_name)  # full path to config file
#     # parse the config file
#     with open(config_file, 'r') as f:
#         config_dict = yaml.load(f, Loader=yaml.FullLoader)
#
#     # convert to Munch object
#     config = Munch.fromDict(config_dict)
#     config.env_id = config.env.env_json_path.split("/")[-1].split(".")[0]
#     config.root_dir = os.path.dirname(working_dir)
#
#     return config

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
    config.BP = True if config.agent.policy_name.__contains__("BP") else False

    # check to save models
    if config.save_models and run_type == "TRAIN":
        config.save_dir = os.path.join(working_dir, "saved_models", config.run_name)
        os.makedirs(config.save_dir, exist_ok=True)

    config.debug = debugger_is_active()


    return config

def clean_rl_ppo_parse_config(config_file_name: str, run_type = "TRAIN"):
    """
    Parses the config file
    """
    working_dir = os.path.dirname(os.path.abspath(__file__)) # directory root i.e. DeepRL_Q_Nets/SB3
    config_file = os.path.join(working_dir, "config_files", config_file_name) # full path to config file
    # parse the config file
    with open(config_file, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    # convert to Munch object
    args = Munch.fromDict(config_dict)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # Extract env name
    args.env_name = args.env_json_path.split("/")[-1].split(".")[0]

    # Create run name
    args.run_name = args.policy_name + "_" + args.env_name + "_"  + run_type + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")

    # Artifact Name
    args.artifact_name = args.policy_name + "_" + args.env_name + "_" + "agent"

    # Agent Save directory

    # Set root directory
    args.root_dir = os.path.dirname(working_dir)

    # check if the policy is backpressure
    args.BP = True if args.policy_name.__contains__("BP") else False

    # check to save models
    if args.save_models and run_type == "TRAIN":
        args.save_dir = os.path.join(working_dir, "saved_models", args.run_name)
        os.makedirs(args.save_dir, exist_ok=True)

    args.debug = debugger_is_active()
    return args

def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


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


def visualize_buffer_pyplot(buffer, item_str, dim, range_ = None):

    if range_ is None:
        range_ = np.arange(buffer._pointer)

    item = getattr(buffer, item_str)
    item = item[range_]
    fig = plt.hist2d(x = range_, y = item[:,dim])
    # Add title to figure
    plt.title(f"{item_str} {dim} t = [{range_[0]}: {range_[-1]}]")
    plt.show()
    return fig

def visualize_buffer(buffer, item_str, dims, range_ = None):
    if isinstance(dims, int):
        dims = [dims]

    fig = make_subplots(rows=len(dims), cols=1, subplot_titles=[f"{item_str} {dim}" for dim in dims])

    if isinstance(buffer, dict):
        item = buffer[item_str]
    else:
        item = getattr(buffer, item_str)


    if range_ is None:
        if hasattr(buffer, "_pointer"):
            range_ = np.arange(buffer._pointer)
        else:
            range_ = np.arange(item.shape[0])

    item = item[range_]

    row = 1
    for dim in dims:
        fig.add_trace(go.Histogram2d(x = range_, y = item[:,dim], name = f"{item_str} {dim}", coloraxis="coloraxis"), row=row, col = 1)
        row+=1
    fig.update_layout(title_text=f"{item_str} {dims} t = [{range_[0]}: {range_[-1]}]")
    fig.update_layout(height=400*row,  title_x=0.5, title_font_size=30)
    fig.show()
    return fig

def wandb_sweep_helper(wandb_config):
    """
    Helper function to convert wandb sweep config to a Munch object
    """
    config = wandb_config.as_dict()
    config.env.env_name = config.env.env_json_path.split("/")[-1].split(".")[0]
    config.run_name = config.agent.policy_name + "_" + config.env.env_name + "_"  + "TRAIN" + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    config.artifact_name = config.agent.policy_name + "_" + config.env.env_name + "_" + "agent"
    config.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config.BP = True if config.agent.policy_name.__contains__("BP") else False
    config.debug = debugger_is_active()
    return config