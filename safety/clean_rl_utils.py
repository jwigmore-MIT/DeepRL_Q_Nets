import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
from distutils.util import strtobool
import numpy as np
from munch import Munch
import yaml
from datetime import datetime
import gymnasium as gym
from Environments.ServerAssignment import ServerAssignment
from Environments.ServerAllocation import ServerAllocation

def observation_checker(obs: torch.Tensor):
    """
    Makes sure each element in the observation vector coming from the environment is within [-1,1]
    """
    if torch.any(obs > 1) or torch.any(obs < -1):
        raise ValueError("Observation value outside of [-1,1]")



def parse_args_or_config(config_path = None):

    if config_path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
            help="the name of this experiment")
        parser.add_argument("--env-id", type=str, default="LunarLanderContinuous-v2",
            help="the id of the gym environment")
        parser.add_argument("--seed", type=int, default=1,
            help="seed of the experiment")
        parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
            help="if toggled, `torch.backends.cudnn.deterministic=False`")
        parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
            help="if toggled, cuda will be enabled by default")
        parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
            help="if toggled, this experiment will be tracked with Weights and Biases")
        parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
            help="the wandb's project name")
        parser.add_argument("--policy-name", type=str, default="IA-AR-PPO",
            help="this is the name of the policy being run")
        parser.add_argument("--save_models", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
            help="whether to save models (probably doesn't work")

        # Algorithm specific arguments
        parser.add_argument("--env-json-path", type=str, default="",
            help="the location of the environment json file")
        parser.add_argument("--total-timesteps", type=int, default=500000,
            help="total timesteps of the experiments")
        parser.add_argument("--learning-rate", type=float, default=1e-4,
            help="the learning rate of the optimizer")
        parser.add_argument("--num-envs", type=int, default=1,
            help="the number of parallel game environments")
        parser.add_argument("--num-steps", type=int, default=128,
            help="the number of steps to run in each environment per policy rollout")
        parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False,
            help="Toggle learning rate annealing for policy and value networks")
        parser.add_argument("--gamma", type=float, default=1.0,
            help="the discount factor gamma")
        parser.add_argument("--gae-lambda", type=float, default=0.95,
            help="the lambda for the general advantage estimation")
        parser.add_argument("--num-minibatches", type=int, default=4,
            help="the number of mini-batches")
        parser.add_argument("--update-epochs", type=int, default=4,
            help="the K epochs to update the policy")
        parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
            help="Toggles advantages normalization")
        parser.add_argument("--clip-coef", type=float, default=0.2,
            help="the surrogate clipping coefficient")
        parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=False,
            help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
        parser.add_argument("--ent-coef", type=float, default=0.01,
            help="coefficient of the entropy")
        parser.add_argument("--vf-coef", type=float, default=0.5,
            help="coefficient of the value function")
        parser.add_argument("--max-grad-norm", type=float, default=0.5,
            help="the maximum norm for the gradient clipping")
        parser.add_argument("--target-kl", type=float, default=None,
            help="the target KL divergence threshold")
        parser.add_argument("--alpha", type = float, default = 0.1,
            help = "the update rate for AR parameters (eta and beta)")
        parser.add_argument("--nu", type = float, default = 0.0,
            help = "the bias coefficient for AR parameter beta")
        parser.add_argument("--reward-scale", type = float, default = 0.001,
            help = "what to scale the observed rewards by")
        parser.add_argument("--obs-scale", type = int, default=50,
            help ="observation scaling factor")
        parser.add_argument("--int-thresh", type = int, default= 25,
            help = "intervention threshold")
        parser.add_argument("--window-size", type = int, default= 1000,
            help = "size of window for window_average_backlogs monitoring")
        args = parser.parse_args()
        args.env_name = args.env_json_path.split("/")[-1].split(".")[0]
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.run_name = args.policy_name + "_" + args.env_name + "_" + datetime.now().strftime(
            "%Y%m%d-%H%M%S")
        args.artifact_name = args.policy_name + "_" + args.env_name + "_" + "agent"
        working_dir = os.path.dirname(os.path.abspath(__file__))  # directory root i.e. DeepRL_Q_Nets/SB3

        args.root_dir = os.path.dirname(working_dir)
        if args.save_models:
            args.save_dir = os.path.join(working_dir, "saved_models", args.run_name)
            os.makedirs(args.save_dir, exist_ok=True)

    else:
        args = clean_rl_ppo_parse_config(config_path)

    # Do checks
    if args.nu > 0.0:
       raise Exception("nu must be non-positive")
    if not hasattr(args, "cuda"):
        args.cuda = torch.cuda.is_available()
    if not hasattr(args, "apply_mask"):
        raise Exception("Must specify whether or not to apply mask")
    if not hasattr(args, "obs_links"):
        raise Exception("Must specify obs_links <bool> (ie whether or not link states are seen by agent")
    # fmt: on
    return args

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

    return args


def generate_clean_rl_env(config, env_type = "ServerAssigment",normalize = True):
    config = config
    normalize = normalize
    env_type = config.env_type if hasattr(config, "env_type") else env_type
    def thunk():
        env_para = parse_env_json(config.root_dir + config.env_json_path, config)
        env_para["seed"] = config.seed
        env_para["obs_links"] = config.obs_links if hasattr(config, "obs_links") else None
        if env_type == "ServerAssignment":
            env = ServerAssignment(env_para)
        elif env_type == "ServerAllocation":
            env = ServerAllocation(env_para)
        if normalize:
            # check to make sure obs_scale is greater than 1
            if config.obs_scale < 1:
                raise ValueError("config.obs_scale must be greater than 1")
            env = gym.wrappers.TransformReward(env, lambda x: x*config.reward_scale)
            env = gym.wrappers.TransformObservation(env, lambda x: 2*x/config.obs_scale-1)
        return env
    return thunk

def parse_env_json(json_path, config_args = None):
    import json
    para = json.load(open(json_path))
    env_para = para["problem_instance"]
    if config_args is not None:
        if hasattr(config_args,'env'):
            for key, value in env_para.items():
                setattr(config_args.env, f"{key}", value)
        else:
            for key, value in env_para.items():
                setattr(config_args, f"env.{key}", value)
    return env_para
