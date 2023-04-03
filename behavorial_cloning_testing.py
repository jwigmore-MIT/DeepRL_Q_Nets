
import os

import pandas as pd

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
from datetime import datetime
import wandb
import torch
import random
import numpy as np
import logging
from collections import defaultdict
logging.getLogger().setLevel(logging.INFO)
import warnings

warnings.filterwarnings('ignore')


from param_extractors import parse_jsons
from testers import test_from_artifact, test_BP
from wandb_utils import CheckpointSaver
from trainers import train_agent
from run_from_file import read_args_file, run_BP_test


if __name__ == '__main__':

    ENV_TEST = True
    # Retrieve training, environment, and test parameters from json files
    args1 = read_args_file("run_settings/bc_run_settings.txt")


    TRAIN = args1["TRAIN"]
    RETRAIN = args1["RETRAIN"]
    TEST = args1["TEST"]
    BP_TEST = args1["BP_TEST"] or True
    STATIC_TEST = args1["STATIC_TEST"]

    train_param_path = args1["train_param_path"]
    train_name = train_param_path.split("/")[-1].replace(".json","")
    # train_args = parse_training_json(train_param_path)

    env_param_path = args1["env_param_path"]
    env_name = env_param_path.split("/")[-1].replace(".json","")
    # env_para = parse_env_json(env_param_path)

    test_param_path = args1["test_param_path"]
    test_name = test_param_path.split("/")[-1].replace(".json","")
    # test_args = parse_test_json(test_param_path)

    config_args, train_args, env_para, test_args = parse_jsons(train_param_path, env_param_path, test_param_path)

    if BP_TEST:
        BP_test_outputs = run_BP_test()

