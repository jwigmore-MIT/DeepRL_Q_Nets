import os
import yaml
from datetime import datetime
from munch import Munch


def parse_config(config_file_name: str):
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
    config.run_name = config.agent.policy_name + "_" + config.env.env_name + "_"  + "TRAIN" + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")

    # Set root directory
    config.root_dir = os.path.dirname(working_dir)

    # check if the policy is backpressure
    config.BP = True if config.agent.policy_name.__contains__("Backpressure") else False

    # check to save models
    if config.save_models:
        config.save_dir = os.path.join(working_dir, "saved_models", config.run_name)
        os.makedirs(config.save_dir, exist_ok=True)


    return config


