# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy

import os

import pandas as pd

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
from datetime import datetime
import wandb
import torch
import random
import numpy as np
import logging
logging.getLogger().setLevel(logging.INFO)
import warnings
warnings.filterwarnings('ignore')


from param_extractors import parse_training_json, parse_test_json, parse_env_json
from testers import test_from_artifact, test_BP
from wandb_utils import CheckpointSaver
from trainers import train_ppo_agent





def get_user_input():
    input_str = input("Input any tags for run")
    tags = input_str.replace(" ", "").split(",")
    if "" in tags:
        tags.remove("")

    notes = input("Input any notes for run")

    return tags, notes

def add_final_notes(run):
    input_str = input("Add any final notes:")
    curr_notes = run.notes
    new_notes = run.notes + f"\n" + "POST RUN NOTES:" + input_str
    run.notes = new_notes


ENV_TEST = True

TRAIN = True
TEST = False
BP_TEST = False
STATIC_TEST = False


#model_load_path = "/home/jwigmore/PycharmProjects/DRL_Stoch_Qs/clean_rl/Best_models/16.03_07_55_CrissCrossTwoClass__PPO_para_1__5031998"
# https://www.reddit.com/r/reinforcementlearning/comments/11txwjw/agent_not_learning_3_problems_and_solutions/

''' TO RUN
1. BP Stability on CrissCross2
2. TRAINING PPO Agent on CrissCross2
3. TEST Learned PPO Agent on CrissCross2
4. Static Stabilizing Policy
'''
if __name__ == "__main__":
    # Retrieve training, environment, and test parameters from json files
    train_param_path = "JSON/Training/PPO-CC2.json"
    train_name = train_param_path.split("/")[-1].replace(".json","")
    train_args = parse_training_json(train_param_path)

    env_param_path = "JSON/Environment/Old/CrissCross2.json"
    env_name = env_param_path.split("/")[-1].replace(".json","")
    env_para = parse_env_json(env_param_path)

    test_param_path = "JSON/Testing/test-CC2.json"
    test_name = test_param_path.split("/")[-1].replace(".json","")
    test_args = parse_test_json(test_param_path)

    static_pol = env_name

    wandb_project = "Scheduling_Routing_DRL"
    wandb_entity = "jwigmore-research"
    if TEST:
        artifact_name  = "jwigmore-research/DRL_For_SQN/BaiNet1_PPO-BaiNet.pt:v57"


    if ENV_TEST:
        from environment_init import make_MCMH_env
        env = make_MCMH_env(env_para)()


    if wandb.run:
        print("Previous wandb process is running... Killing it!")
        wandb.finish()

    if TRAIN:
        dt_string = datetime.now().strftime("%m-%d_%H%M")
        run_name = f"TRAIN_{env_name}_{train_name}_{dt_string}"
        tags, notes = get_user_input()

        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            job_type='Train',
            name=run_name,
            sync_tensorboard=True,
            config=vars(train_args),
            tags = tags,
            notes = notes,
            save_code=True,
        )

        save_model_path = f"Saved_Models/{env_name}/{train_name}/"
        checkpoint_saver = CheckpointSaver(save_model_path, env_string= env_name, algo_string=train_name, decreasing=False, top_n=5)

        outputs = train_ppo_agent(env_para, train_args, test_args, run, checkpoint_saver)
        try:
            add_final_notes(run)
        except Exception:
            pass
        run.finish()

    if TEST:
        dt_string = datetime.now().strftime("%m-%d_%H%M")
        run_name = f"TEST_{env_name}_{test_name}_{dt_string}"
        setattr(test_args,"artifact_name", artifact_name)
        tags, notes = get_user_input()

        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            job_type='Test',
            name=run_name,
            sync_tensorboard=True,
            config=vars(test_args),
            tags = tags,
            notes = notes,
            save_code=True,
        )
        artifact = run.use_artifact(artifact_name, type='model')

        test_outputs = test_from_artifact(run, test_args, env_para, artifact, store_history = True)
        try:
            add_final_notes(run)
        except Exception:
            pass
        run.finish()

    if BP_TEST:
        dt_string = datetime.now().strftime("%m-%d_%H%M")
        run_name = f"TEST_BP_{env_name}_{test_name}_{dt_string}"
        tags, notes = get_user_input()

        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            job_type='Test',
            name=run_name,
            sync_tensorboard=True,
            config=vars(test_args),
            tags = tags,
            notes = notes,
            save_code=True,
        )
        test_outputs = test_BP(run, env_para, test_args, device= 'cpu')
        try:
            add_final_notes(run)
        except Exception:
            pass
        run.finish()

    if STATIC_TEST:
        from testers import test_StaticPolicy

        dt_string = datetime.now().strftime("%m-%d_%H%M")
        run_name = f"TEST_SP_{env_name}_{test_name}_{dt_string}"
        tags, notes = get_user_input()

        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            job_type='Test',
            name=run_name,
            sync_tensorboard=True,
            config=vars(test_args),
            tags=tags,
            notes=notes,
            save_code=True,
        )
        test_outputs = test_StaticPolicy(run, static_pol, env_para, test_args, device= 'cpu')
        try:
            add_final_notes(run)
        except Exception:
            pass
        run.finish()




def plot_qs_vs_time(test_history, merge = True):
    from copy import deepcopy
    import pandas as pd
    q_dfs = []
    for key, value in test_history.items():
        if key is 'Env_seeds':
            continue
        else:
            df = value
            q_cols = [x for x in df.columns if "Q" in x]
            qi_df = df.loc[:, q_cols]
            q_dfs.append(qi_df)
    if merge:
        q_df = pd.concat(q_dfs).groupby(level = 0, axis = 'columns').mean()
        fig = q_df.plot()
        fig.show()
        return q_dfs, q_df
    else:
        return q_dfs, None
















