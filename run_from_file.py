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
from collections import defaultdict
logging.getLogger().setLevel(logging.INFO)
import warnings
from copy import deepcopy

warnings.filterwarnings('ignore')


from param_extractors import parse_jsons
from testers import test_from_artifact, test_BP
from wandb_utils import CheckpointSaver
from trainers import train_agent



file = "run_settings/run_settings.txt"

def get_user_input():
    input_str = input("Input any tags for run: ")
    tags = input_str.replace(" ", "").split(",")
    if "" in tags:
        tags.remove("")

    notes = input("Input any notes for run: ")

    return tags, notes

def add_final_notes(run):
    input_str = input("Add any final notes:")
    curr_notes = run.notes
    new_notes = run.notes + f"\n" + "POST RUN NOTES:" + input_str
    run.notes = new_notes



def read_args_file(args_file):
    args1 = defaultdict(None)

    # open the text file for reading
    with open(args_file, 'r') as f:
        # read the contents of the file
        contents = f.read()

        # split the contents by newline to get individual lines
        lines = contents.split('\n')
        # lines = re.split(r':(?=")', contents)
        # iterate over each line
        for line in lines:
            # split the line by colon to get the argument and value
            parts = line.split(':')
            arg = parts[0].strip()
            if arg == '':
                continue
            value = parts[1].strip()

            # add the argument and value to the dictionary
            args1[arg] = eval(value)

        # set the command line arguments to the values from the dictionary
        # sys.argv = [sys.argv[0]] + list(args1.values())
    return args1

def run_train():
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
            config=vars(config_args),
            tags = tags,
            notes = notes,
            save_code=True,
        )

        save_model_path = f"Saved_Models/{env_name}/{train_name}/"
        checkpoint_saver = CheckpointSaver(save_model_path, env_string= env_name, algo_string=train_name, decreasing=False, top_n=5)

        outputs = train_agent(env_para, train_args, test_args, run, checkpoint_saver)
        try:
            add_final_notes(run)
        except Exception:
            pass
        run.finish()
        return outputs

def run_train_w_agent():
    dt_string = datetime.now().strftime("%m-%d_%H%M")
    run_name = f"RETRAIN_{env_name}_{train_name}_{dt_string}"
    tags, notes = get_user_input()
    setattr(train_args, "artifact_name", artifact_name)
    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        job_type='Retrain',
        name=run_name,
        sync_tensorboard=True,
        config=vars(config_args),
        tags=tags,
        notes=notes,
        save_code=True,
    )
    save_model_path = f"Saved_Models/{env_name}/{train_name}/"
    checkpoint_saver = CheckpointSaver(save_model_path, env_string=env_name, algo_string=train_name, decreasing=False,
                                       top_n=5)

    artifact = run.use_artifact(artifact_name, type = "model")
    outputs = train_agent(env_para, train_args, test_args, run, checkpoint_saver,artifact)
    try:
        add_final_notes(run)
    except Exception:
        pass
    run.finish()
    return outputs
def run_test():
    dt_string = datetime.now().strftime("%m-%d_%H%M")
    run_name = f"TEST_{env_name}_{test_name}_{dt_string}"
    setattr(test_args, "artifact_name", artifact_name)
    tags, notes = get_user_input()

    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        job_type='Test',
        name=run_name,
        sync_tensorboard=True,
        config=vars(config_args),
        tags=tags,
        notes=notes,
        save_code=True,
    )
    artifact = run.use_artifact(artifact_name, type='model')

    test_outputs = test_from_artifact(run, test_args, env_para, artifact, store_history=True)
    try:
        add_final_notes(run)
    except Exception:
        pass
    run.finish()
    return test_outputs

def run_BP_test(M = False):
    dt_string = datetime.now().strftime("%m-%d_%H%M")
    if M:
        run_name = f"TEST_BPM_{env_name}_{test_name}_{dt_string}"
    else:
        run_name = f"TEST_BP_{env_name}_{test_name}_{dt_string}"
    tags, notes = get_user_input()

    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        job_type='BP_Test',
        name=run_name,
        sync_tensorboard=True,
        config=vars(config_args),
        tags=tags,
        notes=notes,
        save_code=True,
    )
    test_outputs = test_BP(run, env_para, test_args,M = M, device='cpu')
    try:
        add_final_notes(run)
    except Exception:
        pass
    run.finish()
    return test_outputs

def run_static_test():
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
        config=vars(config_args),
        tags=tags,
        notes=notes,
        save_code=True,
    )
    test_outputs = test_StaticPolicy(run, static_pol, env_para, test_args, device='cpu')
    try:
        add_final_notes(run)
    except Exception:
        pass
    run.finish()
    return test_outputs

def run_sweep():
    """ Notes
    1. Sweep configuration which is passed to wandb needs to be a dictionary
    2. The parameters being swept would then need to be modified in the training_args
    """
    def create_new_train_args():
        new_config = {}
        new_train_args = deepcopy(train_args)
        for key, value in run.config.items():
            new_config[key] = value
            train_param_key = key.split("/")[1]
            setattr(new_train_args, train_param_key, value)
        new_train_args.batch_size = int(new_train_args.num_envs * new_train_args.num_steps_per_rollout)
        new_train_args.minibatch_size = int(new_train_args.batch_size // new_train_args.num_minibatches_per_update)
        new_train_args.num_resets_per_env = new_train_args.total_timesteps // new_train_args.num_envs // new_train_args.num_steps_per_reset
        return new_train_args




    run = wandb.init()
    #print(run.config)
    sweep_train_args = create_new_train_args()
    run.config.setdefaults(config_args)
    save_model_path = f"Saved_Models/Sweeps{sweep_id}/"
    checkpoint_saver = CheckpointSaver(save_model_path, env_string=env_name, algo_string=train_name, decreasing=False,
                                       top_n=5)

    train_agent(env_para, sweep_train_args, test_args, run, checkpoint_saver)


#model_load_path = "/home/jwigmore/PycharmProjects/DRL_Stoch_Qs/clean_rl/Best_models/16.03_07_55_CrissCrossTwoClass__PPO_para_1__5031998"
# https://www.reddit.com/r/reinforcementlearning/comments/11txwjw/agent_not_learning_3_problems_and_solutions/

''' TO RUN
1. BP Stability on CrissCross2
2. TRAINING PPO Agent on CrissCross2
3. TEST Learned PPO Agent on CrissCross2
4. Static Stabilizing Policy
'''
if __name__ == "__main__":

    ENV_TEST = True


    # Retrieve training, environment, and test parameters from json files
    args1 = read_args_file('run_settings/sweep_testing.txt')


    TRAIN = args1["TRAIN"]
    RETRAIN = args1["RETRAIN"]
    TEST = args1["TEST"]
    BP_TEST = args1["BP_TEST"]
    BPM_TEST = args1["BPM_TEST"]
    STATIC_TEST = args1["STATIC_TEST"]
    SWEEP = args1["SWEEP"]

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

    static_pol = env_name

    wandb_project = args1["wandb_project"]
    wandb_entity = args1["wandb_entity"]
    if TEST or RETRAIN:
        artifact_name  = args1["artifact_name"] + ":" + args1["artifact_version"]


    if ENV_TEST:
        from environment_init import make_MCMH_env
        env = make_MCMH_env(env_para)()


    if wandb.run:
        print("Previous wandb process is running... Killing it!")
        wandb.finish()

    if TRAIN:
        train_outputs = run_train()
    if RETRAIN:
        train_outputs = run_train_w_agent()
    if TEST:
        test_outputs = run_test()
    if BP_TEST:
        BP_test_outputs = run_BP_test(M = False)
    if BPM_TEST:
        BPM_test_outputs = run_BP_test(M = True)
    if STATIC_TEST:
        static_test_outputs = run_static_test()
    if SWEEP:
        sweep_configuration = {
            "method": "bayes",
            "name": "sweep_test4",
            "metric": {"name": "train/avg_eps_backlog", "goal": "minimize"},
            "parameters": {
                "train/actor_learning_rate": {"max": 1e-1, "min": 1e-5},
                "train/critic_learning_rate": {"max": 1e-1, "min": 1e-5},
                "train/critic_lr_decay_rate": {"max": 1.0, "min": 1e-2},
                "train/actor_lr_decay_rate": {"max": 1.0, "min": 1e-2},
                "train/critic_lr_decay" : {"values": ["linear", "exponential"]},
                "train/actor_lr_decay" : {"values": ["linear", "exponential"]},
                "train/num_steps_per_rollout": {"values": [16, 32, 64, 128, 256]},
                "train/gamma" : {"max": 1.0, "min": 0.5},
                "train/gae_lambda":{"max": 0.9999, "min": 0.25},
                "train/time_scaled" : {"values" : [0, 1]},
                "train/ent_coef" :{"values": [0, 0.05, 0.1, 0.2]},
                "train/vf_coef" : {"values" : [0.1, 0.25, 0.5, 0.75, 1]},
                "train/num_envs" : {"values": [1, 2, 5, 10, 15]},
                "train/time_scaled": {"values": [True, False]},
                "train/minibatches_per_update": {"values": [1, 2, 4, 8]},
                "train/updates_per_rollout": {"values": [1, 2, 4, 8]},
                "train/max_grad_norm" : {"values": [0, 0.25, 0.5, 1]},
                "train/clip_coef": {"min": 0.1, "max": 5.0},
                "train/norm_adv": {"values" : [True, False]},
                "train/clip_vloss" : {"values": [True, False]}
            }
        }
        sweep_id = wandb.sweep(sweep = sweep_configuration, project = "my-fourth-sweep")
        wandb.agent(sweep_id, function = run_sweep, count = 500)

    # if False:
    #     dt_string = datetime.now().strftime("%m-%d_%H%M")
    #     run_name = f"TRAIN_{env_name}_{train_name}_{dt_string}"
    #     tags, notes = get_user_input()
    #
    #     run = wandb.init(
    #         project=wandb_project,
    #         entity=wandb_entity,
    #         job_type='Train',
    #         name=run_name,
    #         sync_tensorboard=True,
    #         config=vars(train_args),
    #         tags = tags,
    #         notes = notes,
    #         save_code=True,
    #     )
    #
    #     save_model_path = f"Saved_Models/{env_name}/{train_name}/"
    #     checkpoint_saver = CheckpointSaver(save_model_path, env_string= env_name, algo_string=train_name, decreasing=False, top_n=5)
    #
    #     outputs = train_agent(env_para, train_args, test_args, run, checkpoint_saver)
    #     try:
    #         add_final_notes(run)
    #     except Exception:
    #         pass
    #     run.finish()
    #
    # if False:
    #     dt_string = datetime.now().strftime("%m-%d_%H%M")
    #     run_name = f"TEST_{env_name}_{test_name}_{dt_string}"
    #     setattr(test_args,"artifact_name", artifact_name)
    #     tags, notes = get_user_input()
    #
    #     run = wandb.init(
    #         project=wandb_project,
    #         entity=wandb_entity,
    #         job_type='Test',
    #         name=run_name,
    #         sync_tensorboard=True,
    #         config=vars(test_args),
    #         tags = tags,
    #         notes = notes,
    #         save_code=True,
    #     )
    #     artifact = run.use_artifact(artifact_name, type='model')
    #
    #     test_outputs = test_from_artifact(run, test_args, env_para, artifact, store_history = True)
    #     try:
    #         add_final_notes(run)
    #     except Exception:
    #         pass
    #     run.finish()
    #
    # if False:
    #     dt_string = datetime.now().strftime("%m-%d_%H%M")
    #     run_name = f"TEST_BP_{env_name}_{test_name}_{dt_string}"
    #     tags, notes = get_user_input()
    #
    #     run = wandb.init(
    #         project=wandb_project,
    #         entity=wandb_entity,
    #         job_type='Test',
    #         name=run_name,
    #         sync_tensorboard=True,
    #         config=vars(test_args),
    #         tags = tags,
    #         notes = notes,
    #         save_code=True,
    #     )
    #     test_outputs = test_BP(run, env_para, test_args, device= 'cpu')
    #     try:
    #         add_final_notes(run)
    #     except Exception:
    #         pass
    #     run.finish()
    #
    # if False:
    #     from testers import test_StaticPolicy
    #
    #     dt_string = datetime.now().strftime("%m-%d_%H%M")
    #     run_name = f"TEST_SP_{env_name}_{test_name}_{dt_string}"
    #     tags, notes = get_user_input()
    #
    #     run = wandb.init(
    #         project=wandb_project,
    #         entity=wandb_entity,
    #         job_type='Test',
    #         name=run_name,
    #         sync_tensorboard=True,
    #         config=vars(test_args),
    #         tags=tags,
    #         notes=notes,
    #         save_code=True,
    #     )
    #     test_outputs = test_StaticPolicy(run, static_pol, env_para, test_args, device= 'cpu')
    #     try:
    #         add_final_notes(run)
    #     except Exception:
    #         pass
    #     run.finish()




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
















