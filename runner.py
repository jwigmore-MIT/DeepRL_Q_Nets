# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy

import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
from datetime import datetime
import wandb
import torch
import logging
logging.getLogger().setLevel(logging.INFO)
import warnings
warnings.filterwarnings('ignore')


from param_extractors import parse_training_json, parse_test_json, parse_env_json
from testers import test_from_artifact, test_BP
from wandb_utils import CheckpointSaver
from trainers import train_agent


def load_local_agent(agent_path):

    agent.load_state_dict(torch.load(agent_path))


def get_tags():
    input_str = input("Input any tags")
    tags = input_str.replace(" ", "").split(",")
    if "" in tags:
        tags.remove("")
    return tags

TRAIN = True
TEST = False
BP_TEST = False
WANDB_TEST = False

#model_load_path = "/home/jwigmore/PycharmProjects/DRL_Stoch_Qs/clean_rl/Best_models/16.03_07_55_CrissCrossTwoClass__PPO_para_1__5031998"
# https://www.reddit.com/r/reinforcementlearning/comments/11txwjw/agent_not_learning_3_problems_and_solutions/

if __name__ == "__main__":
    # Retrieve training, environment, and test parameters from json files
    train_param_path = "JSON/Training/PPO1.json"
    train_args = parse_training_json(train_param_path)
    env_param_path = "JSON/Environment/CrissCross1.json"
    env_para = parse_env_json(env_param_path)
    test_param_path = "JSON/Testing/test1.json"
    test_args = parse_test_json(test_param_path)
    wandb_project = "DRL_For_SQN"
    wandb_entity = "jwigmore-research"

    if wandb.run:
        print("Previous wandb process is running... Killing it!")
        wandb.finish()

    if TRAIN:
        dt_string = datetime.now().strftime("%m-%d_%H%M")
        run_name = f"TRAIN_{env_para['name']}_{train_args.name}_{dt_string}"
        tags = get_tags()

        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            job_type='Train',
            name=run_name,
            sync_tensorboard=True,
            config=vars(train_args),
            tags = tags,
            save_code=True,
        )

        save_model_path = f"Saved_Models/{env_para['name']}/{train_args.name}/"
        checkpoint_saver = CheckpointSaver(save_model_path, env_string= env_para["name"], algo_string=train_args.name, decreasing=False, top_n=5)

        outputs = train_agent(env_para, train_args, test_args, run, checkpoint_saver)
        run.finish()

    if TEST:
        dt_string = datetime.now().strftime("%m-%d_%H%M")
        run_name = f"TEST_{env_para['name']}_{test_args.name}_{dt_string}"
        artifact_name  = "jwigmore-research/DRL_For_SQN/Triangle1_PPO-p1.pt:v136"
        setattr(test_args,"artifact_name", artifact_name)
        tags = get_tags()
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            job_type='Test',
            name=run_name,
            sync_tensorboard=True,
            config=vars(test_args),
            tags = tags,
            save_code=True,
        )
        artifact = run.use_artifact('jwigmore-research/DRL_For_SQN/Triangle1_PPO-p1.pt:v136', type='model')

        agent, test_rewards, test_history = test_from_artifact(run, test_args, env_para, artifact, store_history = True)
        run.finish()

    if BP_TEST:
        dt_string = datetime.now().strftime("%m-%d_%H%M")
        run_name = f"TEST_BP_{env_para['name']}_{test_args.name}_{dt_string}"
        tags = get_tags()
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            job_type='Test',
            name=run_name,
            sync_tensorboard=True,
            config=vars(test_args),
            tags = tags,
            save_code=True,
        )
        all_rewards, test_history = test_BP(run, env_para, test_args, device= 'cpu')
        run.finish()








    if WANDB_TEST:
        best_model_dir = "Best_models/"
        dt_string = datetime.now().strftime("%m-%d_%H%M")
        run_name = f"{dt_string}_{env_para['name']}_{train_args.name}"
        run_name = f"{dt_string}_model-retr-test"
        run = wandb.init(project= 'clean_rl',
                         entity = 'jwigmore-research',
                         name = run_name,
                         #config = vars(args),
                         job_type = "retr"
                         )
        if False:
            json_artifact = wandb.Artifact("config-json", type = "json")
            json_artifact.add_file(json_file)
            run.log_artifact(json_artifact)
        if False:
            retrieved_artifact = run.use_artifact('config-json:latest')
            retrieved_artifact_path = retrieved_artifact.download()
            json_paths = []
            for x in os.listdir(retrieved_artifact_path):
                if x.endswith(".json"):
                    json_paths.append(x)
            param_tup = parse_json(os.path.join(retrieved_artifact_path,json_paths[0]))

        if True:
            model_artifact = run.use_artifact('model-ckpt-epoch-154.pt:latest')
            model_weight_dir = model_artifact.download()
            model_dict = {}
            for x in os.listdir(model_weight_dir):
                if x.endswith('.pt'):
                    model_dict = torch.load(os.path.join(model_weight_dir,x))



        run.finish()












