from Environments.ServerAllocation import generate_clean_rl_env
from safety.utils import clean_rl_ppo_parse_config
import wandb
import random
import numpy as np
import torch
from tqdm import tqdm

run_type = "JSQ"
config_file = "clean_rl/N4S3_IA_ARPPO1.yaml"
args = clean_rl_ppo_parse_config(config_file)
env = generate_clean_rl_env(args, normalize = False)()

run_name = f"{args.env_name}__{run_type}"

wandb.init(
    project=args.wandb_project_name,
    sync_tensorboard=False,
    config=vars(args),
    name=run_name,
    monitor_gym=True,
    save_code=True,
)


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


test_length = int(1e5)
wandb_log_interval = 100
backlogs = np.zeros(test_length)
pbar = tqdm(range(int(test_length)))
for t in pbar:
    #action = env.action_space.sample() # JRQ

    action = np.argmin(list(env.buffers.values())[1:-1]) # JSQ

    next_obs, reward, terminated, truncated, info = env.step(action, debug=False)
    backlogs[t] = info["backlog"]
# compute running sum over backlogs
lta_backlogs = np.cumsum(backlogs)/np.arange(1,test_length+1)
for e in range(int(test_length/wandb_log_interval)):
    wandb.log({"rollout/lta_backlogs": lta_backlogs[e*wandb_log_interval]})