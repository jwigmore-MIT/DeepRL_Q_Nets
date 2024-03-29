from Environments.ServerAssignment import generate_clean_rl_env
from safety.utils import clean_rl_ppo_parse_config
import wandb
import random
import numpy as np
import torch
from tqdm import tqdm

run_type = "JSQ"
config_file = "clean_rl/N12S1/N12S1_IA_AR_PPO.yaml"
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


test_length = int(1e6)
wandb_log_interval = 100
backlogs = np.zeros(test_length)
pbar = tqdm(range(int(test_length)))
for t in pbar:
    if run_type == "JRQ":
        action = env.action_space.sample() # JRQ
    elif run_type == "JSQ":
        action = np.argmin(list(env.buffers.values())[1:-1]) # JSQ
    elif run_type == "JWQ":
        action = np.argmin(np.array(list(env.buffers.values()))[1:-1]*(env.unrel)) # JWQ

    next_obs, reward, terminated, truncated, info = env.step(action, debug=False)
    backlogs[t] = info["backlog"]
    if t > 0 and t % wandb_log_interval == 0:
        if t >= args.window_size:
            window_averaged_backlog = np.mean(
                backlogs[t  - args.window_size:t])
        else:
            window_averaged_backlog = np.mean(backlogs[:t])
        lta_backlogs = np.cumsum(backlogs[:t]) / np.arange(1, t + 1)
        wandb.log({"rollout/lta_backlogs": lta_backlogs[-1],
                   "rollout/window_averaged_backlog": window_averaged_backlog,
                   "global_step": t })
# compute running sum over backlogs

