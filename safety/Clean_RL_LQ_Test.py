from safety.clean_rl_utils import  clean_rl_ppo_parse_config, generate_clean_rl_env
import wandb
import random
import numpy as np
import torch
from tqdm import tqdm
run_types = ["LQ", "SQ", "RQ", "LCQ", "MWQ"]
#run_types = ["LQ"]#,"RQ"]
for run_type in run_types:
    config_file = "clean_rl/ServerAllocation/M2/M2A1_IA_AR_PPO.yaml"
    args = clean_rl_ppo_parse_config(config_file)
    env = generate_clean_rl_env(args, env_type = "Allocation",normalize = False)()

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


    test_length = int(2e6)
    wandb_log_interval = 1000
    backlogs = np.zeros(test_length)
    pbar = tqdm(range(int(test_length)))
    for t in pbar:
        if env.get_obs().sum() == 0:
            action = 0
        else:
            action = env.get_stable_action(type = run_type)
            # if run_type == "LQ":
            #     action = np.argmax(env.get_obs()) + 1  # LQ
            # elif run_type == "SQ":
            #     obs = env.get_obs()
            #     obs[obs == 0] = 1000000
            #     action = np.argmin(obs) + 1
            # elif run_type == "RQ":
            #     action = np.random.choice(np.where(env.get_obs() > 0)[0]) + 1
            # elif run_type == "LCQ":
            #     cap = env.get_cap()
            #     obs = env.get_obs()
            #     connected_obs = cap * obs[:-1]
            #     action = np.argmax(connected_obs) + 1
            # elif run_type == "MWQ":
            #     p_cap = 1-env.unreliabilities
            #     obs = env.get_obs()
            #     weighted_obs = p_cap * obs[:-1]
            #     action = np.argmax(weighted_obs) + 1

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
    wandb.finish()
# compute running sum over backlogs
