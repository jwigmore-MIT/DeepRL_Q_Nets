import wandb
import numpy as np

def log_rollouts(rollout, history = None,  policy_name = "policy", glob = "test", include = ["all"], log_vectors = False):

    if history is None:
        start_time = 1
        stop_time = rollout["rewards"].shape[0] + 1
    else:
        start_time = history["rewards"].shape[0]
        stop_time = start_time + rollout["rewards"].shape[0]
    rollout["LTA_Rewards"], _ = get_reward_stats(np.array(rollout["rewards"]).reshape(-1,1), start_time, stop_time)
    rollout["LTA_Backlogs"], _ = get_reward_stats(np.array(rollout["backlogs"]).reshape(-1,1), start_time, stop_time)
    if isinstance(include, str):
        include = [include]
    if not isinstance(include,list):
        Exception("include must be a list")
    if "all" in include:
        include = rollout.keys()
    if history is None:
        history = {}
        for key in include:
            if key in rollout.keys():
                if len(rollout[key].shape) > 1 and  rollout[key].shape[1] > 1 and log_vectors:
                    for j in range(rollout[key].shape[1]):
                        wandb.define_metric(f"{glob}/{key}_{j}")
                else:
                    wandb.define_metric(f"{glob}/{key}")
                history[key]  = rollout[key]
        wandb.define_metric(f"{glob}/step")
        start_step = 0
    else:
        start_step = history["rewards"].shape[0]
        for key in include:
            if key in rollout.keys():
                # check if history contains key, if not add it
                if key not in history.keys():
                    history[key] = rollout[key]
                else:
                    history[key] = np.concatenate([history[key], rollout[key]])

    for i in range(len(rollout["rewards"])):
        log_dict = {}
        for key in include:
            if len(rollout[key].shape) > 1 and  rollout[key].shape[1] > 1 and log_vectors:
                for j in range(rollout[key].shape[1]):
                    log_dict[f"{glob}/{key}_{j}"] = rollout[key][i,j]
            else:
                log_dict[f"{glob}/{key}"] = rollout[key][i]
        log_dict.update({f"{glob}/step": start_step + i})
        wandb.log(log_dict)

    # Add all to history


    return history, rollout["LTA_Rewards"][-1]

def get_reward_stats(reward_vec, startime = None, stoptime = None):
    if startime is None:
        startime = 1
    if stoptime is None:
        reward_vec.shape[0] + 1
    normalizer = np.arange(startime, stoptime)
    sum_average_rewards = np.array(
        [np.cumsum(reward_vec[:,i], axis = 0)/normalizer
         for i in range(reward_vec.shape[1])]).T

    time_averaged_rewards = sum_average_rewards.mean(axis = 1)
    time_averaged_errors = sum_average_rewards.std(axis=1)
    return time_averaged_rewards, time_averaged_errors


def log_pretrain_metrics(metrics):
    for key in metrics.keys():
        wandb.define_metric(f"pretrain/{key}")
    wandb.define_metric(f"pretrain/step")
    for i in range(metrics["critic_loss"].shape[0]):
        log_dict = {}
        for key in metrics.keys():
            log_dict[f"pretrain/{key}"] = metrics[key][i]
        log_dict.update({"pretrain/step": i})
        wandb.log(log_dict)
