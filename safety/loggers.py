import wandb
import numpy as np

def log_rollouts(rollout, history = None,  policy_name = "policy", glob = "test", include = ["all"], log_vectors = False):

    # Per time step logging
    rollout_length = len(rollout["rewards"])
    # Make sure includes is set up correctly
    if isinstance(include, str):
        include = [include]
    if not isinstance(include,list):
        Exception("include must be a list")
    if "all" in include:
        include = rollout.keys()
    # initialize history if first time calling log_rollouts
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
    else: # if history already exists, append rollout to history
        start_step = history["rewards"].shape[0]
        for key in include:
            if key in rollout.keys():
                # check if history contains key, if not add it
                if key not in history.keys():
                    history[key] = rollout[key]
                else:
                    history[key] = np.concatenate([history[key], rollout[key]])
    # Compute LTA Rewards from history
    history["LTA_Rewards"], _ = get_reward_stats(np.array(history["rewards"]).reshape(-1, 1))
    history["LTA_Backlogs"], _ = get_reward_stats(np.array(history["backlogs"]).reshape(-1, 1))
    rollout["LTA_Rewards"] = history["LTA_Rewards"][-rollout_length:]
    rollout["LTA_Backlogs"] = history["LTA_Backlogs"][-rollout_length:]
    for i in range(rollout_length):
        log_dict = {}
        for key in include:
            if len(rollout[key].shape) > 1 and  rollout[key].shape[1] > 1 and log_vectors:
                for j in range(rollout[key].shape[1]):
                    log_dict[f"{glob}/{key}_{j}"] = rollout[key][i,j]
            else:
                log_dict[f"{glob}/{key}"] = rollout[key][i]
        log_dict.update({f"{glob}/step": start_step + i})
        wandb.log(log_dict)


    return history, rollout["LTA_Rewards"][-1]

def log_rollout_summary(rollout, eps, glob = "rollout_summary"):
    log_dict = {}
    log_dict[f"{glob}/mean_reward"] = np.mean(rollout["rewards"])
    log_dict[f"{glob}/intervention_rate"] = np.mean(rollout["interventions"])
    log_dict[f"{glob}/mean_backlog"] = np.mean(rollout["backlogs"])
    log_dict[f"{glob}/eps"] = eps
    wandb.log(log_dict)



def get_reward_stats(reward_vec, startime = None, stoptime = None):
    if startime is None:
        startime = 1
    if stoptime is None:
        stoptime = reward_vec.shape[0] + 1
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

def log_offline_metrics(metrics):
    for key in metrics.keys():
        wandb.define_metric(f"offline/{key}")
    wandb.define_metric(f"offline/step")
    for i in range(metrics["critic_loss"].shape[0]):
        log_dict = {}
        for key in metrics.keys():
            log_dict[f"offline/{key}"] = metrics[key][i]
        log_dict.update({"offline/step": i})
        wandb.log(log_dict)

def log_update_metrics(metrics, eps = None, type = "all"):
    # type can be "all" or "final", refers to logging metrics from all epochs
    # or just the final epoch
    if type == "all":
        for key in metrics.keys():
            if len(metrics[key]) == 0:
                continue
            wandb.define_metric(f"update/{key}-mean", summary = "mean")
            wandb.define_metric(f"update/{key}-max", summary = "max")
            wandb.define_metric(f"update/{key}-min", summary = "min")
        log_dict = {}
        for key in metrics.keys():
            if len(metrics[key]) == 0:
                continue
            log_dict[f"update/{key}-mean"] = np.mean(metrics[key])
            log_dict[f"update/{key}-max"] = np.max(metrics[key])
            log_dict[f"update/{key}-min"] = np.min(metrics[key])
        if eps is not None:
            log_dict.update({"update/eps": eps})
    if type == "final":
        for key in metrics.keys():
            if len(metrics[key]) == 0:
                continue
            wandb.define_metric(f"update/{key}")
        log_dict = {}
        for key in metrics.keys():
            if len(metrics[key]) == 0:
                continue
            log_dict[f"update/{key}"] = metrics[key][-1]
        if eps is not None:
            log_dict.update({"update/eps": eps})

    wandb.log(log_dict)

