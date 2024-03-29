import wandb
import numpy as np
import pandas as pd


def log_rollouts(rollout, history = None,  rolling_statistics = False, rolling_window = 1000, glob = "test", include = ["all"], log_vectors = False):

    md_keys = ["obs", "nn_obs", "next_obs", "next_nn_obs", "flows", "arrivals", "state", "next_state", "actions", "obs_norm_factor"]
    # Per time step logging
    rollout_length = len(rollout["rewards"])
    # Make sure includes is set up correctly
    if isinstance(include, str):
        include = [include]
    if not isinstance(include,list):
        Exception("include must be a list")

    if "all" in include:
        all = rollout.keys()
        include = [key for key in all if key not in md_keys]

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
    history["LTA_Rewards"], _, history[f"Reward_Var(w = {rolling_window})"], history[f"Reward_Mean(w = {rolling_window})"] = get_reward_stats(np.array(history["rewards"]).reshape(-1, 1), rolling_window = rolling_window)
    history["LTA_Backlogs"], _, history[f"Backlog_Var(w = {rolling_window})"], history[f"Backlog_Mean(w = {rolling_window})"] = get_reward_stats(np.array(history["backlogs"]).reshape(-1, 1), rolling_window = rolling_window)
    rollout["Live_LTA_Rewards"] = history["LTA_Rewards"][-rollout_length:]
    rollout["Live_LTA_Backlogs"] = history["LTA_Backlogs"][-rollout_length:]
    rollout[f"RA_Reward_Var(w = {rolling_window})"] = history[f"Reward_Var(w = {rolling_window})"][-rollout_length:]
    rollout[f"RA_Reward_Mean(w = {rolling_window})"] = history[f"Reward_Mean(w = {rolling_window})"][-rollout_length:]
    rollout[f"RA_Backlog_Var(w = {rolling_window})"] = history[f"Backlog_Var(w = {rolling_window})"][-rollout_length:]
    rollout[f"RA_Backlog_Mean(w = {rolling_window})"] = history[f"Backlog_Mean(w = {rolling_window})"][-rollout_length:]

    all = rollout.keys()
    include = [key for key in all if key not in md_keys]

    if rolling_statistics:
        if rolling_window == 0:
            rolling_window = rollout_length
        rollout[f"Reward_Variance(window)={rolling_window}"] = pd.Series(rollout["rewards"][:,0]).rolling(rolling_window, min_periods=1).var().values
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



    return history, rollout["Live_LTA_Rewards"][-1]



def log_rollout_summary(rollout, eps = 0, glob = "test"):

    log = {}
    # What do we want to log in a rollout?
    # 1. Backlog distribution over the entire rollout
    # log["backlog_dist_mean"] = rollout["backlogs"].mean(axis = 0)
    # log["backlog_dist_std"] = rollout["backlogs"].std(axis = 0)

    # 2. LTA_backlog
    LTA_backlog, LTA_Error, LTA_Backlog_Var, LTA_Backlog_Mean = get_reward_stats(np.array(rollout["backlogs"]).reshape(-1,1))
    log["Eps_LTA_backlog"] = LTA_backlog[-1]

    # 3. State distribution
    # log["state_dist_mean"] = rollout["obs"].mean(axis = 0)
    # log["state_dist_std"] = rollout["obs"].std(axis = 0)

    # 4. Action distribution statistics
    # log["action_dist_mean"] = rollout["actions"].mean(axis = 0)
    # log["action_dist_std"] = rollout["actions"].std(axis = 0)


    #5. Simple means
    log["mean_reward"] = np.mean(rollout["rewards"])
    log["var_reward"] = np.var(rollout["rewards"])
    log["mean_backlog"] = np.mean(rollout["backlogs"])
    log["var_backlog"] = np.var(rollout["backlogs"])
    log["intervention_rate"] = np.mean(rollout["interventions"])
    log["obs_norm_factor"] = rollout["obs_norm_factor"]

    log_dict = {}
    for key in log.keys():
        log_dict[f"{glob}/{key}"] = log[key]
    log_dict[f"{glob}/rollout"] = eps
    wandb.log(log_dict)
    return log["Eps_LTA_backlog"]


def log_optimizer_statistics(state_dict, glob = "optimizer"):
    #for p, metrics in state_dict["state"].items():
    pass
def get_reward_stats(reward_vec, startime = None, stoptime = None, rolling_window = 100):
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

    # Get rolling average of the errors from startime to stoptime
    rolling_var = pd.Series(reward_vec[:,0]).rolling(rolling_window, min_periods=1).var().values
    rolling_mean = pd.Series(reward_vec[:,0]).rolling(rolling_window, min_periods=1).mean().values
    return time_averaged_rewards, time_averaged_errors, rolling_var, rolling_mean


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

def log_update_metrics(metrics, eps = None, type = "all", glob = "update"):
    # type can be "all" or "final", refers to logging metrics from all epochs
    # or just the final epoch
    if isinstance(type, str):
        type = [type]
    log_dict = {"eps": eps}
    if "all" in type:
        for key in metrics.keys():
            if len(metrics[key]) == 0:
                continue
            wandb.define_metric(f"{glob}/{key}-mean", summary = "mean")
            wandb.define_metric(f"{glob}/{key}-max", summary = "max")
            wandb.define_metric(f"{glob}/{key}-min", summary = "min")

        for key in metrics.keys():
            if len(metrics[key]) == 0:
                continue
            log_dict[f"{glob}/{key}-mean"] = np.mean(metrics[key])
            log_dict[f"{glob}/{key}-max"] = np.max(metrics[key])
            log_dict[f"{glob}/{key}-min"] = np.min(metrics[key])
        if eps is not None:
            log_dict.update({f"{glob}/eps": eps})
    if "mean" in type:
        for key in metrics.keys():
            if len(metrics[key]) == 0:
                continue
            wandb.define_metric(f"{glob}/{key}-mean", summary = "mean")
        for key in metrics.keys():
            if len(metrics[key]) == 0:
                continue
            log_dict[f"{glob}/{key}-mean"] = np.mean(metrics[key])
        if eps is not None:
            log_dict.update({f"{glob}/eps": eps})

    if "final" in type:
        for key in metrics.keys():
            if len(metrics[key]) == 0:
                continue
            wandb.define_metric(f"{glob}/{key}")
        for key in metrics.keys():
            if len(metrics[key]) == 0:
                continue
            log_dict[f"{glob}/{key}"] = metrics[key][-1]
        if eps is not None:
            log_dict.update({f"{glob}/eps": eps})

    if "minibatches" in type:
        for key in metrics.keys():
            if len(metrics[key]) == 0:
                continue
            wandb.define_metric(f"{glob}/{key}")
        n_mbs = len(metrics["critic_loss"])
        mb_init_step = n_mbs*eps
        for i in range(n_mbs):
            log_dict[f"{glob}/mb"] =  mb_init_step + i
            for key in metrics.keys():
                log_dict[f"{glob}/{key}"] = metrics[key][i]
            wandb.log(log_dict)

