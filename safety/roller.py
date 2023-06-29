import numpy as np
import torch
from tqdm import tqdm
import wandb
import pandas as pd

def gen_rollout(env, agent, length = 1000, device = "cpu", pbar_desc = "Generating Rollout", frac = "", show_progress = True, reset = True):

    # Initialize temporary storage
    obs = np.zeros([length, env.observation_space.shape[0]])
    state = np.zeros([length, env.observation_space.shape[0]])
    next_obs = np.zeros([length, env.observation_space.shape[0]])
    next_state = np.zeros([length, env.observation_space.shape[0]])
    nn_obs = np.zeros([length, env.observation_space.shape[0]]) # actor/critic observation i.e. post normalization
    next_nn_obs = np.zeros([length, env.observation_space.shape[0]])
    rewards = np.zeros([length, 1])
    terminals = np.zeros([length, 1])
    interventions = np.zeros([length, 1])
    intervention_prob = np.zeros([length, 1])
    timeouts = np.zeros([length, 1])
    backlogs = np.zeros([length, 1])
    actions = np.zeros([length, env.action_space.shape[0]])
    flows = np.zeros([length, env.action_space.shape[0]])
    arrivals = np.zeros([length, env.flat_qspace_size])


    # Get the current state of the environment
    next_ob = env.get_f_state()
    # check if NormalizeObservation wrapper is used

    if show_progress:
        pbar = tqdm(range(length), desc=f"{pbar_desc} {frac}")
    else:
        pbar = range(length)

    # Perform Rollout
    for t in pbar:
        obs[t] = next_ob

        """ Should look something like this for the PPO agent
        action, actor_obs = agent.act(obs[t], device) # where actor_obs is normalized if using a normalization agent
    
        """
        if hasattr(agent,"safe_agent") and agent.safe_agent:
            actions[t], nn_obs[t], interventions[t], intervention_prob[t] = agent.act(obs[t], device)
        elif agent.__str__() == "SafeAgent()":
            actions[t], nn_obs[t], interventions[t], intervention_prob[t] = agent.act(obs[t], device)
            next_obs[t], rewards[t], terminals[t], timeouts[t], info = env.step(actions[t])
            next_nn_obs[t] = agent.obs_normalizer.normalize(next_obs[t], update=True)

        elif agent.__str__() == 'MCMHBackPressurePolicy()':
            # BP Agent
            actions[t] = agent.act(obs[t], device)
            interventions[t] = True
            intervention_prob[t] = 1
            next_obs[t], rewards[t], terminals[t], timeouts[t], info = env.step(actions[t])
        else:
            # NN Agent
            actions[t], nn_obs[t] = agent.act(obs[t], device)
            interventions[t] = False
            intervention_prob[t] = 0
            next_obs[t], rewards[t], terminals[t], timeouts[t], info = env.step(actions[t])
            next_nn_obs[t] = agent.obs_normalizer.normalize(next_obs[t], update=True)



        if "final_info" in info:
            # info won't contain flows nor arrivals
            pass
        else:
            flows[t] = info['flows'][-1]
            arrivals[t] = info['arrivals'][-1]
            backlogs[t] = info['backlog'][-1]
            state[t] = info['state'][-1]
            next_state[t] = info['next_state'][-1]
        next_ob = next_obs[t]

    if reset:
        env.reset()
        terminals[t] = 1
    return {
        "obs": obs,
        "nn_obs": nn_obs,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
        "next_obs": next_obs,
        "next_nn_obs": next_nn_obs,
        "flows": flows,
        "arrivals": arrivals,
        "interventions": interventions,
        "intervention_prob": intervention_prob,
        "backlogs": backlogs,
        "state": state,
        "next_state": next_state
    }

def gen_step(env, agent, random = False, device = "cpu"):

    obs = env.get_f_state()
    if random:
        actions = env.action_space.sample()
        nn_obs = agent.obs_normalizer.normalize(obs, update=True)
        interventions = False
        intervention_prob = 0
        next_obs, rewards, terminals, timeouts, info = env.step(actions)
        next_nn_obs = agent.obs_normalizer.normalize(next_obs, update=True)

    elif agent.__str__() == "SafeAgent()":
        actions, nn_obs, interventions, intervention_prob = agent.act(obs, device)
        next_obs, rewards, terminals, timeouts, info = env.step(actions)
        next_nn_obs = agent.obs_normalizer.normalize(next_obs, update=True)
    else:
        actions, nn_obs = agent.act(obs, device)
        interventions = False
        intervention_prob = 0
        next_obs, rewards, terminals, timeouts, info = env.step(actions)
        next_nn_obs = agent.obs_normalizer.normalize(next_obs, update=True)
    if "final_info" in info:
        # info won't contain flows nor arrivals
        pass
    else:
        flows = info['flows'][-1]
        arrivals = info['arrivals'][-1]
        backlogs = info['backlog'][-1]
        state = info['state'][-1]
        next_state = info['next_state'][-1]


    return {
        "obs": obs,
        "nn_obs": nn_obs,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
        "next_obs": next_obs,
        "next_nn_obs": next_nn_obs,
        "flows": flows,
        "arrivals": arrivals,
        "interventions": interventions,
        "intervention_prob": intervention_prob,
        "backlogs": backlogs,
        "state": state,
        "next_state": next_state
    }



def get_reward_stats(reward_vec):
    sum_average_rewards = np.array(
        [np.cumsum(reward_vec[:,i], axis = 0)/np.arange(1,reward_vec.shape[0]+1)
         for i in range(reward_vec.shape[1])]).T

    time_averaged_rewards = sum_average_rewards.mean(axis = 1)
    time_averaged_errors = sum_average_rewards.std(axis=1)
    return time_averaged_rewards, time_averaged_errors
def log_rollouts(rollout, history = None,  policy_name = "policy", glob = "test", include = ["all"], log_vectors = False):

    rollout["LTA_Rewards"], rollout["LTA_Error"] = get_reward_stats(np.array(rollout["rewards"]).reshape(-1,1))
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