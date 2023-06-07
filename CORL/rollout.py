import numpy as np
from tqdm import tqdm
import torch
import os
import pickle
from copy import deepcopy
def gen_rollout(env, agent, length, device = "cpu", init_reset = True, frac = None, show_pbar = True, pbar_desc = "Generating Rollout"):
    """

    :param env:
    :param rollout_length:
    :return: trajectory = {obs: , actions:, rewards:, terminals: , timeouts, next_obs:  }
    """
    # Seeding : np.random should be seeded outside of method
    seed = np.random.randint(1,1e6)
    # Initialize trajectory storage
    obs = np.zeros([length, env.observation_space.shape[0]])
    next_obs = np.zeros([length, env.observation_space.shape[0]])
    rewards = np.zeros([length, 1])
    terminals = np.zeros([length, 1])
    timeouts = np.zeros([length, 1])
    actions = np.zeros([length, env.action_space.shape[0]])
    flows = np.zeros([length, env.action_space.shape[0]])
    arrivals = np.zeros([length, env.flat_qspace_size])


    # If needed, reset environment
    if init_reset:
        next_ob, _ = env.reset(seed=seed)
    else:
        next_ob = env.get_f_state()
    # Setup progress bar
    if show_pbar:
        pbar = tqdm(range(length), desc=f"{pbar_desc} {frac}")
    else:
        pbar = range(length)

    # Perform Rollout
    for t in pbar:
        obs[t] = next_ob
        actions[t] = agent.act(obs[t], device)
        next_obs[t], rewards[t], terminals[t], timeouts[t], info = env.step(actions[t])
        if "final_info" in info:
            # info won't contain flows nor arrivals
            pass
        else:
            flows[t] = info['flows'][-1]
            arrivals[t] = info['arrivals'][-1]
        next_ob = next_obs[t]
    terminals[t] = 1
    return {
        "obs": obs,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
        "next_obs": next_obs,
        "flows": flows,
        "arrivals": arrivals,
        "seed": np.array([seed])
    }


def gen_bp_dataset(config, M = True, env = None, pbar_desc = "Generating BP Dataset"):
    from environment_init import make_MCMH_env
    from NonDRLPolicies.Backpressure import MCMHBackPressurePolicy

    # init storage
    rollout_length = config.offline_data.rollout_length
    num_rollouts = config.offline_data.num_rollouts
    traj_dicts = []

    # Initialize BP 'Agent'
    env = deepcopy(env)
    agent = MCMHBackPressurePolicy(env, M = M)

    with torch.no_grad():
        for n_rollout in range(num_rollouts):
            traj_dicts.append(gen_rollout(env, agent, rollout_length, pbar_desc= pbar_desc, frac= f"{n_rollout + 1}/{num_rollouts}"))
    if num_rollouts > 1:
        dataset = {}
        for key, value in traj_dicts[0].items():
            dataset[key] = value
            for n_rollout in range(1, num_rollouts):
                dataset[key] = np.concatenate([dataset[key], traj_dicts[n_rollout][key]], axis = 0)
    else:
        dataset = traj_dicts[0]

    if config.offline_data.save_path is not None:
        datainfo = {
            "rollout_length": config.offline_data.rollout_length,
            "num_rollouts": config.offline_data.num_rollouts,
        }
        data = {"dataset": dataset, "info": datainfo}

        data_name = f"{config.env.name}_{config.name}" # Environment_wandb-name
        data_path =os.path.join(config.offline_data.save_path,data_name +".data")

        pickle.dump(data, open(data_path,'wb'))
        print(f"{data_name} dumped to {data_path}")

    return dataset, datainfo

def gen_random_action_dataset(config, env = None, pbar_desc = "Generating Random Action Dataset"):
    from environment_init import make_MCMH_env
    from NonDRLPolicies.Randomized_policy import RandomPolicy

    # init storage
    rollout_length = config.offline_data.rollout_length
    num_rollouts = config.offline_data.num_rollouts
    traj_dicts = []

    # Initialize BP 'Agent'
    env = deepcopy(env)
    agent = RandomPolicy(env)

    with torch.no_grad():
        for n_rollout in range(num_rollouts):
            traj_dicts.append(gen_rollout(env, agent, rollout_length, pbar_desc = pbar_desc, frac = f"{n_rollout + 1}/{num_rollouts}"))
    if num_rollouts > 1:
        dataset = {}
        for key, value in traj_dicts[0].items():
            dataset[key] = value
            for n_rollout in range(1, num_rollouts):
                dataset[key] = np.concatenate([dataset[key], traj_dicts[n_rollout][key]], axis = 0)
    else:
        dataset = traj_dicts[0]

    if config.offline_data.save_path is not None:
        datainfo = {
            "rollout_length": config.offline_data.rollout_length,
            "num_rollouts": config.offline_data.num_rollouts,
        }
        data = {"dataset": dataset, "info": datainfo}

        data_name = f"{config.env.name}_{config.name}" # Environment_wandb-name
        data_path =os.path.join(config.offline_data.save_path,data_name +".data")

        pickle.dump(data, open(data_path,'wb'))
        print(f"{data_name} dumped to {data_path}")

    return dataset, datainfo