# Library imports
import numpy as np
import pandas as pd
import gymnasium as gym
import torch
import os
import random
from tqdm import tqdm
from NonDRLPolicies.Backpressure import MCMHBackPressurePolicy
import wandb

# Custom imports
from environment_init import make_MCMH_env
from agent_init import Agent
from wandb_utils import wandb_plot_avg_vs_time, wandb_plot_rewards_vs_time


def environment_test(env_para, test_length):
    """
    Solely for testing if the environment dynamics are correct
    :param env_para: parameters from JSON to initialize the environment
    :param test_length: length of test trajectory to run
    :return: result: Dataframe - all necessary trajectory history to diagnose issues
    """
    env = make_MCMH_env(env_para, max_steps = test_length)()
    state_keys = env.get_flat_obs_keys()
    action_keys = env.get_flat_action_keys(mod = 'A')
    flow_keys = env.get_flat_action_keys(mod = "F")
    arrival_keys = env.get_flat_arrival_keys(mod = "V")
    next_obs, _ = env.reset()
    obs = np.zeros([test_length, env.observation_space.shape[0]])
    actions = np.zeros([test_length, env.action_space.shape[0]])
    flows = np.zeros([test_length, env.action_space.shape[0]])
    arrivals = np.zeros([test_length, len(arrival_keys)])
    rewards = np.zeros([test_length, 1])
    for t in range(test_length):
        obs[t] = next_obs
        action = env.action_space.sample()
        actions[t] = action
        next_obs, reward, terminated, truncated, info = env.step(action)
        flows[t] = info['flows']
        arrivals[t] = info['arrivals']
        rewards[t]= reward
    pd_rewards = pd.DataFrame(np.array(rewards), columns=['rewards'])
    pd_obs = pd.DataFrame(np.array(obs),
                         columns=state_keys)
    pd_actions = pd.DataFrame(np.array(actions),
                             columns=action_keys)
    pd_flows = pd.DataFrame(np.array(flows),
                             columns=flow_keys)
    pd_arrivals = pd.DataFrame(np.array(arrivals),
                             columns=arrival_keys)
    results = pd.concat([pd_rewards, pd_obs, pd_actions, pd_flows, pd_arrivals], axis=1)
    return results

def test_from_artifact(run, test_args, env_para, artifact, store_history = True):
    model_weight_dir = artifact.download()
    model_dict = {}
    for x in os.listdir(model_weight_dir):
        if x.endswith('.pt'):
            model_dict = torch.load(os.path.join(model_weight_dir, x))
    env = gym.vector.SyncVectorEnv([make_MCMH_env(env_para,max_steps = test_args.num_steps)])
    agent = Agent(env)
    agent.load_state_dict(state_dict=model_dict)
    all_rewards, test_history = agent_test(run, agent, env_para, test_args)
    return agent, all_rewards, test_history

def agent_test(run, agent, env_para, test_args, store_history = True):
    from utils import plot_performance_vs_time

    # Seeding
    random.seed(test_args.seed)
    np.random.seed(test_args.seed)
    torch.manual_seed(test_args.seed)
    torch.backends.cudnn.deterministic = True

    # Torch/Cuda setup
    device = torch.device("cuda" if torch.cuda.is_available() and test_args.cuda else "cpu")


    # Set agent to test
    agent.eval()
    agent.deterministic = False

    # Init test history - stores all trajectory information for debugging
    if store_history:
        test_history = {}
        test_history["Env_seeds"] = []
    else:
        test_history = ""

    # test parameters
    n_envs = test_args.num_envs
    test_length = test_args.num_steps
    # For storing all test rewards
    all_rewards = np.zeros([test_args.num_steps, test_args.num_envs])
    # Loop through number of tests manually
    with torch.no_grad():
        for n_env in tqdm(range(test_args.num_envs), desc = "Running Test on Agent"):
            env = gym.vector.SyncVectorEnv([make_MCMH_env(env_para, max_steps=test_length, test = True)])
            test_seed = np.random.random_integers(0, 1e6)
            next_obs, _ = env.reset(seed = [test_seed] )
            next_obs = torch.Tensor(next_obs).to(device)
            if store_history:
                state_keys = env.envs[0].unwrapped.get_flat_obs_keys()
                action_keys = env.envs[0].unwrapped.get_flat_action_keys(mod='A')
                flow_keys = env.envs[0].unwrapped.get_flat_action_keys(mod="F")
                arrival_keys = env.envs[0].unwrapped.get_flat_arrival_keys(mod="V")
                obs = np.zeros([test_length, env.single_observation_space.shape[0]])
                actions = np.zeros([test_length, env.single_action_space.shape[0]])
                flows = np.zeros([test_length, env.single_action_space.shape[0]])
                arrivals = np.zeros([test_length, len(arrival_keys)])
                test_history["Env_seeds"].append(test_seed)
            for t in range(test_length):
                if store_history:
                    obs[t] = np.array(next_obs)
                action, logprob, _, value  = agent.get_action_and_value(next_obs)
                next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())

                if store_history:
                    if "final_info" in info:
                        # info won't contain flows nor arrivals
                        pass
                    else:
                        flows[t] = info['flows'][-1]
                        arrivals[t] = info['arrivals'][-1]
                    actions[t] = action

                all_rewards[t, n_env] = reward[-1]
                next_obs = torch.Tensor(next_obs).to(device)
            if store_history:
                pd_rewards = pd.DataFrame(np.array(all_rewards[:,n_env]), columns=['rewards'])
                pd_obs = pd.DataFrame(np.array(obs),
                                      columns=state_keys)
                pd_actions = pd.DataFrame(np.array(actions),
                                          columns=action_keys)
                pd_flows = pd.DataFrame(np.array(flows),
                                        columns=flow_keys)
                pd_arrivals = pd.DataFrame(np.array(arrivals),
                                           columns=arrival_keys)
                test_history[n_env] = pd.concat([pd_rewards, pd_obs, pd_actions, pd_flows, pd_arrivals], axis=1)

    fig = plot_performance_vs_time(np.array(all_rewards).T, "AGENT") # need to get policy name
    #wandb.log({"figure": wandb.Image(fig)})
    wandb_plot_rewards_vs_time(all_rewards, "AGENT")

    return all_rewards, test_history


def test_BP(run, env_para, test_args, device='cpu', store_history=True):
    from utils import plot_performance_vs_time
    from environment_init import make_MCMH_env

    # Seeding
    random.seed(test_args.seed)
    np.random.seed(test_args.seed)
    torch.manual_seed(test_args.seed)
    torch.backends.cudnn.deterministic = True

    # Torch device setup
    device = torch.device(device)

    # init storage
    test_length = test_args.num_steps
    n_envs = test_args.num_envs
    all_rewards = np.zeros([test_length, n_envs])
    if store_history:
        test_history = {}
        test_history["Env_seeds"] = []
    else:
        test_history = ""

    # Initialize BP 'Agent'
    env = make_MCMH_env(env_para, max_steps=test_length, test = True)()
    agent = MCMHBackPressurePolicy(env)

    with torch.no_grad():
        for n_env in tqdm(range(n_envs), desc="Running Test on BP Agent"):
            test_seed = np.random.random_integers(1e6)
            next_obs, _ = env.reset(seed=test_seed)
            if store_history:
                state_keys = env.get_flat_obs_keys()
                action_keys = env.get_flat_action_keys(mod='A')
                flow_keys = env.get_flat_action_keys(mod="F")
                arrival_keys = env.get_flat_arrival_keys(mod="V")
                obs = np.zeros([test_length, env.observation_space.shape[0]])
                actions = np.zeros([test_length, env.action_space.shape[0]])
                flows = np.zeros([test_length, env.action_space.shape[0]])
                arrivals = np.zeros([test_length, len(arrival_keys)])
                test_history["Env_seeds"].append(test_seed)

            for t in range(test_length):
                if store_history:
                    obs[t] = next_obs
                action = agent.forward(next_obs)
                next_obs, reward, terminated, truncated, info = env.step(action)

                if store_history:
                    if "final_info" in info:
                        # info won't contain flows nor arrivals
                        pass
                    else:
                        flows[t] = info['flows'][-1]
                        arrivals[t] = info['arrivals'][-1]
                    actions[t] = action
                    all_rewards[t, n_env] = reward[-1]
                if store_history:
                    pd_rewards = pd.DataFrame(np.array(all_rewards[:, n_env]), columns=['rewards'])
                    pd_obs = pd.DataFrame(np.array(obs),
                                          columns=state_keys)
                    pd_actions = pd.DataFrame(np.array(actions),
                                              columns=action_keys)
                    pd_flows = pd.DataFrame(np.array(flows),
                                            columns=flow_keys)
                    pd_arrivals = pd.DataFrame(np.array(arrivals),
                                               columns=arrival_keys)
                    test_history[n_env] = pd.concat([pd_rewards, pd_obs, pd_actions, pd_flows, pd_arrivals], axis=1)

    fig = plot_performance_vs_time(np.array(all_rewards).T, "BP")  # need to get policy name
    # wandb.log({"figure": wandb.Image(fig)})
    wandb_plot_rewards_vs_time(all_rewards, "BP")

    return all_rewards, test_history