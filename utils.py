from tqdm import tqdm
#from tianshou.data import Batch
#from tianshou.env import DummyVectorEnv
from copy import deepcopy
from NonDRLPolicies.Backpressure import MCMHBackPressurePolicy
from Environments.MCMH_tools import *
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import re
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def get_stats(reward_vec):
    #average_rewards = reward_vec.mean(axis =0)
    #cumulative_rewards = np.cumsum(average_rewards)
    #time_averaged_rewards =  cumulative_rewards/np.arange(1,cumulative_rewards.shape[1]+1)

    sum_average_rewards = - np.cumsum(reward_vec, axis = 1)/np.arange(1,reward_vec.shape[1]+1)
    time_averaged_rewards = sum_average_rewards.mean(axis = 0)
    time_averaged_errors = sum_average_rewards.std(axis=0)
    return time_averaged_rewards, time_averaged_errors

def plot_performance_vs_time(reward_vec, policy_name, time = None):
    if isinstance(reward_vec, np.ndarray):
        TA_reward, TA_error = get_stats(reward_vec)
        n_tests = reward_vec.shape[0]
        test_episode_length = reward_vec.shape[1]
        fig, ax = plt.subplots()
        plt.title(f"TEST RESULTS: Time-Averaged Queue Backlog for {policy_name} over {n_tests} tests")
        if time is not None:
            policy_name = f"{policy_name}_{time}"
        ax.plot(TA_reward, label = policy_name)
        lower = np.array(TA_reward - TA_error, dtype=np.float64)
        upper = np.array(TA_reward + TA_error, dtype=np.float64)
        ax.fill_between(np.arange(test_episode_length), lower, upper, alpha=0.2)
        plt.xlabel('Timesteps')
        plt.ylabel('Avg. Queue Backlog')
        plt.legend()
        plt.show()
    else:
        n_tests = reward_vec[0].shape[0]
        test_episode_length = reward_vec[0].shape[1]
        fig, ax = plt.subplots()
        plt.title(f"TEST RESULTS: Time-Averaged Queue Backlog")
        for rew_vec, pol_name in zip(reward_vec,policy_name):
            TA_reward, TA_error = get_stats(rew_vec)
            if time is not None:
                pol_name=f"{policy_name}_{time}"
            ax.plot(TA_reward, label=pol_name)
            lower = np.array(TA_reward - TA_error, dtype=np.float64)
            upper = np.array(TA_reward + TA_error, dtype=np.float64)
            ax.fill_between(np.arange(test_episode_length), lower, upper, alpha=0.2)
        plt.xlabel('Timesteps')
        plt.ylabel('Avg. Queue Backlog')
        plt.legend()
        plt.show()
    return fig

def extract_from_buffer(buffer):
    rewards = buffer.rew # array
    terminated = [a or b for a,b in zip(buffer.terminated,buffer.truncated)]# array
    # get index of terminated calls
    end_eps = np.where(terminated)[0]
    # compute different between all terminations to determine episode durations
    diff = np.zeros([end_eps.size-1])
    for i in range(end_eps.size - 1):
        diff[i] = int(abs(end_eps[i] - end_eps[i + 1]))
    if not all(ele == diff[0] for ele in diff):
        raise Exception('ERROR EXTRACTING FROM TEST BUFFER: The episode durations are not the same')
    reward_vec = np.reshape(rewards,[end_eps.size, end_eps[0]+1])

    return reward_vec

class bern_rv:

    def __init__(self, num = 1, prob = 0.5):
        self.num = num
        self.prob = prob

    def sample(self):
        if self.prob == 1:
            return self.num
        else:
            return int(np.random.choice([0, self.num], 1, p=[1 - self.prob, self.prob]))

# def json_dict_to_net_para_dict(json_d: dict):
#     net_para =  {}
#     net_para['nodes'] = deepcopy(json_d['nodes'])
#     net_para['links'] = deepcopy(json_d['link'])
#     net_para['classes'] = {}
#     for cls_num, spec in json_d['classes'].items():
#         net_para['classes'][int(cls_num)] = (spec[0][0], spec[0][1], bern_rv(spec[1][0], spec[1][1]))
#
#     net_para['capacities'] = {}
#     for link,  in json_d['classes'].items():
#         net_para['capacities'][int(class_num)] = bern_rv()

def pretty_json(hp):
  hp2 = {}
  for k, v in hp.items():
    if isinstance(v, list):
          v = str(v)
    hp2[k] = v
  json_hp = json.dumps(hp2, indent=2,)
  txt = "".join("\t" + line for line in json_hp.splitlines(True))
  return txt



# def run_manual_test(policy, test_env, test_para):
#
#     n_test = test_para['n_episodes']
#     n_step  = test_para['episode_length']
#
#     envs = [lambda: test_env for _ in range(n_test)]
#
#     reward_vec = np.zeros([n_test, n_step])
#     bp_flag = isinstance(policy, MCMHBackPressurePolicy)
#     for n in tqdm(range(n_test), desc= f'Running Manual Test'):
#         env = envs[n]()
#         for t in range(n_step):
#             if bp_flag:
#                 state = env.get_state()
#             else:
#                 state = Batch(obs=env.flatten_obs(env.get_state()), info={})
#             action = policy(state)
#             if bp_flag:
#                 act = action['act'][0]
#             else:
#                 act = int(action['act'].data)
#             flatten_obs, reward, terminated, truncated, info = env.step(act)
#             reward_vec[n,t] = reward
#         env.reset()
#
#     return reward_vec






