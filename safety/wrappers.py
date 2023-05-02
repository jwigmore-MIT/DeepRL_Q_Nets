import gymnasium as gym
import numpy as np
from copy import deepcopy
class FlatActionWrapper(gym.ActionWrapper):
    """
    This action wrapper maps flattened actions <nd.array> back to dictionary
    actions of which the Base environment understands
    """

    def __init__(self, MCMH_env):
        super(FlatActionWrapper, self).__init__(MCMH_env)
        self.action_space = self.flatten_action_space()

    def action(self, action: np.ndarray):
        return self.unflatten_action(action)


def wrap_env(in_env, **kwargs): #state_mean = None, state_std = None, self_normalize = False, state_max = None, eps = 1e-8):
    env = deepcopy(in_env)
    def normalize_state(state):
        return (state - state_mean) / (state_std + eps)
    def normalize_reward(reward):
        return (reward - reward_min/2) / (-reward_min/2 + eps)
    state_mean = kwargs.get("state_mean", None)
    state_std = kwargs.get("state_std", None)
    eps = kwargs.get("eps", 1e-8)
    self_normalize_obs = kwargs.get("self_normalize_obs", False)
    self_normalize_rew = kwargs.get("self_normalize_rew", False)
    state_max = kwargs.get("state_max", None)
    reward_min = kwargs.get("reward_min", None)



    env = FlatActionWrapper(env)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.ClipAction(env)
    if self_normalize_obs:
        env = gym.wrappers.NormalizeObservation(env)
    if self_normalize_rew:
        env = gym.wrappers.NormalizeReward(env)
    if state_mean is not None and state_std is not None:
        env = gym.wrappers.TransformObservation(env, normalize_state)
    if state_max is not None:
        env = gym.wrappers.ClipObservation(env, -state_max, state_max)
    if reward_min is not None:
        env = gym.wrappers.TransformReward(env, normalize_reward)

    return env

def reward_wrapper(env, min_reward, eps = 1e-8, constant = 1):
    # Constant included to allow for normalizing by a constant such as episode length
    def mod_reward(reward):
        return (reward - min_reward/2) / (-min_reward/2 + eps) / constant
    return gym.wrappers.TransformReward(env, mod_reward)



def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)
def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0