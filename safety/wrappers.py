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


def wrap_env(in_env, state_mean = None, state_std = None, eps = 1e-8):
    env = deepcopy(in_env)
    def normalize_state(state):
        return (state - state_mean) / (state_std + eps)
    env = FlatActionWrapper(env)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.ClipAction(env)
    if state_mean is not None and state_std is not None:
        env = gym.wrappers.TransformObservation(env, normalize_state)


    return env



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