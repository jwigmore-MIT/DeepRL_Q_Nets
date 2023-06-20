from functools import singledispatch
import gymnasium as gym
# from Environments.MultiClassMultiHop import MultiClassMultiHop
from stable_baselines3.common.monitor import Monitor
import numpy as np
from dataclasses import dataclass
from param_extractors import parse_env_json


@dataclass
class Config:
    pass



@singledispatch
def keys_to_strings(ob):
    return ob

@keys_to_strings.register
def _handle_dict(ob: dict):
    return {str(k): keys_to_strings(v) for k, v in ob.items()}

@keys_to_strings.register
def _handle_list(ob: list):
    return [keys_to_strings(v) for v in ob]

@singledispatch
def keys_to_ints(ob):
    return ob

@singledispatch
def keys_to_tup(ob):
    return ob

@keys_to_tup.register
def _handle_dict(ob: dict):
    return {tuple(k): v for k, v in ob.items}

@keys_to_ints.register
def _handle_dict(ob: dict):
    return {int(k): keys_to_ints(v) for k,v in ob.items()}

@keys_to_ints.register
def _handle_list(ob: list):
    return [keys_to_ints(v) for v in ob]


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


class StepLoggingWrapper(gym.Wrapper):
    "Custom wrapper to log some of the outputs of the step function"

    def __init__(self, env, log_keys = ["backlog"], filename = "log.csv"):
        super(StepLoggingWrapper, self).__init__(env)
        self.log_keys = log_keys
        self.filename = filename
        self.eps = 0
        self.log ={self.eps: {}}
        for key in self.log_keys:
            self.log[self.eps][key] = []



    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        for key in self.log_keys:
            self.log[self.eps][key].extend(info[key])
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.eps += 1
        self.log[self.eps] = {}
        for key in self.log_keys:
            self.log[self.eps][key] = []
        return self.env.reset(**kwargs)

    def save_log(self):
        import csv
        with open(self.filename, 'w') as f:
            writer = csv.writer(f)
            for key, value in self.log.items():
                writer.writerow([key, value])



def generate_env(config: Config, monitor_settings = None, backpressure = False):
    """
    Generates the environment and applies the wrappers
    """
    from Environments.MultiClassMultiHop import MultiClassMultiHop
    parse_env_json(config.root_dir + config.env.env_json_path, config)
    env = MultiClassMultiHop(config=config)
    env = gym.wrappers.time_limit.TimeLimit(env, max_episode_steps=100)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if monitor_settings is not None:
        monitor_settings["info_keywords"] = tuple(monitor_settings["info_keywords"])
        env = Monitor(env, **monitor_settings)
    env = FlatActionWrapper(env)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.ClipAction(env)
    if not backpressure:
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.NormalizeReward(env)
    env = StepLoggingWrapper(env)
    return env


