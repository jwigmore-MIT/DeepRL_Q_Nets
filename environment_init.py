import gymnasium as gym
import numpy as np
from Environments.MultiClassMultiHop import MultiClassMultiHop
# Don't need
def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk

def make_MCMH_env(env_para, max_steps = None, BP = False):
    class FlatActionWrapper(gym.ActionWrapper):
        """
        This action wrapper maps flattened actions <nd.array> back to dictionary
        actions of which the Base environment understands
        """

        def __init__(self, MCMH_env):
            super(FlatActionWrapper, self).__init__(MCMH_env)
            self.action_space= self.flatten_action_space()

        def action(self, action: np.ndarray):
            return self.unflatten_action(action)

    def thunk():
        env = MultiClassMultiHop(env_para)
        if max_steps is not None:
                env = gym.wrappers.TimeLimit(env, max_episode_steps= max_steps)
        env = FlatActionWrapper(env)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        #env = FlatObservationWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        #env = gym.wrappers.NormalizeObservation(env)
        #env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        #env = gym.wrappers.NormalizeReward(env, gamma=args.gamma)
        #env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env


    return thunk