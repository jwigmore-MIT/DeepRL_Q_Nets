import gymnasium as gym
import numpy as np
from Environments.MultiClassMultiHop import MultiClassMultiHop


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


class LongTermAverageRewardWrapper(gym.RewardWrapper):

    def __init__(self, env, max_steps):
        super().__init__(env)
        self.t = 0
        self.max_steps = max_steps

    def reward(self, reward):
        if self.t >= self.max_steps:
            self.t = 0
        self.t += 1
        return reward / self.t


class NoStatePenaltyWrapper(gym.RewardWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.last_reward = 0

    def reward(self, reward):
        if self.last_reward < reward:
            mod_reward = reward - self.last_reward
        else:
            mod_reward = reward
        self.last_reward = reward
        return mod_reward


class ClipRewardWrapper(gym.RewardWrapper):

    def __init__(self, env, min_reward):
        super().__init__(env)
        self.min_reward = min_reward

    def reward(self, reward):
        return max(self.min_reward, reward)


class DeliveredRewardsWrapper(gym.RewardWrapper):

    def __init__(self, env):
        super().__init__(env)

    def reward(self, rewards):
        return self.unwrapped.delivered


class HorizonScaledRewardWrapper(gym.RewardWrapper):

    def __init__(self, env, max_steps):
        super().__init__(env)
        self.max_steps = max_steps

    def reward(self, reward):
        return reward / self.max_steps

def make_MCMH_env(env_para, max_steps = None, time_scaled = False,
                  moving_average = False, test = False,
                  no_state_penalty = False, min_reward = False,
                  delivered_rewards = False, horizon_scaled = False):


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
        if time_scaled:
            env = LongTermAverageRewardWrapper(env, max_steps)
        if moving_average:
            env = gym.wrappers.NormalizeReward(env, gamma=0.99)
        if no_state_penalty:
            env = NoStatePenaltyWrapper(env)
        if min_reward:
            env = ClipRewardWrapper(env, min_reward)
        if delivered_rewards:
            env = DeliveredRewardsWrapper(env)
        if horizon_scaled:
            env = HorizonScaledRewardWrapper(env, max_steps)
        #env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env


    return thunk

def register_env(env_para):
    '''
    UNFINISHED
    '''
    from gymnasium.envs.registration import register
    env = MultiClassMultiHop(env_para)

    register(
        id=f"Environements/Registered/{env_para['name']}",
        entry_point="gym_examples.envs:GridWorldEnv",
    )




def wrap_env(env, max_steps = None):
    '''
     UNFINISHED
    '''
    if max_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    env = FlatActionWrapper(env)
    env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
    # env = FlatObservationWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    return env

