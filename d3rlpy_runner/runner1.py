import gymnasium as gym
import pyrallis
import numpy as np
from tqdm import tqdm
from dataclasses import asdict, dataclass

from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import AWAC

from param_extractors import parse_env_json
from environment_init import make_MCMH_env
from Environments import MultiClassMultiHop
from NonDRLPolicies.Backpressure import MCMHBackPressurePolicy

def collect_offline_data(env, agent, off_config):

    length = off_config.num_episodes * off_config.max_steps
    obs = np.zeros([length, env.observation_space.shape[0]])
    next_obs = np.zeros([length, env.observation_space.shape[0]])
    rewards = np.zeros([length, 1])
    terminals = np.zeros([length, 1])
    timeouts = np.zeros([length, 1])
    actions = np.zeros([length, env.action_space.shape[0]])

    # Reset the environment
    next_ob, _ = env.reset(seed=off_config.seed)

    for t in tqdm(range(length), desc="Generating Rollout"):
        obs[t] = next_ob
        actions[t] = agent.forward(next_ob)
        next_obs[t], rewards[t], terminals[t], timeouts[t], info = env.step(actions[t])
        next_ob = next_obs[t]
        # if t+1 % off_config.max_steps == 0:
        #     terminals[t] = 1


    return {
        "observations": obs,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
        "next_obs": next_obs,
    }


@dataclass
class OfflineConfig:

    # Offline data collection
    max_steps: int = 100
    seed: int = 42
    num_episodes: int = 10

    #Offline Training
    n_steps: int = 10_000




@dataclass
class Config:
    offline: OfflineConfig = OfflineConfig


if __name__ == "__main__":

    # Config
    config = Config

    # Setup environment
    env_json_path = "../JSON/Environment/Env1a.json"
    env_para = parse_env_json(env_json_path)
    #env = MultiClassMultiHop(env_para) # Non-wrapped
    env = make_MCMH_env(env_para, max_steps= config.offline.max_steps)()

    # Initialize Backpressure Agent
    agent = MCMHBackPressurePolicy(env, M = True)

    # Collect offline data
    dataset = collect_offline_data(env, agent, config.offline)

    # Convert offline data to MDPDataset
    mdp_dataset = MDPDataset(
        observations= dataset["observations"],
        actions = dataset["observations"],
        rewards=dataset['rewards'],
        terminals=dataset['terminals'],
        episode_terminals = dataset["timeouts"],
        discrete_action= False
    )

    # Initialize Agent
    agent = AWAC(action_scaler= "min_max")

    # Pretrain
    agent.fit(
        dataset = mdp_dataset,
        n_steps = config.offline.n_steps
    )









