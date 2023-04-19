# Original Library Imports
from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy
import os
import random
import uuid
import gym
import numpy as np
import torch
import torch.nn.functional
import wandb
import pyrallis
import pickle
import pandas as pd
import torch.nn as nn

# My Library Imports
from tqdm import tqdm
from datetime import datetime

# My Custom Library Imports
from environment_init import make_MCMH_env
from param_extractors import parse_env_json
from configuration import Config
from buffer import ReplayBuffer
from awac_agents import AdvantageWeightedActorCritic, Actor, Critic




TensorBatch = List[torch.Tensor]




def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    env = gym.wrappers.TransformObservation(env, normalize_state)
    return env


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: Actor, device: str, n_steps: int, n_episodes: int, seed: int, pbar: tqdm = None, log: bool = False
) -> np.ndarray:
    actor.eval()
    episode_rewards = []
    if log:
        reward_log = np.zeros([n_steps, n_episodes])
    for ep in range(n_episodes):
        if pbar is not None:
            pbar.set_description(f"Eval Eps {ep}/{n_episodes}")
        (state, info)= env.reset(seed = seed)
        #done = False
        episode_reward = 0.0
        #while not done:
        for t in range(n_steps):
            action = actor.act(state, device)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            reward_log[t, ep] = reward
            #if terminated or truncated:
            #    done = True
        episode_rewards.append(episode_reward)
    #pbar.set_description(f"Eval Eps {n_episodes}/{n_episodes}")
    actor.train()
    if log:
        return np.asarray(episode_rewards), reward_log
    else:
        return np.asarray(episode_rewards), None


def test_actor(env, actor, device, n_steps, n_eps, seed):
    """
    Test trained agent, plot average long-term average reward
    :param env:
    :param agent:
    :param length:
    :param n_envs:
    :return:
    """

    episode_rewards, rewards_log = eval_actor(env, actor, device,n_steps, n_eps, seed, log = True)
    log_test_rewards(rewards_log, "AWAC")

def get_reward_stats(reward_vec):
    sum_average_rewards = np.array(
        [np.cumsum(reward_vec[:,i], axis = 0)/np.arange(1,reward_vec.shape[0]+1)
         for i in range(reward_vec.shape[1])]).T

    time_averaged_rewards = sum_average_rewards.mean(axis = 1)
    time_averaged_errors = sum_average_rewards.std(axis=1)
    return time_averaged_rewards, time_averaged_errors

def log_test_rewards(rewards_log, policy_name = "policy"):
    # NOT FINISHED NOR USED
    df = pd.DataFrame(rewards_log, columns = [f"Env {e}" for e in range(rewards_log.shape[1])])
    df["LTA_Rewards"], df["LTA_Error"] = get_reward_stats(rewards_log)
    wandb.define_metric("test/step")
    wandb.define_metric("test/LTA", summary = "mean", step = "test/step")

    for i in range(df.shape[0]):
        log_dict = {
            "test/step": i,
            f"test/LTA - {policy_name}": df["LTA_Rewards"][i],
        }
        wandb.log(log_dict)


    #table = wandb.Table(dataframe = df) # columns = [f"Env {e}" for e in range(data.shape[0])])
    #wandb.log({"test_rewards_table": table})
    return df



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


def wandb_init(config: dict) -> None:
    run = wandb.init(
        config=config,
        project=config.wandb.project,
        group=config.wandb.group,
        name=config.wandb.name,
        id=str(uuid.uuid4()),
    )




def gen_rollout(env, agent, length, frac = None):
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


    # Reset the environment
    next_ob, _ = env.reset(seed=seed)

    for t in tqdm(range(length), desc=f"Generating Rollout {frac}"):
        obs[t] = next_ob
        if isinstance(agent, nn.Module):
            actions[t] = agent.act(next_obs[t], config.device)
        else:
            actions[t] = agent.forward(next_ob)
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


def gen_BP_dataset(config, M = True):
    from utils import plot_performance_vs_time
    from environment_init import make_MCMH_env
    from NonDRLPolicies.Backpressure import MCMHBackPressurePolicy

    # init storage
    rollout_length = config.offline_data.max_steps
    n_envs = config.offline_data.offline_envs
    traj_dicts = []

    # Initialize BP 'Agent'
    env = make_MCMH_env(config = config)()
    agent = MCMHBackPressurePolicy(env, M = M)

    with torch.no_grad():
        for n_env in range(n_envs):
            #print(f"Generating Rollout {n_env}/{n_envs}")
            traj_dicts.append(gen_rollout(env, agent, rollout_length, frac = f"{n_env+1}/{n_envs}"))
    if n_envs > 1:
        dataset = {}
        for key, value in traj_dicts[0].items():
            dataset[key] = value
            for n_env in range(1, n_envs):
                dataset[key] = np.concatenate([dataset[key], traj_dicts[n_env][key]], axis = 0)
    else:
        dataset = traj_dicts[0]

    if config.offline_data.save_path is not None:
        datainfo = {
            "max_steps": config.offline_data.max_steps,
            "num_envs": config.offline_data.offline_envs
        }
        data = {"dataset": dataset, "info": datainfo}

        data_name = f"{config.env.name}_{config.name}" # Environment_wandb-name
        data_path =os.path.join(config.offline_data.save_path,data_name +".data")

        pickle.dump(data, open(data_path,'wb'))
        print(f"{data_name} dumped to {data_path}")

    return dataset


def gen_awac(actor_critic_kwargs, config):
    actor = Actor(**actor_critic_kwargs)
    actor.to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.offline_train.learning_rate)
    critic_1 = Critic(**actor_critic_kwargs)
    critic_2 = Critic(**actor_critic_kwargs)
    critic_1.to(config.device)
    critic_2.to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.offline_train.learning_rate)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.offline_train.learning_rate)

    awac = AdvantageWeightedActorCritic(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic_1=critic_1,
        critic_1_optimizer=critic_1_optimizer,
        critic_2=critic_2,
        critic_2_optimizer=critic_2_optimizer,
        gamma=config.offline_train.gamma,
        tau=config.offline_train.tau,
        awac_lambda=config.offline_train.awac_lambda,
    )

    return awac

def load_awac_checkpoint(env, config, checkpoint_path):
    # Load agent if needed
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    actor_critic_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": config.neural_net.hidden_dim,
    }

    awac: AdvantageWeightedActorCritic = gen_awac(actor_critic_kwargs, config)

    awac.load_state_dict(torch.load(checkpoint_path))
    return awac
def init_replay_buffer(config):

    if config.offline_data.load_path is None:
        dataset = gen_BP_dataset(config, M=True)
    else:
        data = pickle.load(open(config.offline_data.load_path, 'rb'))
        dataset = data["dataset"]
        datainfo = data["info"]

    replay_buffer = ReplayBuffer(
        config.env.flat_state_dim,
        config.env.flat_action_dim,
        config.buffer_size,
        config.device,
    )
    state_mean, state_std = compute_mean_std(dataset["obs"], eps=1e-3)
    dataset["obs"] = normalize_states(
        dataset["obs"], state_mean, state_std
    )
    dataset["next_obs"] = normalize_states(
        dataset["next_obs"], state_mean, state_std
    )
    # For reward logging
    terminal_indices = np.where(dataset["terminals"] == 1)[0] + 1
    reward_log = np.array(np.split(dataset["rewards"], terminal_indices[:-1]))[:, :, 0].T

    replay_buffer.load_dataset(dataset)
    replay_buffer.set_state_mean_std(state_mean, state_std)
    return replay_buffer, reward_log


# @pyrallis.wrap()
def offline_train(config: Config, replay_buffer = None):
    ##  Initialize environment
    parse_env_json(config.env.env_json_path)
    env = make_MCMH_env(config =  config)()


    if replay_buffer is None:
        replay_buffer = init_replay_buffer(config)

    ## Log data rewards log

    # Normalize observations
    env = wrap_env(env, state_mean=replay_buffer.get_state_mean(), state_std=replay_buffer.get_state_std())


    ## Initialize awac agent
    actor_critic_kwargs = {
        "state_dim": config.env.flat_state_dim,
        "action_dim": config.env.flat_state_dim,
        "hidden_dim": config.neural_net.hidden_dim,
    }

    awac = gen_awac(actor_critic_kwargs, config)

    # Initialize Weights and Biases
    if wandb.run is None:
        wandb_init(config)
    else:
        wandb.config.update(config)

    ## Initialize Checkpoint Path
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    ## Offline Training Loop
    pbar = tqdm(range(config.offline_train.train_steps), ncols=80, desc= "Offline Training")
    for t in pbar: #trange(config.offline_train.train_steps, ncols=80):
        # Sample from replay buffer
        batch = replay_buffer.sample(config.offline_train.batch_size)
        batch = [b.to(config.device) for b in batch]

        # Call awac.update using batch
        update_result = awac.update(batch)

        # Log result
        wandb.log(update_result, step=t)

        # Evaluation Loop
        if t == 0 or (t + 1) % config.eval.eval_freq == 0:
            # Test current Policy
            eps_rewards, rewards_log  = eval_actor(
                env, awac._actor, config.device, config.eval.max_steps, config.eval.eval_episodes, config.eval.eval_seed, pbar = pbar, log = config.eval.log
            )

            pbar.set_description("Offline Training")
            # Log scores of current policy
            wandb.log({"eval_rewards": eps_rewards.mean()}, step=t)
            wandb.log({"eval_LTA_reward": eps_rewards.mean() / config.eval.max_steps}, step = t)
            if hasattr(env, "get_normalized_score"):
                normalized_eval_scores = env.get_normalized_score(eps_rewards) * 100.0
                wandb.log(
                    {"normalized_eval_score": normalized_eval_scores.mean()}, step=t
                )
            if config.eval.log:
                log_test_rewards(rewards_log, policy_name= f"Eval t={t+1}")

            # Save agent checkpoint
            if config.checkpoints_path is not None:
                torch.save(
                    awac.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )


    wandb.finish()
    return awac

def online_train(config: Config, agent = None, replay_buffer = None):
    pass









def add_transitions_test(buffer, env, agent, n_steps):

    data = gen_rollout(env, agent, n_steps)





if __name__ == "__main__":
    ## Get configuration parameters
    config = Config()
    wandb_init(config)


    ## Initialize Base Environment
    parse_env_json(config.env.env_json_path, config)
    base_env = make_MCMH_env(config = config)()


    ## Generate or Load the offline dataset
    replay_buffer, batch_reward_log = init_replay_buffer(config)
    log_test_rewards(batch_reward_log, "BPM")
    #agent = load_awac_checkpoint(base_env, config, checkpoint_path="Saved_Models/AWAC/checkpoint_9999.pt")
    #new_data = gen_rollout(base_env, agent._actor, 100)
    #replay_buffer.add_transitions(new_data)



    ## Offline Training
    awac = offline_train(config, replay_buffer)

    #awac = load_awac_checkpoint(env, config, checkpoint_path= "Saved_Models/AWAC/checkpoint_9999.pt")
    test_actor(base_env, awac._actor, device = config.device, n_steps=config.test.n_steps, n_eps=config.test.test_episodes, seed=config.test.test_seed)
    wandb.finish()