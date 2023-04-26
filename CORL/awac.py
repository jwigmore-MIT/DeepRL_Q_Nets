# Original Library Imports
from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy
import os
import random
import uuid
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional
import wandb
import pyrallis
import pickle
import pandas as pd
import torch.nn as nn
from Environments.MultiClassMultiHop import MultiClassMultiHop
# My Library Imports
from tqdm import tqdm
from datetime import datetime

# My Custom Library Imports
from environment_init import make_MCMH_env
from param_extractors import parse_env_json
from configuration import Config
from buffer import init_replay_buffer
from rollout import gen_rollout
from awac_agents import AdvantageWeightedActorCritic, Actor, Critic
from wrappers import wrap_env


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


def test_actor(env, actor, device, n_steps, n_eps, seed, agent_name = "AWAC"):
    """
    Test trained agent, plot average long-term average reward
    :param env:
    :param agent:
    :param length:
    :param n_envs:
    :return:
    """

    episode_rewards, rewards_log = eval_actor(env, actor, device,n_steps, n_eps, seed, log = True)
    log_rewards(rewards_log, agent_name)

def get_reward_stats(reward_vec):
    sum_average_rewards = np.array(
        [np.cumsum(reward_vec[:,i], axis = 0)/np.arange(1,reward_vec.shape[0]+1)
         for i in range(reward_vec.shape[1])]).T

    time_averaged_rewards = sum_average_rewards.mean(axis = 1)
    time_averaged_errors = sum_average_rewards.std(axis=1)
    return time_averaged_rewards, time_averaged_errors

def log_rewards(rewards_log, policy_name = "policy", glob = "test", hidden = False):
    df = pd.DataFrame(rewards_log, columns = [f"Env {e}" for e in range(rewards_log.shape[1])])
    df["LTA_Rewards"], df["LTA_Error"] = get_reward_stats(rewards_log)
    wandb.define_metric(f"{glob}/step")
    wandb.define_metric(f"{glob}/LTA - {policy_name}", summary = "mean", step_metric = f"{glob}/step", hidden = hidden)

    for i in range(df.shape[0]):
        log_dict = {
            f"{glob}/step": i,
            f"{glob}/LTA - {policy_name}": df["LTA_Rewards"][i],
        }
        wandb.log(log_dict)


    #table = wandb.Table(dataframe = df) # columns = [f"Env {e}" for e in range(data.shape[0])])
    #wandb.log({"test_rewards_table": table})
    return df

def log_rollouts(rollout, log_df = None,  policy_name = "policy", glob = "test", hidden = False):
    if log_df is None:
        log_df = pd.DataFrame(rollout["rewards"], columns= ["rewards"])
        wandb.define_metric(f"{glob}/step")
        wandb.define_metric(f"{glob}/LTA - {policy_name}", summary = "mean", step_metric = f"{glob}/step", hidden = hidden)
        start_step = 0
    else:
        new_df = pd.DataFrame(rollout["rewards"], columns=["rewards"])
        # extend log_df with new_df
        log_df = pd.concat([log_df, new_df], axis = 0).reset_index(drop = True)
        start_step = log_df.shape[0] - new_df.shape[0]
    log_df["LTA_Rewards"], log_df["LTA_Error"] = get_reward_stats(np.array(log_df["rewards"]).reshape(-1,1))
    for i in range(start_step, log_df.shape[0]):
        log_dict = {
            f"{glob}/step": i,
            f"{glob}/LTA - {policy_name}": log_df["LTA_Rewards"].loc[i],
        }
        wandb.log(log_dict)
    eps_LTA_reward = log_df["LTA_Rewards"].iloc[-1]
    return log_df, eps_LTA_reward








def wandb_init(config: dict) -> None:
    """Initialize wandb."""
    run = wandb.init(
        config=vars(config),
        project=config.wandb.project,
        group=config.wandb.group,
        name=config.wandb.name,
        id=str(uuid.uuid4()),
    )









def gen_awac(actor_critic_kwargs, train_config):
    actor = Actor(**actor_critic_kwargs)
    actor.to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=train_config.learning_rate)
    critic_1 = Critic(**actor_critic_kwargs)
    critic_2 = Critic(**actor_critic_kwargs)
    critic_1.to(config.device)
    critic_2.to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=train_config.learning_rate)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=train_config.learning_rate)

    awac = AdvantageWeightedActorCritic(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic_1=critic_1,
        critic_1_optimizer=critic_1_optimizer,
        critic_2=critic_2,
        critic_2_optimizer=critic_2_optimizer,
        gamma=train_config.gamma,
        tau=train_config.tau,
        awac_lambda=train_config.awac_lambda,
    )

    return awac

def load_awac_checkpoint(env, config, checkpoint_path, on_off = "on"):
    # Load agent if needed
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    actor_critic_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": config.neural_net.hidden_dim,
    }
    if on_off == "on":
        train_config = config.online_train
    elif on_off == "off":
        train_config = config.offline_train

    awac: AdvantageWeightedActorCritic = gen_awac(actor_critic_kwargs, train_config)

    awac.load_state_dict(torch.load(checkpoint_path))
    return awac



# @pyrallis.wrap()
def offline_train(config: Config, env = None, replay_buffer = None, agent = None):
    ##  Initialize environment
    if env is None:
        parse_env_json(config.env.env_json_path)
        env = make_MCMH_env(config=config)()
    else:
        env = deepcopy(env)

    # initialize replay buffer
    if replay_buffer is None:
        raise Warning("Offline Training: No replay buffer given. Generating based on config")
        replay_buffer = init_replay_buffer(config, how = "gen")

    # Normalize States
    # if config.normalize_states:
    #     replay_buffer.compute_state_mean_std()
    #     env = wrap_env(env, state_mean=replay_buffer.get_state_mean(), state_std=replay_buffer.get_state_std())

    ## Initialize awac agent
    if agent is None:
        actor_critic_kwargs = {
            "state_dim": config.env.flat_state_dim,
            "action_dim": config.env.flat_action_dim,
            "hidden_dim": config.neural_net.hidden_dim,
        }

        agent = gen_awac(actor_critic_kwargs, config)

    # Initialize Weights and Biases
    if wandb.run is None:
        wandb_init(config)
    else:
        wandb.config.update(vars(config))
    # Intialize wandb offline epoch counter
    wandb.define_metric("offline_epoch")

    ## Initialize Checkpoint Path
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    ## Best Checkpoint
    best_checkpoint = None
    ## Offline Training Loop
    pbar = tqdm(range(-(config.offline_train.num_epochs), 0, 1), ncols=80, desc= "Offline Training")
    for epoch in pbar: #trange(config.offline_train.train_steps, ncols=80):

        # Estimate V(s) under current agent
        #v = agent.estimate_V()
        # Sample from replay buffer
        batch = replay_buffer.sample(config.offline_train.batch_size)
        batch = [b.to(config.device) for b in batch]

        # Call awac.update using batch
        update_result = agent.update(batch) # update result contains actor and critic loss values
        update_result.update({"epoch": epoch})




        # Evaluation Loop
        if epoch == 0 or (epoch) % config.offline_train.eval_freq == 0:

            # Test current Policy
            eps_rewards, rewards_log  = eval_actor(
                env, agent._actor, config.device, config.eval.num_steps, config.eval.eval_episodes, config.eval.eval_seed, pbar = pbar, log = config.eval.log
            )
            # Check if best checkpoint
            eps_LTA_reward = eps_rewards.mean() / config.eval.num_steps
            if best_checkpoint is None or eps_LTA_reward > best_checkpoint["LTA_reward"]:
                best_checkpoint = {
                    "epoch": epoch,
                    "LTA_reward": eps_LTA_reward,
                    "agent": deepcopy(agent),
                }
                # Save best checkpoint
                if config.checkpoints_path is not None:
                    torch.save(
                        best_checkpoint["agent"].state_dict(),
                        os.path.join(config.checkpoints_path, "offline_best.pt"),
                    )
            pbar.set_description("Offline Training")
            # Log scores of current policy
            update_result.update({"eval_LTA_reward": eps_LTA_reward})

            if (epoch + 1) % config.offline_train.reward_log_freq == 0:
                log_rewards(rewards_log, policy_name=f"Eval epoch={epoch}", glob = "eval", hidden  = True)

            # Save agent checkpoint
            if config.checkpoints_path is not None:
                torch.save(
                    agent.state_dict(),
                    os.path.join(config.checkpoints_path, f"epoch_{epoch}.pt"),
                )
        wandb.log(update_result)


    return agent, best_checkpoint

def online_train(config: Config, env = None,  replay_buffer = None, agent = None):
    """NEED TO VERIFY THIS ACTUALLY WORKS!"""
    ##  Initialize environment
    if env is None:
        parse_env_json(config.env.env_json_path)
        env = make_MCMH_env(config =  config)()
    else:
        env = deepcopy(env)

    # initialize replay buffer if needed
    if replay_buffer is None:
        replay_buffer = init_replay_buffer(config, empty = True)


    ## Initialize awac agent
    if agent is None:
        actor_critic_kwargs = {
            "state_dim": config.env.flat_state_dim,
            "action_dim": config.env.flat_action_dim,
            "hidden_dim": config.neural_net.hidden_dim,
        }

        agent = gen_awac(actor_critic_kwargs, config)
    agent.grad_clip = False
    # Initialize Weights and Biases
    if wandb.run is None:
        wandb_init(config)
    else:
        wandb.config.update(config)

    # Normalize States
    # if config.normalize_states:
    #     env = wrap_env(env, state_mean=replay_buffer.get_state_mean(), state_std=replay_buffer.get_state_std())


    ## Initialize Checkpoint Path
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)



    # Best Checkpoint
    best_checkpoint = None
    eps_LTA_reward = - np.inf
    # For rollout logging
    log_df = None
    ## Online Training Loop
    pbar = tqdm(range(0,config.online_train.num_epochs, 1), ncols=80, desc="Online Training")
    actor1 = deepcopy(agent._actor)
    for epoch in pbar:
        # Use agent to generate a rollout
        if epoch < 0:
            actor = actor1
        else:
            actor= agent._actor
        rollout = gen_rollout(env, actor, config.online_train.rollout_length, init_reset= config.online_train.reset_env, show_pbar= False)

        # Add rollout to replay buffer
        replay_buffer.add_transitions(deepcopy(rollout), normalize= True, debug = True)
        #replay_buffer.compute_state_mean_std()


        # Sample from replay buffer
        batch = replay_buffer.sample(config.online_train.batch_size)
        batch = [b.to(config.device) for b in batch]

        # Call awac.update using batch
        update_result = agent.update(batch)
        update_result.update({"epoch": epoch})

        # Log rollout rewards
        # if config.online_train.record_rollout:
        #     update_result.update({"rollout_LTA": rollout["rewards"].sum()/config.online_train.rollout_length})

        if config.online_train.reset_env: # Meaning we are evaluating the policy on different environment than the rollout
            # Evaluation Loop
            log_df, _ = log_rollouts(rollout, log_df, policy_name="Online Agent", glob="rollout")
            if epoch == 0 or (epoch) % config.online_train.eval_freq == 0:

                # Test current Policy
                eps_rewards, rewards_log = eval_actor(
                    env, actor, config.device, config.eval.num_steps, config.eval.eval_episodes,
                    config.eval.eval_seed, pbar=pbar, log=config.eval.log
                )
                eps_LTA_reward = eps_rewards.mean() / config.eval.num_steps
                update_result.update({"eval_LTA_reward": eps_LTA_reward})

                if (epoch) % config.online_train.reward_log_freq == 0:
                    log_rewards(rewards_log, policy_name=f"Eval epoch={epoch}", glob="eval", hidden=True)


        else:
            # Log rollout as policy performance
            log_df, eps_LTA_reward = log_rollouts(rollout, log_df, policy_name="Online Agent", glob="rollout")
            #eps_LTA_reward = rollout["rewards"].sum()/config.online_train.rollout_length
            update_result.update({"eval_LTA_reward": eps_LTA_reward})

        if best_checkpoint is None or eps_LTA_reward > best_checkpoint["LTA_reward"]:
            best_checkpoint = {
                "epoch": epoch,
                "LTA_reward": eps_LTA_reward,
                "agent": deepcopy(agent),
            }
            # Save best checkpoint
            if config.checkpoints_path is not None:
                torch.save(
                    best_checkpoint["agent"].state_dict(),
                    os.path.join(config.checkpoints_path, "online_best.pt"),
                )
        pbar.set_description("Online Training")

        if epoch % config.online_train.save_freq:# Save agent checkpoint
            if config.checkpoints_path is not None:
                torch.save(
                    agent.state_dict(),
                    os.path.join(config.checkpoints_path, f"epoch_{epoch}.pt"),
                )
        wandb.log(update_result)


    #wandb.finish()
    return agent, best_checkpoint, replay_buffer


RUN_SETTINGS = {
    "Dataset": ["BPM"], # "BPM", "load", "random"
    "Train": [], #["offline","online"], # ["offline","load","online"]
    "Load": "Saved_Models/AWAC-lambda0.3-04-26_0800/epoch_-45.pt",#"Saved_Models/AWAC-04-24_1624/epoch_-50.pt",
    "Test": []#["from_train", "best"]#["from_train", "best"]
}

MULTI_TEST_PATHS = []
RUN_SETTINGS.update({"Multi_Test": MULTI_TEST_PATHS})

if __name__ == "__main__":




    print(f"Running with settings: {RUN_SETTINGS}")

    ## Get configuration parameters
    config = Config()
    # set each attribute of config.run based on RUN_SETTINGS
    for k, v in RUN_SETTINGS.items(): setattr(config.run, k, v)

    wandb_init(config)
    config.print_all()



    ## Initialize Base Environment
    parse_env_json(config.env.env_json_path, config)
    base_env = MultiClassMultiHop(config = config)
    data_gen_env = wrap_env(base_env, state_mean=None, state_std=None)


    ## Generate or Load the offline dataset
    replay_buffer, batch_reward_logs = init_replay_buffer(config, how = RUN_SETTINGS["Dataset"], env = data_gen_env)

    train_env = wrap_env(deepcopy(base_env), replay_buffer.get_state_mean(), replay_buffer.get_state_std())
    if len(RUN_SETTINGS["Dataset"]) > 1:
        for i in range(len(RUN_SETTINGS["Dataset"])):
            log_rewards(batch_reward_logs[RUN_SETTINGS["Dataset"][i]], RUN_SETTINGS["Dataset"][i])


    ## Create agent
    actor_critic_kwargs = {
        "state_dim": config.env.flat_state_dim,
        "action_dim": config.env.flat_action_dim,
        "hidden_dim": config.neural_net.hidden_dim,
    }

    agent = gen_awac(actor_critic_kwargs, config.offline_train)
    #wandb.watch(agent._actor, log="all", log_freq=100)

    ## Offline Training
    if "offline" in RUN_SETTINGS["Train"]:
        offline_env = wrap_env(MultiClassMultiHop(config=config), state_mean=None, state_std=None)#deepcopy(train_env) #make_MCMH_env(config = config)()
        agent, offline_best = offline_train(config, env = offline_env, replay_buffer = replay_buffer, agent=agent)
        offline_agent = deepcopy(agent)
    elif "load" in RUN_SETTINGS["Train"]:
        agent = load_awac_checkpoint(train_env, config, checkpoint_path= RUN_SETTINGS["Load"])
    if "online" in RUN_SETTINGS["Train"]:
        online_agent = gen_awac(actor_critic_kwargs, config.online_train)
        online_agent.load_state_dict(agent.state_dict())
        online_env = wrap_env(MultiClassMultiHop(config=config), state_mean=None, state_std=None) #make_MCMH_env(config = config, record_stats = config.online_train.reset_env)()
        online_agent, online_best, replay_buffer = online_train(config, env = online_env, replay_buffer=replay_buffer, agent = online_agent)
        online_agent = deepcopy(online_agent)


    test_env = wrap_env(deepcopy(base_env), None, None)
    if RUN_SETTINGS["Test"] == "load":
        agent = load_awac_checkpoint(base_env, config, checkpoint_path= "Saved_Models/AWAC-04-24_2035/epoch_-5.pt")
        test_actor(deepcopy(test_env), agent._actor, device=config.device, n_steps=config.test.n_steps,
                   n_eps=config.test.test_episodes, seed=config.test.test_seed)
    if "from_train" in RUN_SETTINGS["Test"]:
        if "offline" in RUN_SETTINGS["Train"]:
            test_actor(deepcopy(test_env), offline_agent._actor, device=config.device, n_steps=config.test.n_steps,
                   n_eps=config.test.test_episodes, seed=config.test.test_seed, agent_name= "Final Offline")
            if "best" in RUN_SETTINGS["Test"]:
                test_actor(deepcopy(test_env), offline_best["agent"]._actor, device=config.device, n_steps=config.test.n_steps,
                           n_eps=config.test.test_episodes, seed=config.test.test_seed, agent_name= "Best Offline")
        if "online" in RUN_SETTINGS["Train"]:
            test_actor(deepcopy(test_env), online_agent._actor, device=config.device, n_steps=config.test.n_steps,
                   n_eps=config.test.test_episodes, seed=config.test.test_seed, agent_name= "Final Online")
            if "best" in RUN_SETTINGS["Test"]:
                test_actor(deepcopy(test_env), online_best["agent"]._actor, device=config.device, n_steps=config.test.n_steps,
                           n_eps=config.test.test_episodes, seed=config.test.test_seed, agent_name= "Best Online")

    elif RUN_SETTINGS["Test"] == "multi":
        for model_path in MULTI_TEST_PATHS:
            awac = load_awac_checkpoint(base_env, config, checkpoint_path= model_path)
            test_actor(deepcopy(test_env), awac._actor, device=config.device, n_steps=config.test.n_steps,
                          n_eps=config.test.test_episodes, seed=config.test.test_seed)

    wandb.finish()

    # Load Offline Dataset
    # agent = load_awac_checkpoint(base_env, config, checkpoint_path="Saved_Models/AWAC/checkpoint_9999.pt")
    # new_data = gen_rollout(base_env, agent._actor, 100)
    # replay_buffer.add_transitions(new_data)