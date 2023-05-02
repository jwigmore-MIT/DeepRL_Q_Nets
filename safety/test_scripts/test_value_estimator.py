import wandb

import safety.roller as roller
import safety.buffers as buffer
import pyrallis
from copy import deepcopy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


from param_extractors import parse_env_json
from safety.buffers import Buffer
from Environments.MultiClassMultiHop import MultiClassMultiHop
from NonDRLPolicies.Backpressure import MCMHBackPressurePolicy
from safety.wrappers import wrap_env
from safety.agent import Actor, Critic, SafeAgent, Interventioner
import torch
from safety.wandb_funcs import wandb_init
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


from typing import Any, Dict, List, Optional, Tuple, Union
import pyrallis
from dataclasses import asdict, dataclass, field
from datetime import datetime
import os

@dataclass
class AgentConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    actor_hidden_dim: int = 64
    critic_hidden_dim: int = 64

@dataclass
class EnvConfig:
    env_json_path: str = "../../JSON/Environment/Env1b.json"
    flat_state_dim: int = None
    flat_action_dim: int = None
    self_normalize_obs: bool = False
    def __post_init__(self):
        self.name = os.path.splitext(os.path.basename(self.env_json_path))[0]

@dataclass
class RunSettings:
    save_freq: int = 1

@dataclass
class IAOPGConfig:
    rollout_length: int = 100
    horizon:int = 10000
    trigger_state: int = 30


    def __post_init__(self):
        self.num_rollouts = self.horizon // self.rollout_length

@dataclass
class WandBConfig:
    project: str = "IAOPG"
    group: str = "ValueFunctionEstimation"
    name: str = "IAOPG-Env1b"
    checkpoints_path: Optional[str] = "../Saved_Models/AWAC/"
@dataclass
class Config:
    device: str = "cpu"
    buffer_size: int = 2_000_000  # Replay Buffer
    checkpoints_path: str = "Saved_Models"

    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    run_settings: RunSettings = field(default_factory=RunSettings)
    iaopg: IAOPGConfig = field(default_factory=IAOPGConfig)
    def print_all(self):
        print("-" * 80)
        print(f"Config: \n{pyrallis.dump(self)}")
        print("-" * 80)
        print("-" * 80)


RUN_SETTINGS = {}

def test_fit_then_run():
    config = Config()
    # set each attribute of config.run based on RUN_SETTINGS
    for k, v in RUN_SETTINGS.items(): setattr(config.run, k, v)
    config.print_all()

    # initialize environment
    parse_env_json(config.env.env_json_path, config)
    base_env = MultiClassMultiHop(config=config)
    env = wrap_env(base_env, self_normalize_obs=config.env.self_normalize_obs, reward_min=-40)

    # initialize replay buffer
    buffer = Buffer(config.env.flat_state_dim, config.env.flat_action_dim, config.buffer_size, config.device)

    # initialize actor, critic, optimizers, and agent
    actor = Actor(config.env.flat_state_dim, config.env.flat_action_dim, config.agent.actor_hidden_dim)
    actor.to(config.device)
    critic = Critic(config.env.flat_state_dim, config.agent.critic_hidden_dim)
    critic.to(config.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.agent.learning_rate)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=config.agent.learning_rate)

    # Initialize Intervention Actor
    safe_actor = MCMHBackPressurePolicy(env, M=True)
    trigger_state = config.iaopg.trigger_state
    interventioner = Interventioner(safe_actor, trigger_state)

    # Initialize Intervention Agent
    agent = SafeAgent(actor, critic, actor_optim, critic_optim, interventioner, config.agent.gamma)

    horizon = config.iaopg.horizon
    num_rollouts = config.iaopg.num_rollouts
    rollout_length = config.iaopg.rollout_length
    pbar = tqdm(range(num_rollouts),ncols=80, desc="Training")

    for eps in pbar:
        # Collect a single rollout
        rollout = roller.gen_rollout(env, agent, length=rollout_length, show_progress= False)

        # Store rollout in bufffer
        buffer.add_transitions(rollout)

        # Fit critic
        if eps < 1:
            batch = buffer.get_last_rollout()
            fit_metrics = agent.fit_critic(batch, fit_epochs=1000)
            pbar.set_postfix("Fitting Critic to first rollout")
        batch = buffer.get_last_rollout()
        update_result = agent.update(batch)
        update_result.update({"eps": eps})

def test_value_function():
    # Collect rollout
    config = Config()
    # set each attribute of config.run based on RUN_SETTINGS
    for k, v in RUN_SETTINGS.items(): setattr(config.run, k, v)
    config.print_all()

    # initialize environment
    parse_env_json(config.env.env_json_path, config)
    base_env = MultiClassMultiHop(config=config)
    env = wrap_env(base_env, self_normalize_obs=config.env.self_normalize_obs, reward_min=-40)

    # initialize replay buffer
    buffer = Buffer(config.env.flat_state_dim, config.env.flat_action_dim, config.buffer_size, config.device)

    # initialize actor, critic, optimizers, and agent
    actor = Actor(config.env.flat_state_dim, config.env.flat_action_dim, config.agent.actor_hidden_dim)
    actor.to(config.device)
    critic = Critic(config.env.flat_state_dim, config.agent.critic_hidden_dim)
    critic.to(config.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.agent.learning_rate)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=config.agent.learning_rate)

    # Initialize Intervention Actor
    safe_actor = MCMHBackPressurePolicy(env, M=True)
    trigger_state = config.iaopg.trigger_state
    interventioner = Interventioner(safe_actor, trigger_state)

    # Initialize Intervention Agent
    agent = SafeAgent(actor, critic, actor_optim, critic_optim, interventioner, config.agent.gamma)

    # Initialize wandb
    # wandb_init(config)
    # wandb.watch(agent.actor, log_freq = 1)
    # wandb.watch(agent.critic, log_freq = 1)
    # For logging trajectories
    log_df = None
    iter = 0
    for n in range(1):
        # Collect a single rollout
        rollout = roller.gen_rollout(env, agent, length = 1000)

        # Store rollout in bufffer
        buffer.add_transitions(rollout)
        #log_df, eps_LTA_reward = roller.log_rollouts(rollout, log_df=log_df, policy_name="IAOPG", glob="online")
        critic_loss =  1000000
        # Train the agent
        batch_run = 0
        losses =  []
        devs = []
        vs = []
        targets = []
        if True:
            batch = buffer.get_last_rollout()
            metrics = agent.fit_critic(batch, fit_epochs = 1000)


        if False:
            while(critic_loss > 5):
                iter +=1
                batch_run +=1
                batch = buffer.get_last_rollout()
                x =  agent.update_critic(batch)
                critic_loss = x["critic_loss"]
                losses.append(critic_loss)

                mean_deviations = (x["values"] - x["targets"]).abs().mean()

                devs.append(mean_deviations.item())

                vs.append(x["values"].mean().item())
                targets.append(x["targets"].mean().item())
                max = (x["values"] - x["targets"]).max()
                std = (x["values"] - x["targets"]).std()
                print(f"Batch: {batch_run}")
                print(f"Critic loss: {critic_loss}")
                print(f"Mean Deviation: {mean_deviations}")
                #print(f"Max Deviation: {max}")
                #print(f"Std Deviation: {std}")
                #print(f"Mean Value: {x['values'].mean()}")
                #print(f"Mean Target: {x['targets'].mean()}")
                print("")
            if True:
                fig, ax = plt.subplots(nrows=3)
                # plot losses, deviations, values and targets
                ax[0].set_title("Losses")
                ax[1].set_title("Deviations")
                ax[2].set_title("Values and Targets")
                ax[0].plot(range(iter-batch_run, iter), losses)
                ax[1].plot(range(iter-batch_run, iter), devs)
                ax[2].plot(range(iter-batch_run, iter), vs, label="Values")
                ax[2].plot(range(iter-batch_run, iter), targets, label="Targets")
                fig.show()
    return metrics




if __name__ == "__main__":
    metrics = test_value_function()


