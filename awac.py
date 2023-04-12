# Original Library Imports
from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy
from dataclasses import asdict, dataclass
import os
import gymnasium as gym
import random
import uuid
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional
from tqdm import trange
import wandb
import pyrallis

# My Library Imports
from tqdm import tqdm
from datetime import datetime
from utils import get_stats

# My Custom Library Imports
from environment_init import make_MCMH_env
from wandb_utils import wandb_plot_rewards_vs_time

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    project: str = "CORL"
    group: str = "Implementation_4_12"
    name: str = "AWAC"
    checkpoints_path: Optional[str] = "Saved_Models/AWAC/"

    env_json_path: str = "JSON/Environment/CrissCross4v2.json"
    max_steps: int = 1000
    pretrain_seed: int = 42
    pretrain_envs: int = 1
    test_seed: int = 69
    deterministic_torch: bool = False
    device: str = "cpu"

    buffer_size: int = 2_000_000
    num_train_ops: int = 10_000
    batch_size: int = 256
    eval_frequency: int = 1000
    n_test_episodes: int = 2
    normalize_reward: bool = False

    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 5e-3
    awac_lambda: float = 1.0


    def __post_init__(self):
        #self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        self.name = f"{self.name}-{datetime.now().strftime('%m-%d_%H%M')}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)




class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["obs"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["obs"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"])
        self._next_states[:n_transitions] = self._to_tensor(data["next_obs"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self._log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self._min_log_std = min_log_std
        self._max_log_std = max_log_std
        self._min_action = min_action
        self._max_action = max_action

    def _get_policy(self, state: torch.Tensor) -> torch.distributions.Distribution:
        mean = self._mlp(state)
        log_std = self._log_std.clamp(self._min_log_std, self._max_log_std)
        policy = torch.distributions.Normal(mean, log_std.exp())
        return policy

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        policy = self._get_policy(state)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return log_prob

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy = self._get_policy(state)
        action = policy.rsample()
        action.clamp_(self._min_action, self._max_action)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        state_t = torch.tensor(state[None], dtype=torch.float32, device=device)
        policy = self._get_policy(state_t)
        if self._mlp.training:
            action_t = policy.sample()
        else:
            action_t = policy.mean
        action = action_t[0].cpu().numpy()
        return action


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_value = self._mlp(torch.cat([state, action], dim=-1))
        return q_value


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class AdvantageWeightedActorCritic:
    def __init__(
        self,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 5e-3,  # parameter for the soft target update,
        awac_lambda: float = 1.0,
        exp_adv_max: float = 100.0,
    ):
        self._actor = actor
        self._actor_optimizer = actor_optimizer

        self._critic_1 = critic_1
        self._critic_1_optimizer = critic_1_optimizer
        self._target_critic_1 = deepcopy(critic_1)

        self._critic_2 = critic_2
        self._critic_2_optimizer = critic_2_optimizer
        self._target_critic_2 = deepcopy(critic_2)

        self._gamma = gamma
        self._tau = tau
        self._awac_lambda = awac_lambda
        self._exp_adv_max = exp_adv_max

    def _actor_loss(self, states, actions):
        with torch.no_grad():
            pi_action, _ = self._actor(states)
            v = torch.min(
                self._critic_1(states, pi_action), self._critic_2(states, pi_action)
            )

            q = torch.min(
                self._critic_1(states, actions), self._critic_2(states, actions)
            )
            adv = q - v
            weights = torch.clamp_max(
                torch.exp(adv / self._awac_lambda), self._exp_adv_max
            )

        action_log_prob = self._actor.log_prob(states, actions)
        loss = (-action_log_prob * weights).mean()
        return loss

    def _critic_loss(self, states, actions, rewards, dones, next_states):
        with torch.no_grad():
            next_actions, _ = self._actor(next_states)

            q_next = torch.min(
                self._target_critic_1(next_states, next_actions),
                self._target_critic_2(next_states, next_actions),
            )
            q_target = rewards + self._gamma * (1.0 - dones) * q_next

        q1 = self._critic_1(states, actions)
        q2 = self._critic_2(states, actions)

        q1_loss = nn.functional.mse_loss(q1, q_target)
        q2_loss = nn.functional.mse_loss(q2, q_target)
        loss = q1_loss + q2_loss
        return loss

    def _update_critic(self, states, actions, rewards, dones, next_states):
        loss = self._critic_loss(states, actions, rewards, dones, next_states)
        self._critic_1_optimizer.zero_grad()
        self._critic_2_optimizer.zero_grad()
        loss.backward()
        self._critic_1_optimizer.step()
        self._critic_2_optimizer.step()
        return loss.item()

    def _update_actor(self, states, actions):
        loss = self._actor_loss(states, actions)
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()
        return loss.item()

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        critic_loss = self._update_critic(states, actions, rewards, dones, next_states)
        actor_loss = self._update_actor(states, actions)

        soft_update(self._target_critic_1, self._critic_1, self._tau)
        soft_update(self._target_critic_2, self._critic_2, self._tau)

        result = {"critic_loss": critic_loss, "actor_loss": actor_loss}
        return result

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self._actor.state_dict(),
            "critic_1": self._critic_1.state_dict(),
            "critic_2": self._critic_2.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._actor.load_state_dict(state_dict["actor"])
        self._critic_1.load_state_dict(state_dict["critic_1"])
        self._critic_2.load_state_dict(state_dict["critic_2"])


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
    env: gym.Env, actor: Actor, device: str, n_episodes: int, seed: int, record: bool = False,
) -> np.ndarray:
    actor.eval()
    episode_returns = []
    if record:
        rewards = []
    for _ in range(n_episodes):
        (state, info)= env.reset(seed = seed)
        done = False
        episode_return = 0.0
        episode_rewards = []
        while not done:
            action = actor.act(state, device)
            state, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            if record:
                episode_rewards.append(reward.item())
            if terminated or truncated:
                done = True
        episode_returns.append(episode_return)
        if record:
            rewards.append(episode_rewards)
    actor.train()
    if record:

        return np.asarray(episode_rewards).T, np.asarray(rewards).T
    else:
        return np.asarray(episode_returns).T



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
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )


def gen_rollout(env, agent, length):
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

    for t in tqdm(range(length), desc="Generating Rollout"):
        obs[t] = next_ob
        actions[t] = agent.forward(next_ob)
        next_obs[t], rewards[t], terminals[t], timeouts[t], info = env.step(actions[t])
        if "final_info" in info:
            # info won't contain flows nor arrivals
            pass
        else:
            flows[t] = info['flows'][-1]
            arrivals[t] = info['arrivals'][-1]
        next_ob = next_obs[t]
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


def gen_pretrain_dataset(env_para, config, M = True, device='cpu'):
    '''
    Generate the pretraining dataset, currently this is done using BPM as the pretraining "Guide" algorithm
    '''
    from environment_init import make_MCMH_env
    from NonDRLPolicies.Backpressure import MCMHBackPressurePolicy

    # init storage
    rollout_length = config.max_steps
    n_envs = config.pretrain_envs
    traj_dicts = []



    # Initialize BP 'Agent'
    env = make_MCMH_env(env_para)()
    agent = MCMHBackPressurePolicy(env, M = M)

    with torch.no_grad():
        for n_env in range(n_envs):
            traj_dicts.append(gen_rollout(env, agent, rollout_length))
    if n_envs > 1:
        dataset = {}
        for key, value in traj_dicts[0].items():
            dataset[key] = value
            for n_env in range(1, n_envs):
                dataset[key] = np.concatenate([dataset[key], traj_dicts[n_env][key]], axis = 0)
    else:
        dataset = traj_dicts[0]

    return dataset

def wb_plot_eval(rewards: np.ndarray, plot_id: str = "", column_modifier = ""):
    t = list(range(rewards.shape[0]))
    ys = rewards.T.tolist()
    col_names = [f"{column_modifier}{i}" for i in range(rewards.shape[1])]
    wandb.log({plot_id : wandb.plot.line_series(
        xs = t,
        ys = ys,
        keys = col_names,
    )})

def wb_log_LTA(rewards: np.ndarray, series_name: str  =""):
    t = list(range(rewards.shape[0]))
    LTA_rewards, LTA_std = get_stats(rewards.T)
    print("Here")




@pyrallis.wrap()
def train(config: TrainConfig):
    from param_extractors import parse_env_json
    # initialize environment
    env_para = parse_env_json(config.env_json_path, config)
    env = make_MCMH_env(env_para, max_steps= config.max_steps)()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = gen_pretrain_dataset(env_para, config, M=True, device=config.device)

    if config.normalize_reward:
        modify_reward(dataset, config.env_name)

    state_mean, state_std = compute_mean_std(dataset["obs"], eps=1e-3)
    dataset["obs"] = normalize_states(
        dataset["obs"], state_mean, state_std
    )
    dataset["next_obs"] = normalize_states(
        dataset["next_obs"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_dataset(dataset)

    actor_critic_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": config.hidden_dim,
    }

    actor = Actor(**actor_critic_kwargs)
    actor.to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_1 = Critic(**actor_critic_kwargs)
    critic_2 = Critic(**actor_critic_kwargs)
    critic_1.to(config.device)
    critic_2.to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.learning_rate)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.learning_rate)

    awac = AdvantageWeightedActorCritic(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic_1=critic_1,
        critic_1_optimizer=critic_1_optimizer,
        critic_2=critic_2,
        critic_2_optimizer=critic_2_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        awac_lambda=config.awac_lambda,
    )
    wandb_init(asdict(config))

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    eval_LTA_data = {}

    for t in trange(config.num_train_ops, ncols=80):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        update_result = awac.update(batch)
        wandb.log(update_result, step=t)
        if (t + 1) % config.eval_frequency == 0:
            eval_returns, eval_rewards = eval_actor(
                env, actor, config.device, config.n_test_episodes, config.test_seed, record = True
            )

            wandb.log({"eval_returns": eval_returns.mean()}, step=t)
            wandb.log({"eval_LTA_reward": eval_returns.mean() / config.max_steps}, step = t)
            eval_LTA_data[f"Eval_t({t+1})"] = pd.DataFrame(get_stats(eval_rewards), columns= ["LTA", "std"])


            if config.checkpoints_path is not None:
                torch.save(
                    awac.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
    returns, reward = eval_actor(env, actor, config.device, config.n_test_episodes, config.test_seed, record= True)
    wandb.finish()











if __name__ == "__main__":
    train()