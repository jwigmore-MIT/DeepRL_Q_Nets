"""
Here we define the buffers that are used in the safety module.
It stores only recent on-policy data.
It must store (obs, action, reward, next_obs, done) tuples.
... Maybe should store (obs, action, reward, next_obs, done, intervention) tuples.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy
import pickle
TensorBatch = List[torch.Tensor]

class Buffer:

    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            buffer_size: int,
            device: str = "cpu",
    ):
        if not isinstance(buffer_size, int):
            buffer_size = int(buffer_size)
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._buffer_size = buffer_size
        self._device = device

        self._obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self._actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros([buffer_size,1], dtype=np.float32)
        self._backlogs = np.zeros([buffer_size,1], dtype=np.float32)
        self._next_obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self._dones = np.zeros([buffer_size,1], dtype=np.float32)
        self._interventions = np.zeros([buffer_size,1], dtype=np.float32)
        self.norm_obs = False
        self._nn_obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self._next_nn_obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)

        self._obs_mean = None
        self._obs_std = None

        self._pointer = 0
        self._size = 0
        self._full = False
        self._rollout_indices = [] # list of tuples (start, end) of indices of rollouts in buffer

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def add_transitions(self, data: Dict[str, np.ndarray]):

        n_transitions = data["obs"].shape[0]
        self._rollout_indices.append((self._pointer, self._pointer + n_transitions))
        if self._pointer + n_transitions > self._buffer_size:
            self._full = True
            self._pointer = 0
            Exception("Buffer overflow!")
        self._obs[self._pointer:self._pointer + n_transitions] = self._to_tensor(data["obs"])
        self._next_obs[self._pointer:self._pointer + n_transitions] = self._to_tensor(data["next_obs"])
        self._nn_obs[self._pointer:self._pointer + n_transitions] = self._to_tensor(data["nn_obs"])
        self._next_nn_obs[self._pointer:self._pointer + n_transitions] = self._to_tensor(data["next_nn_obs"])
        self._actions[self._pointer:self._pointer + n_transitions] = self._to_tensor(data["actions"])
        self._rewards[self._pointer:self._pointer + n_transitions] = self._to_tensor(data["rewards"])
        self._backlogs[self._pointer:self._pointer + n_transitions] = self._to_tensor(data["backlogs"])
        self._dones[self._pointer:self._pointer + n_transitions] = self._to_tensor(data["terminals"])
        self._interventions[self._pointer:self._pointer + n_transitions] = self._to_tensor(data["interventions"])
        self._size += n_transitions
        self._pointer = self._size
        return
    def normalize_obs(self, obs: np.ndarray):
        # compute rolling meanand std
        if self._obs_mean is None:
            self._obs_mean = np.mean(obs, axis=0)
            self._obs_std = np.std(obs, axis=0) + 1e-6
        else:
            self._obs_mean = 0.99 * self._obs_mean + 0.01 * np.mean(obs, axis=0)
            self._obs_std = 0.99 * self._obs_std + 0.01 * np.std(obs, axis=0)

        return (obs - self._obs_mean) / self._obs_std

    def get_last_rollout(self):
        rollout_indices = self._rollout_indices[-1]
        if self.norm_obs:
            obs = self._n_obs[rollout_indices[0]:rollout_indices[1]]
            next_obs = self._n_next_obs[rollout_indices[0]:rollout_indices[1]]
        else:
            obs = self._obs[rollout_indices[0]:rollout_indices[1]]
            next_obs = self._next_obs[rollout_indices[0]:rollout_indices[1]]
        nn_obs = self._nn_obs[rollout_indices[0]:rollout_indices[1]]
        actions = self._actions[rollout_indices[0]:rollout_indices[1]]
        rewards = self._rewards[rollout_indices[0]:rollout_indices[1]]
        backlogs = self._backlogs[rollout_indices[0]:rollout_indices[1]]
        dones = self._dones[rollout_indices[0]:rollout_indices[1]]
        interventions = self._interventions[rollout_indices[0]:rollout_indices[1]]
        return {"obs": self._to_tensor(obs),
                "nn_obs": self._to_tensor(nn_obs),
                "actions": self._to_tensor(actions),
                "rewards" : self._to_tensor(rewards),
                "backlogs": backlogs,
                "next_obs" : self._to_tensor(next_obs),
                "dones" :  dones,
                "interventions": self._to_tensor(interventions)}

    def sample(self, batch_size: int):
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        if self.norm_obs:
            obs = self._n_obs[indices]
            next_obs = self._n_next_obs[indices]
        else:
            obs = self._obs[indices]
            next_obs = self._next_obs[indices]
        nn_obs = self._nn_obs[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        backlogs = self._backlogs[indices]
        dones = self._dones[indices]
        interventions = self._interventions[indices]
        return {"obs": self._to_tensor(obs),
                "nn_obs": self._to_tensor(nn_obs),
                "actions": self._to_tensor(actions),
                "rewards" : rewards,
                "backlogs": backlogs,
                "next_obs" : self._to_tensor(next_obs),
                "dones" :  dones,
                "interventions": interventions}



