"""
Here we define the buffers that are used in the safety module.
It stores only recent on-policy data.
It must store (state, action, reward, next_state, done) tuples.
... Maybe should store (state, action, reward, next_state, done, intervention) tuples.
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
            state_dim: int,
            action_dim: int,
            buffer_size: int,
            norm_states: bool = False, # state normalization is handled be gym
            device: str = "cpu",
    ):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._buffer_size = buffer_size
        self._device = device

        self._states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self._actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros([buffer_size,1], dtype=np.float32)
        self._backlogs = np.zeros([buffer_size,1], dtype=np.float32)
        self._next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self._dones = np.zeros([buffer_size,1], dtype=np.float32)
        self._interventions = np.zeros([buffer_size,1], dtype=np.float32)
        self.norm_states = False
        if norm_states:
            self._n_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
            self._n_next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)

        self._state_mean = None
        self._state_std = None

        self._pointer = 0
        self._size = 0
        self._full = False
        self._rollout_indices = [] # list of tuples (start, end) of indices of rollouts in buffer

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def add_transitions(self, data: Dict[str, np.ndarray], normalize = False):

        n_transitions = data["obs"].shape[0]
        self._rollout_indices.append((self._pointer, self._pointer + n_transitions))
        if self._pointer + n_transitions > self._buffer_size:
            self._full = True
            self._pointer = 0
            Exception("Buffer overflow!")
        self._states[self._pointer:self._pointer + n_transitions] = self._to_tensor(data["obs"])
        self._next_states[self._pointer:self._pointer + n_transitions] = self._to_tensor(data["next_obs"])
        if self.norm_states:
            self._n_states[self._pointer:self._pointer + n_transitions] = self.normalize_states(self._states[self._pointer:self._pointer + n_transitions])
            self._n_next_states[self._pointer:self._pointer + n_transitions] = self.normalize_states(self._next_states[self._pointer:self._pointer + n_transitions])
        self._actions[self._pointer:self._pointer + n_transitions] = self._to_tensor(data["actions"])
        self._rewards[self._pointer:self._pointer + n_transitions] = self._to_tensor(data["rewards"])
        self._backlogs[self._pointer:self._pointer + n_transitions] = self._to_tensor(data["backlogs"])
        self._dones[self._pointer:self._pointer + n_transitions] = self._to_tensor(data["terminals"])
        self._interventions[self._pointer:self._pointer + n_transitions] = self._to_tensor(data["interventions"])
        self._size += n_transitions
        self._pointer = self._size
        return
    def normalize_states(self, states: np.ndarray):
        # compute rolling meanand std
        if self._state_mean is None:
            self._state_mean = np.mean(states, axis=0)
            self._state_std = np.std(states, axis=0) + 1e-6
        else:
            self._state_mean = 0.99 * self._state_mean + 0.01 * np.mean(states, axis=0)
            self._state_std = 0.99 * self._state_std + 0.01 * np.std(states, axis=0)

        return (states - self._state_mean) / self._state_std

    def get_last_rollout(self):
        rollout_indices = self._rollout_indices[-1]
        if self.norm_states:
            states = self._n_states[rollout_indices[0]:rollout_indices[1]]
            next_states = self._n_next_states[rollout_indices[0]:rollout_indices[1]]
        else:
            states = self._states[rollout_indices[0]:rollout_indices[1]]
            next_states = self._next_states[rollout_indices[0]:rollout_indices[1]]
        actions = self._actions[rollout_indices[0]:rollout_indices[1]]
        rewards = self._rewards[rollout_indices[0]:rollout_indices[1]]
        backlogs = self._backlogs[rollout_indices[0]:rollout_indices[1]]
        dones = self._dones[rollout_indices[0]:rollout_indices[1]]
        interventions = self._interventions[rollout_indices[0]:rollout_indices[1]]
        return {"obs": self._to_tensor(states),
                "actions": self._to_tensor(actions),
                "rewards" : rewards,
                "backlogs": backlogs,
                "next_obs" : self._to_tensor(next_states),
                "dones" :  dones,
                "interventions": interventions}

    def sample(self, batch_size: int):
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        if self.norm_states:
            states = self._n_states[indices]
            next_states = self._n_next_states[indices]
        else:
            states = self._states[indices]
            next_states = self._next_states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        backlogs = self._backlogs[indices]
        dones = self._dones[indices]
        interventions = self._interventions[indices]
        return {"obs": self._to_tensor(states),
                "actions": self._to_tensor(actions),
                "rewards" : rewards,
                "backlogs": backlogs,
                "next_obs" : self._to_tensor(next_states),
                "dones" :  dones,
                "interventions": interventions}



