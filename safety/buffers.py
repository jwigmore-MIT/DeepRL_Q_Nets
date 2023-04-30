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
            device: str = "cpu",
    ):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._buffer_size = buffer_size
        self._device = device

        self._states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self._actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros([buffer_size,1], dtype=np.float32)
        self._next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self._dones = np.zeros([buffer_size,1], dtype=np.float32)
        self._interventions = np.zeros([buffer_size,1], dtype=np.float32)

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
        self._actions[self._pointer:self._pointer + n_transitions] = self._to_tensor(data["actions"])
        self._rewards[self._pointer:self._pointer + n_transitions] = self._to_tensor(data["rewards"])
        self._next_states[self._pointer:self._pointer + n_transitions] = self._to_tensor(data["next_obs"])
        self._dones[self._pointer:self._pointer + n_transitions] = self._to_tensor(data["terminals"])
        self._interventions[self._pointer:self._pointer + n_transitions] = self._to_tensor(data["interventions"])
        self._size += n_transitions
        self._pointer = self._size
        return

    def get_last_rollout(self):
        rollout_indices = self._rollout_indices[-1]
        states = self._states[rollout_indices[0]:rollout_indices[1]]
        actions = self._actions[rollout_indices[0]:rollout_indices[1]]
        rewards = self._rewards[rollout_indices[0]:rollout_indices[1]]
        next_states = self._next_states[rollout_indices[0]:rollout_indices[1]]
        dones = self._dones[rollout_indices[0]:rollout_indices[1]]
        interventions = self._interventions[rollout_indices[0]:rollout_indices[1]]
        return {"obs": self._to_tensor(states),
                "actions": self._to_tensor(actions),
                "rewards" : rewards,
                "next_obs" : self._to_tensor(next_states),
                "dones" :  dones,
                "interventions": interventions}

def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std