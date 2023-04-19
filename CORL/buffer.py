import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
TensorBatch = List[torch.Tensor]


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
        self._state_mean = None
        self._state_std = None

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

    def add_transitions(self, data: Dict[str, np.ndarray]):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        print(f"Old Dataset size: {self._size}")

        n_transitions = data["obs"].shape[0]

        self._states[self._pointer:self._pointer+n_transitions] = self._to_tensor(data['obs'])
        self._actions[self._pointer:self._pointer+n_transitions] = self._to_tensor(data["actions"])
        self._rewards[self._pointer:self._pointer+n_transitions] = self._to_tensor(data["rewards"])
        self._next_states[self._pointer:self._pointer+n_transitions] = self._to_tensor(data["next_obs"])
        self._dones[self._pointer:self._pointer+n_transitions] = self._to_tensor(data["terminals"])
        self._size += n_transitions
        self._pointer = self._size

        print(f"New Dataset size: {self._size}")

    def set_state_mean_std(self, mean, std):
        self._state_mean = mean
        self._state_std = std

    def get_state_mean(self):
        return self._state_mean
    def get_state_std(self):
        return self._state_std

