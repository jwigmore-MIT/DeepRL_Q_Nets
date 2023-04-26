import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from rollout import gen_bp_dataset
from copy import deepcopy
import pickle
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

    def load_dataset(self, data: Dict[str, np.ndarray], normalize_states = False):
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
        if normalize_states:
            self._normalized = True
            self.compute_state_mean_std()
            self._u_states = deepcopy(self._states)
            self._u_next_states = deepcopy(self._next_states)
            self._states = (self._states - self._state_mean) / (self._state_std + 1e-8)
            self._next_states = (self._next_states - self._state_mean) / (self._state_std + 1e-8)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]

        if self._normalized:
            u_next_states = self._u_next_states[indices]
            u_states = self._u_states[indices]
        else:
            u_next_states = None
            u_states = None
        return [states, actions, rewards, next_states, dones, u_states, u_next_states]

    def add_transitions(self, data: Dict[str, np.ndarray], normalize = False, debug = False):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        #print(f"Old Dataset size: {self._size}")
        n_transitions = data["obs"].shape[0]
        if debug:
            state_mean = self._states[:self._pointer,:].mean(0).numpy()
            state_std = (self._states[:self._pointer,:].std(0) + 1e-8).numpy()
            #print(f"Old mean: {state_mean}")
            #print(f"Old std: {state_std}")

        if self._normalized and normalize:
            obs = normalize_states(deepcopy(data["obs"]), self._state_mean, self._state_std)
            next_obs = normalize_states(deepcopy(data["next_obs"]), self._state_mean, self._state_std)
            self._u_states[self._pointer:self._pointer+n_transitions] = self._to_tensor(data["obs"])
            self._u_next_states[self._pointer:self._pointer+n_transitions] = self._to_tensor(data["next_obs"])
        else:
            obs = data["obs"]
            next_obs = data["next_obs"]

        self._states[self._pointer:self._pointer+n_transitions] = self._to_tensor(obs)
        self._actions[self._pointer:self._pointer+n_transitions] = self._to_tensor(data["actions"])
        self._rewards[self._pointer:self._pointer+n_transitions] = self._to_tensor(data["rewards"])
        self._next_states[self._pointer:self._pointer+n_transitions] = self._to_tensor(next_obs)
        self._dones[self._pointer:self._pointer+n_transitions] = self._to_tensor(data["terminals"])
        self._size += n_transitions
        self._pointer = self._size

        if debug:
            new_state_mean = self._states[:self._pointer,:].mean(0).numpy()
            new_state_std = (self._states[:self._pointer,:].std(0) + 1e-8).numpy()
            old_u_mean = self._u_states[:self._pointer - n_transitions, :].mean(0)
            new_u_mean = self._u_states[self._pointer - n_transitions:self._pointer, :].mean(0)

            #print(f"New mean: {state_mean}")
            #print(f"New std: {state_std}")

        return
    def compute_state_mean_std(self):
        self._state_mean = self._states[:self._pointer,:].mean(0).numpy()
        self._state_std = (self._states[:self._pointer,:].std(0) + 1e-8).numpy()

    def get_state_mean(self):
        return self._state_mean
    def get_state_std(self):
        return self._state_std

    def get_size(self):
        return self._size


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std

def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std
def init_replay_buffer(config, how = "gen", env = None):# need to normalize rewards
    """
    Initialize a replay buffer object for storing experience data.

    Parameters
    ----------
    config : object
        A configuration object containing the following attributes:
        - env.flat_state_dim : int
            Integer representing the dimensionality of the flattened state space.
        - env.flat_action_dim : int
            Integer representing the dimensionality of the flattened action space.
        - buffer_size : int
            Integer representing the maximum number of transitions that can be stored in the buffer.
        - device : torch.device object
            Represents the device used for computation.
    empty : bool, optional
        A boolean flag indicating whether to create an empty replay buffer or initialize it with data. Default is False.

    Returns
    -------
    tuple
        A tuple containing the initialized replay buffer object and a reward log array, or None if empty=True.

    Notes
    -----
    This function creates a `ReplayBuffer` object with the specified state and action dimensions, buffer size, and device.
    If `empty=True`, an empty buffer is returned along with a `None` object. If `empty=False`, the function loads a dataset
    into the buffer by either generating it with the `gen_BP_dataset()` function or loading it from a file specified by the
    `config.offline_data.load_path` attribute.

    The function then computes the mean and standard deviation of the state observations in the dataset using the
    `compute_mean_std()` function, and normalizes the state observations and next state observations using the computed mean
    and standard deviation. To enable reward logging, the function identifies the terminal states in the dataset and creates an
    array of rewards corresponding to each episode.

    Finally, the function loads the normalized dataset into the replay buffer using the `load_dataset()` method, sets the mean
    and standard deviation of the state observations using the `set_state_mean_std()` method, and returns the replay buffer object
    along with the reward log array.
    """
    if how not in ["gen", "load", "empty"]:
        raise ValueError("how must be in ['gen', 'load', 'empty']")

    replay_buffer = ReplayBuffer(
        config.env.flat_state_dim,
        config.env.flat_action_dim,
        config.buffer_size,
        config.device,
    )

    if how is "gen":
        dataset, datainfo = gen_bp_dataset(config, M=True, env = env)
    elif how is "load":
        data = pickle.load(open(config.offline_data.load_path, 'rb'))
        dataset = data["dataset"]
        datainfo = data["info"]
    elif how is "empty":
        return replay_buffer, None

    # if config.normalize_states:
    #     state_mean, state_std = compute_mean_std(dataset["obs"], eps=1e-3)
    #     dataset["obs"] = normalize_states(
    #         dataset["obs"], state_mean, state_std
    #     )
    #     dataset["next_obs"] = normalize_states(
    #         dataset["next_obs"], state_mean, state_std
    #     )
    # else:
    #     state_mean, state_std = None, None
    # For reward logging
    terminal_indices = np.where(dataset["terminals"] == 1)[0] + 1
    reward_log = np.array(np.split(dataset["rewards"], terminal_indices[:-1]))[:, :, 0].T

    replay_buffer.load_dataset(dataset, normalize_states = config.normalize_states)
    #replay_buffer.compute_state_mean_std()

    return replay_buffer, reward_log


