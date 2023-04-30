from typing import Any, Dict, List, Optional, Tuple, Union
import pyrallis
from dataclasses import asdict, dataclass, field
from datetime import datetime
import os


@dataclass
class WandbConfig:
    project: str = "Large Dataset Testing"
    group: str = ""
    name: str = "AWAC"
    checkpoints_path: Optional[str] = "../Saved_Models/AWAC/"

@dataclass
class EnvConfig:
    env_json_path: str = "../JSON/Environment/CrissCross4.json"

    def __post_init__(self):
        self.name = os.path.splitext(os.path.basename(self.env_json_path))[0]

@dataclass
class OfflineDataConfig:
    load_path: str = "offline_data/Env1b_100x1000.data"  # "offline_data\\CrissCross4v2_AWAC-04-18_1546.data"
    save_path: str = "offline_data"
    rollout_length: int = 1000
    num_rollouts: int = 100
    bp_seed: int = 102921
    num_transitions: int = 0
    modified: bool = False # if these parameters are changed at run-time i.e. the data loaded used different parameters during its generation

    def __post_init__(self):
        self.num_transitions = self.num_rollouts * self.rollout_length


@dataclass
class EvalConfig:
    num_steps: int = 1000
    eval_seed: int = 69
    eval_episodes: int = 1
    log: bool = True

@dataclass
class TestConfig:
    n_steps: int = 1000
    test_seed: int = 10101
    test_episodes: int = 10

@dataclass
class OfflineTrainConfig:
    num_epochs: int =  1000
    eval_freq: int = 5 # how often to evaluate the policy during training in terms of epochs
    save_freq: int = 100 # how often to save the model in terms of epochs
    reward_log_freq: int = 10 # how often to log the reward in terms of epochs
    batch_size: int = 1000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 5e-3
    awac_lambda: float = 1.0
    num_samples: int = 0

    def __post_init__(self):
        self.num_samples = self.num_epochs * self.batch_size


@dataclass
class OnlineTrainConfig:
    num_epochs = 1000
    reset_env = False
    record_rollout: bool = True
    save_freq: int = 10
    eval_freq: int = 5
    reward_log_freq: int = 10 # how often to evaluate the policy during training in terms of epochs
    rollout_length = 1000
    batch_size = 1000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 5e-2
    awac_lambda: float = 1.0
    num_samples: int= 0
    num_transitions: int = 0

    def __post_init__(self):
        self.num_samples = self.num_epochs * self.batch_size
        self.num_transitions = self.num_epochs * self.rollout_length

@dataclass
class EnvModifyConfig:
    normalize_reward: bool = False

@dataclass
class NeuralNetworkConfig:
    hidden_dim: int = 256
    deterministic_torch: bool = False

@dataclass
class RunSettings:
    pass


@dataclass
class Config:
    device: str = "cpu"
    buffer_size: int = 2_000_000 # Replay Buffer
    checkpoints_path: str = "Saved_Models"
    normalize_states = True
    save_final_buffer = True

    run: RunSettings = field(default_factory=RunSettings)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    offline_data: OfflineDataConfig = field(default_factory=OfflineDataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    offline_train: OfflineTrainConfig = field(default_factory=OfflineTrainConfig)
    online_train: OnlineTrainConfig = field(default_factory=OnlineTrainConfig)
    env_mod: EnvModifyConfig = field(default_factory=EnvModifyConfig)
    neural_net: NeuralNetworkConfig = field(default_factory=NeuralNetworkConfig)
    test: TestConfig = field(default_factory=TestConfig)


    def __post_init__(self):

        self.name = f"{self.wandb.name}-{datetime.now().strftime('%m-%d_%H%M')}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)
        self.num_samples = self.offline_train.num_samples+self.online_train.num_samples




    def convert_to_dict(self):
        pass


    def print_all(self):
        print("-" * 80)
        print(f"Config: \n{pyrallis.dump(self)}")
        # print(f".env: \n{pyrallis.dump(self.env)}")
        # print(f".env_mod: \n{pyrallis.dump(self.env_mod)}")
        # print(f".neural_net: \n{pyrallis.dump(self.neural_net)}")
        # print(f".offline_data: \n{pyrallis.dump(self.offline_data)}")
        # print(f".eval: \n{pyrallis.dump(self.eval)}")
        # print(f".offline_train: \n{pyrallis.dump(self.offline_train)}")
        # print(f".online_train: \n{pyrallis.dump(self.online_train)}")
        # print(f".test: \n{pyrallis.dump(self.test)}")
        print("-" * 80)
        print("-" * 80)

        # print the contents of each subclass


