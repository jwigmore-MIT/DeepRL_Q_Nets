from typing import Any, Dict, List, Optional, Tuple, Union
import pyrallis
from dataclasses import asdict, dataclass
from datetime import datetime
import os


@dataclass
class WandbConfig:
    project: str = "CORL"
    group: str = "Implementation_4_18"
    name: str = "AWAC"
    checkpoints_path: Optional[str] = "../Saved_Models/AWAC/"

@dataclass
class EnvConfig:
    env_json_path: str = "../JSON/Environment/CrissCross4v2.json"

@dataclass
class OfflineDataConfig:
    load_path: str = None #"offline_data\\CrissCross4v2_AWAC-04-18_0958.data"
    save_path: str = "offline_data"
    max_steps: int = 1000
    bp_seed: int = 42
    offline_envs: int = 5

@dataclass
class EvalConfig:
    eval_freq: int = 1000
    max_steps: int = 250
    eval_seed: int = 69
    eval_episodes: int = 3
    log: bool = True

@dataclass
class TestConfig:
    n_steps: int = 500
    test_seed: int = 10101
    test_episodes: int = 10

@dataclass
class OfflineTrainConfig:
    train_steps: int = 10_000
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 5e-3
    awac_lambda: float = 1.0

@dataclass
class EnvModifyConfig:
    normalize_reward: bool = False

@dataclass
class NeuralNetworkConfig:
    hidden_dim: int = 256
    deterministic_torch: bool = False


@dataclass
class Config:
    device: str = "cpu"
    buffer_size: int = 2_000_000 # Replay Buffer
    checkpoints_path: str = "Saved_Models"


    wandb = WandbConfig
    env = EnvConfig
    offline_data = OfflineDataConfig
    eval = EvalConfig
    offline_train = OfflineTrainConfig
    env_mod = EnvModifyConfig
    neural_net = NeuralNetworkConfig
    test = TestConfig


    def __post_init__(self):
        self.name = f"{self.wandb.name}-{datetime.now().strftime('%m-%d_%H%M')}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.wandb.name)

