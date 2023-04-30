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
    env_json_path: str = "../JSON/Environment/Env1b.json"
    flat_state_dim: int = None
    flat_action_dim: int = None
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
    group: str = "EarlyExperimentation"
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
