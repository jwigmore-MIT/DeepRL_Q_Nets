# Pyrallis Imports
import pyrallis
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict, dataclass, field
import yaml

@dataclass
class Agent:
    name: str = "agent"
    agent_type: str = "PPO"
    kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Env:
    env_json_path: str
    normalize_obs: bool = False
    normalize_reward: bool = False

@dataclass
class Wandb:
    project: str = "project"

@dataclass
class Learn:
    total_timesteps: int = 100000
    reset_steps: int = 128

@dataclass
class Test:
    n_test_episodes: int = 10
    reset_steps: int = 128
    deterministic: bool = True

@dataclass
class Config:
    root_dir: str = "../"
    seed: int = 5031997
    device: str = "cpu"
    deterministic_torch: bool = True
    tasks: List[str] = field(default_factory=list)

    agent: Agent = field(default_factory=Agent)
    env: Env = field(default_factory=Env)
    wandb: Wandb = field(default_factory=Wandb)
    learn: Learn = field(default_factory=Learn)
    test: Test = field(default_factory=Test)

    def __post_init__(self, yaml_path):
        print("Here")
        pass