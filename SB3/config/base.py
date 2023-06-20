import pyrallis
from typing import Any, Dict, List, Optional, Tuple, Union, Type
from dataclasses import asdict, dataclass, field
import os

# Stable Baselines 3 import
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule



# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class EnvConfig:
    env_json_path: str = project_root + "/JSON/Environment/Env1/Env1a.json"
    flat_state_dim: int = None
    flat_action_dim: int = None
    def __post_init__(self):
        self.name = os.path.splitext(os.path.basename(self.env_json_path))[0]

class AgentConfig:
    policy_name: str = None # PPO, TD3, etc
    policy: Union[str, Type[ActorCriticPolicy]] = "MlpPolicy"
    policy_kwargs: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class LearningConfig:
    kwargs: Dict[str, Any] = field(default_factory=lambda: {})

@dataclass
class EvalConfig:
    kwargs: Dict[str, Any] = field(default_factory=lambda: {})

@dataclass
class WandbConfig:
    kwargs: Dict[str, Any] = field(default_factory=lambda: {})

@dataclass
class Config:
    seed: int = 5031997

    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def print_all(self):
        print("-" * 80)
        print(f"Config: \n{pyrallis.dump(self)}")
        print("-" * 80)
        print("-" * 80)