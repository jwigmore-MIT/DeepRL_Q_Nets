from safety.utils import parse_config
from Environments.MCMH_tools import generate_env


# === Config ===
config_file = "PPO1.yaml"
config = parse_config(config_file)

# === Environment ===
env = generate_env(config, max_steps = config.train.reset_steps)

# === Agent === #



""" TODO:
- Add a "Normalizer" class that normalizes the observations for the agent
    - Environment should output unnormalized observations but the agent should pre-process them using the Normalizer
    - Actors and critics used processed observations (and values?)
    
    
    
"""

