import wandb
import uuid
from typing import Dict
import numpy as np
import pandas as pd
import pickle
import os

def wandb_init(config) -> None:
    """Initialize wandb."""

    tags = get_tags(config)
    run = wandb.init(
        config=vars(config),
        project=config.wandb.project,
        group=config.wandb.group,
        name=config.run_name,
        id=str(uuid.uuid4()),
        tags=tags,
        tensorboard=True,
    )
    return run

def save_agent_wandb(agent, mod = ""):
    """Save model to wandb."""
    save_path = os.path.join(wandb.run.dir, f"agent{mod}.pkl")
    pickle.dump(agent, open(save_path, "wb"))
    wandb.save(save_path)

def load_agent_wandb(mod = ""):
    agent = wandb.restore(f"agent{mod}.pkl", run_path = wandb.run.path)
    return agent

def get_tags(config) -> list:

    tags = []

    try: tags.append(config.env.name)
    except: pass

    try: tags.append(config.agent.policy_name)
    except: pass

    try: tags.append(config.agent.actor.type)
    except: pass

    return tags

def log_history(history: Dict[str, np.ndarray], items  = ["obs", "actions"]):
    """Log history to wandb."""
    if "all" in items:
        items = list(history.keys())

    for item in items:
        if item in history:
            df = pd.DataFrame(history[item])
            table = wandb.Table(dataframe=df)
            wandb.log({f"history/{item}": table})


