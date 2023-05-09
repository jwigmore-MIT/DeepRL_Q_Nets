import wandb
import uuid
from typing import Dict
import numpy as np
import pandas as pd

def wandb_init(config) -> None:
    """Initialize wandb."""
    run = wandb.init(
        config=vars(config),
        project=config.wandb.project,
        group=config.wandb.group,
        name=config.wandb.name,
        id=str(uuid.uuid4()),
    )

def log_history(history: Dict[str, np.ndarray], items  = ["obs", "actions"]):
    """Log history to wandb."""
    if "all" in items:
        items = list(history.keys())

    for item in items:
        if item in history:
            df = pd.DataFrame(history[item])
            table = wandb.Table(dataframe=df)
            wandb.log({f"history/{item}": table})


