import wandb
import uuid

def wandb_init(config) -> None:
    """Initialize wandb."""
    run = wandb.init(
        config=vars(config),
        project=config.wandb.project,
        group=config.wandb.group,
        name=config.wandb.name,
        id=str(uuid.uuid4()),
    )