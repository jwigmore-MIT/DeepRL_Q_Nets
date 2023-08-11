import torch

def observation_checker(obs: torch.Tensor):
    """
    Makes sure each element in the observation vector coming from the environment is within [-1,1]
    """
    if torch.any(obs > 1) or torch.any(obs < -1):
        raise ValueError("Observation value outside of [-1,1]")

