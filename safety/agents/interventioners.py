import torch.nn as nn
import torch
import numpy as np



class Interventioner(nn.Module):

    def __init__(self, safe_actor, trigger_state = 0):
        super().__init__()
        self.safe_actor = safe_actor
        self.trigger_state = trigger_state



    def check_safety(self, state):
        # In unsafe state, return False, otherwise True

        if np.sum(state) > self.trigger_state:
            return False
        else:
            return True

    def act(self, state, device = None):
        return self.safe_actor.act(state)


class ProbabilisticInterventioner(nn.Module):
    """
    Allow for some probability of intervention in unsafe states near the boundary
    """
    def __init__(self, safe_actor, trigger_state = 0, omega = 1.0):
        super().__init__()
        self.safe_actor = safe_actor
        self.trigger_state = trigger_state
        self.omega = omega

    def check_safety(self, state):
        # In unsafe state, return False, otherwise True
        gap = self.trigger_state - np.sum(state)
        prob = min(1,np.exp(-self.omega * gap))  # probability of intervention
        if np.random.rand() < prob:
            return (False, prob)
        else:
            return (True, prob)


    def act(self, state, device = None):
        return self.safe_actor.act(state)