
import pulp
from copy import deepcopy
import numpy as np


from Environments.MCMH_tools import *
from torch import nn


class RandomPolicy(nn.Module):

    # initialization
    def __init__(self, env):
        super(RandomPolicy, self).__init__()
        self.env = deepcopy(env)
        self.links = env.get_links()
        self.action_format = env.get_action_format()
        self.max_action = env.max_actions


    def forward(self, state : dict, old_state = None):
        return self._forward(state, old_state)

    def act(self, state: dict, old_state=None):
        return self._forward(state, old_state)

    def _forward(self, state : dict, old_state = None):
        f = self.env.action_space.sample()

        return f

