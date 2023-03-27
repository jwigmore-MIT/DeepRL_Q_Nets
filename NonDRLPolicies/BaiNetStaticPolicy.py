# Imports
import pulp
from copy import deepcopy
import numpy as np
from Environments.MCMH_tools import *
from torch import nn


class BaiNetStaticPolicy(nn.Module):

    def __init__(self, env, map_to_discrete = False, tianshou = False):
        super(BaiNetStaticPolicy, self).__init__()
        self.env = deepcopy(env)
        self.links = env.get_links()
        self.action_format = env.get_action_format()

    def forward(self, flat_state: np.ndarray, old_state= None):
        action = {(1, 2):  {1: 0, 2: 0, 3: 0},
                  (1, 6):  {1: 5, 2: 0, 3: 0}, #(1,6, C1)
                  (2, 3):  {1: 0, 2: 0, 3: 0},
                  (3, 7):  {1: 0, 2: 0, 3: 0},
                  (3, 8):  {1: 0, 2: 0, 3: 0},
                  (3, 4):  {1: 0, 2: 5, 3: 0}, #(3,4, C2)
                  (4, 5):  {1: 0, 2: 5, 3: 0}, #(4,5, C2)
                  (5, 16): {1: 0, 2: 5, 3: 0}, # (5,16,C2)
                  (6, 11): {1: 5, 2: 0, 3: 0}, #(6,11, C1)
                  (6, 7):  {1: 0, 2: 0, 3: 0},
                  (7, 12): {1: 0, 2: 0, 3: 0},
                  (7, 8):  {1: 0, 2: 0, 3: 5}, # (7,8, C3)
                  (8, 9):  {1: 0, 2: 0, 3: 5}, #(8,9, C3)
                  (9, 10): {1: 0, 2: 0, 3: 1}, #
                  (9, 15): {1: 0, 2: 0, 3: 2},
                  (10, 15):{1: 0, 2: 0, 3: 5},
                  (10, 5): {1: 0, 2: 0, 3: 0},
                  (11, 16):{1: 5, 2: 0, 3: 0}, #(11,16, C1)
                  (12, 11):{1: 0, 2: 0, 3: 0},
                  (12, 7): {1: 0, 2: 0, 3: 5}, #(12,7, C3)
                  (13, 12):{1: 0, 2: 0, 3: 3}, #(13, 12, C3) = 3
                  (13, 9): {1: 0, 2: 0, 3: 3},
                  (13, 14):{1: 0, 2: 0, 3: 3},
                  (14, 15):{1: 0, 2: 0, 3: 5},
                  (15, 16):{1: 0, 2: 0, 3: 5}}

        flat_action = np.array([[0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0,
        5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 1, 0, 0,
        2, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 3, 0, 0, 3,
        0, 0, 3, 0, 0, 5, 0, 0, 5]])

        return flat_action[0,:]
