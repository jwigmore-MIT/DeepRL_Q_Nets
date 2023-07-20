# Imports
import pulp
from copy import deepcopy
import numpy as np
from Environments.MCMH_tools import *
from torch import nn


class BaiNetStaticPolicy(nn.Module):

    def __init__(self, env):
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


class CrissCross2StaticPolicy(nn.Module):
    def __init__(self, env):
        super(CrissCross2StaticPolicy, self).__init__()
        self.env = deepcopy(env)
        self.links = env.get_links()
        self.action_format = env.get_action_format()

    def forward(self, flat_state: np.ndarray, old_state= None):

        f = self.env.get_action_format()
        # Class 1
        f[1,3][1] = 3
        f[3,2][1] = 3
        f[2,5][1] = 3

        # Class 2
        f[1,2][2] =2
        f[2,6][2] = 3

        # Class 3
        f[1,3][3] = 1
        f[3,7][3] = 2

        # Class 4
        f[1,4][4] = 2
        f[1,3][4] = 1
        f[4,8][4] = 3
        f[3,4][4]= 2

        #flat_f = self.env.flatten_action(f)
        flat_f = np.array([[0, 2, 0, 0, 3, 0, 1, 1, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3,
        0, 0, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 3]])

        return flat_f[0,:]


class DiamondOptimalPolicy(nn.Module):
    def __init__(self, env):
        super(DiamondOptimalPolicy, self).__init__()
        self.env = deepcopy(env)
        self.links = env.get_links()
        self.action_format = env.get_action_format()
        # check if env has attribute 'unwrapped'
        self.optimal_action = np.ones([1,env.action_space.shape[0]])*env.action_space.high[-1]


    def forward(self, state: dict, old_state=None):
        return self._forward()

    def act(self, state: dict, old_state=None):
        return self._forward()

    def _forward(self):
        return self.optimal_action


class JoinTheShortestQueuePolicy:

    def __init__(self, env):
        self.env = deepcopy(env)
        self.links = env.get_links()

    def __str__(self):
        return "JoinTheShortestQueuePolicy"
    def act(self, state: dict, device=None):
        return self.forward(state)
    def forward(self, state: dict, old_state=None):
        d_state = self.env.unflatten_obs(state.reshape(-1,1)) # state as dictionary

        # Convert Queues to integer based keys
        Q = keys_to_ints(d_state['Q'])
        # Get all queue sizes except the source Q[1] and the sink Q[Q.__len__()]
        Qs = np.array([Q[i][1] for i in range(2,Q.__len__())])
        min_Q = np.argmin([Qs]) + 2
        # Create action as dictionary
        f = self.env.get_action_format()
        for link in f.keys():
            if link[0] == 1:
                if link[1] == min_Q:
                    f[link][1] = min(1, int(Q[link[0]][1]))
            else:
                f[link][1] = min(1, int(Q[link[0]][1]))

        flat_f = self.env.flatten_action(f)
        return flat_f[0,:]
# class ShortestPath:
#
#     def __init__(self, env):
#         self.env = deepcopy(env)
#         self.links = env.get_links()
#         self.action_format = env.get_action_format()
#         self.shortest_path = self.get_shortest_path()
#         self.shortest_path_action = self.get_shortest_path_action()
#         self.shortest_path_action = self.env.flatten_action(self.shortest_path_action)
#
#     def get_shortest_path(self):
#         # Shortest path
#         shortest_path = nx.shortest_path(self.links, source=1, target=16, weight='weight')
#         shortest_path = list(shortest_path)
#         return shortest_path
#
#     def get_shortest_path_action(self):
#         shortest_path_action = []
#         for i in range(len(self.shortest_path)-1):
#             shortest_path_action.append(self.action_format[self.shortest_path[i], self.shortest_path[i+1]])
#         return shortest_path_action
#
#     def forward(self, state: dict, old_state=None):
#         return self._forward()
#
#     def act(self, state: dict, old_state=None):
#         return self._forward()
#
#     def _forward(self):
#         return self.shortest_path_action
#     self.env = deepcopy(env)
#     self.links = env.get_links()
#     self.action_format = env.get_action_format()
