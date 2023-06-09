
import pulp
from copy import deepcopy
import numpy as np


from Environments.MCMH_tools import *
from torch import nn


class MCMHBackPressurePolicy(nn.Module):

    # initialization
    def __init__(self, env, M = False, d_scale = None):
        super(MCMHBackPressurePolicy, self).__init__()
        # check if env has attribute unwrapped
        if hasattr(env, 'unwrapped'):
            self.env = env.unwrapped
        else:
            self.env = deepcopy(env)
        self.links = env.get_links()
        self.action_format = env.get_action_format()
        if M not in [False, "R", "SP"]: #R: reachability, SP: shortest path
            raise ValueError("M must be False, 'R', or 'SP'")
        else:
            self.modified = M
        if self.modified == "SP" and d_scale is None:
            raise ValueError("Must provide d_scale when using shortest path")
        else:
            self.d_scale = d_scale
        self.max_action = env.max_actions
        # self.map_to_discrete = map_to_discrete
        self.mu = {} # transmission variable for each link (i.e. total flow)
        for link in self.links:
            self.mu[link] = pulp.LpVariable('mu' + '_' + str(link[0]) + '_' + str(link[1]), lowBound=0, cat='Integer')
        # initialize the solver
        self.solver = pulp.PULP_CBC_CMD(msg=False, warmStart=False)
        self.dist = env.get_distances()
        self.destinations = env.get_destinations()
        self.solution_table = {}
        self.resolve = True




    def forward(self, state : dict, old_state = None):
        return self._forward(state, old_state)

    def act(self, state: dict, old_state=None):
        action  = self._forward(state, old_state)
        return action

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        return [self._forward(observation)], state


    def _forward(self, state : dict, old_state = None):
        #state = keys_to_ints(batch_state)
        state = state
        t_state = tuple(state)
        if t_state not in list(self.solution_table.keys()) or self.resolve:
            f = deepcopy(self.action_format) # flow for each class on each link
            # formulate the problem
            problem, opt_ij = self.Formulate(state)
            # solve the problem
            problem.solve(self.solver)

            # get the result for service
            for link in self.links:
                val = pulp.value(self.mu[link])
                f[link][opt_ij[link][0]] = int(val) if val != None else 0
            # if self.map_to_discrete:
            #     f = self.get_action_key(f)
            # else: # need to convert keys to strings for batching
            f = self.env.flatten_action(f)
            self.solution_table[t_state] = f[0,:]
            return f[0,:]
        else:
            return self.solution_table[t_state]

    def get_action_key(self, action):
        keys = [k for k, v in self.env.action_map.items() if v == action]
        if keys:
            return np.array([keys[0]])
        return None

    def str_to_tup_str(self, tup_string):
        "Converts: '(x,y)' -> ('x','y') "
        return tuple(str(int(x)) for x in tup_string[1:-1].split(','))

    def tup_str_to_tup_int(self, tup):
        "converts: ('x,'y') -> (x,y)"
        return tuple(int(x) for x in tup)

    def str_to_tup_int(self, string):
        "converts: '(x,y)' -> (x,y)"
        return tuple(int(x) for x in string[1:-1].split(','))
    # initialize the problem
    def Formulate(self, state):
        # check if state is tianshou's Batch({obs: , info:})
        # if 'obs' in state:
        #     # check if flattened
        #     if hasattr(state.obs, 'ndim') and state.obs.ndim == 3:
        #         state = self.env.unflatten_obs(state.obs[0,0,:])
        if isinstance(state, np.ndarray):
            state = self.env.unflatten_obs(state.reshape(-1,1))

        Q = keys_to_ints(state['Q'])
        #Cap = keys_to_ints(state['Cap'])
        Cap = self.env.Cap

        # determine the optimal class for each link
        opt_ij = {}  # (i,j): (number of optimal class for (i,j) :  weight (W_ij))
        Q_c = deepcopy(Q) #Want to work with a modifiable copy so we account for decisions that will be made prior
        for link, cap in Cap.items():
            if link == (0,0):
                continue
            #link = self.str_to_tup_str(link)
            #Q_i = Q[str(link[0])]  # dict
            #Q_j = Q[str(link[1])]  # dict
            Q_i = Q_c[link[0]]
            Q_j = Q_c[link[1]]
            diff = {}
            for cls, amt in Q_i.items():
                if self.modified != "SP":
                    diff[cls] = Q_i[cls] - Q_j[cls]
                else:
                    diff[cls] = (Q_i[cls] + self.d_scale * self.dist[link[0], self.destinations[cls]]) - (
                                Q_j[cls] + self.d_scale * self.dist[link[1], self.destinations[cls]])
                if self.modified == "R" and self.max_action[link][cls] == 0:
                    diff[cls] = -np.inf
            opt_cls = max(diff, key=diff.get)  # optimal class
            #opt_ij[self.tup_str_to_tup_int(link)] = (opt_cls, max(diff[opt_cls], 0))
            opt_ij[link] = (opt_cls, max(diff[opt_cls], 0))

        # formulate the problem
        problem = pulp.LpProblem("MW", pulp.LpMaximize)

        problem += pulp.lpSum([self.mu[link] * tup[1] for link, tup in opt_ij.items()])
        for link,cap in Cap.items():
            #problem += self.mu[self.str_to_tup_int(link)] <= cap
            problem += self.mu[link] <= cap
        return problem, opt_ij

    def learn(self, input_batch):
        return None



class MCMHBackPressurePolicySP(nn.Module):

    # initialization
    def __init__(self, env, d_scale = 1):
        super().__init__()
        # check if env has attribute unwrapped
        if hasattr(env, 'unwrapped'):
            self.env = env.unwrapped
        else:
            self.env = deepcopy(env)
        self.d_scale = d_scale
        self.links = env.get_links()
        self.distances = env.get_distances()
        self.destinations = env.get_destinations()
        self.action_format = env.get_action_format()
        self.max_action = env.max_actions

        self.mu = {} # transmission variable for each link (i.e. total flow)
        for link in self.links:
            self.mu[link] = pulp.LpVariable('mu' + '_' + str(link[0]) + '_' + str(link[1]), lowBound=0, cat='Integer')
        # initialize the solver
        self.solver = pulp.PULP_CBC_CMD(msg=False)



    def forward(self, state : dict, old_state = None):
        return self._forward(state, old_state)

    def act(self, state: dict, old_state=None):
        action  = self._forward(state, old_state)
        return action

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        return [self._forward(observation)], state


    def _forward(self, state : dict, old_state = None):
        #state = keys_to_ints(batch_state)
        state = state
        f = deepcopy(self.action_format) # flow for each class on each link
        # formulate the problem
        problem, opt_ij = self.Formulate(state)
        # solve the problem
        problem.solve(self.solver)
        # get the result for service
        for link in self.links:
            val = pulp.value(self.mu[link])
            f[link][opt_ij[link][0]] = int(val) if val != None else 0
        if self.map_to_discrete:
            f = self.get_action_key(f)
        else: # need to convert keys to strings for batching
            f = self.env.flatten_action(f)
        result = {'act': f,
                  'state': None}

        return f[0,:]

    def get_action_key(self, action):
        keys = [k for k, v in self.env.action_map.items() if v == action]
        if keys:
            return np.array([keys[0]])
        return None

    def str_to_tup_str(self, tup_string):
        "Converts: '(x,y)' -> ('x','y') "
        return tuple(str(int(x)) for x in tup_string[1:-1].split(','))

    def tup_str_to_tup_int(self, tup):
        "converts: ('x,'y') -> (x,y)"
        return tuple(int(x) for x in tup)

    def str_to_tup_int(self, string):
        "converts: '(x,y)' -> (x,y)"
        return tuple(int(x) for x in string[1:-1].split(','))
    # initialize the problem
    def Formulate(self, state):
        # check if state is tianshou's Batch({obs: , info:})
        # if 'obs' in state:
        #     # check if flattened
        #     if hasattr(state.obs, 'ndim') and state.obs.ndim == 3:
        #         state = self.env.unflatten_obs(state.obs[0,0,:])
        if isinstance(state, np.ndarray):
            state = self.env.unflatten_obs(state.reshape(-1,1))

        Q = keys_to_ints(state['Q'])
        #Cap = keys_to_ints(state['Cap'])
        Cap = self.env.Cap

        # determine the optimal class for each link
        opt_ij = {}  # (i,j): (number of optimal class for (i,j) :  weight (W_ij))
        Q_c = deepcopy(Q) #Want to work with a modifiable copy so we account for decisions that will be made prior
        for link, cap in Cap.items():
            #link = self.str_to_tup_str(link)
            #Q_i = Q[str(link[0])]  # dict
            #Q_j = Q[str(link[1])]  # dict
            Q_i = Q_c[link[0]]
            Q_j = Q_c[link[1]]

            diff = {}
            for cls, amt in Q_i.items():
                diff[cls] = (Q_i[cls] + self.d_scale*self.self.dist[link[0], self.destinations[cls]]) - (Q_j[cls]+ self.d_scale* self.dist[link[1], self.destinations[cls]])
                if self.modified and self.max_action[link][cls] == 0:
                    diff[cls] = -np.inf
            opt_cls = max(diff, key=diff.get)  # optimal class
            #opt_ij[self.tup_str_to_tup_int(link)] = (opt_cls, max(diff[opt_cls], 0))
            opt_ij[link] = (opt_cls, max(diff[opt_cls], 0))

        # formulate the problem
        problem = pulp.LpProblem("MW", pulp.LpMaximize)

        problem += pulp.lpSum([self.mu[link] * tup[1] for link, tup in opt_ij.items()])
        for link,cap in Cap.items():
            #problem += self.mu[self.str_to_tup_int(link)] <= cap
            problem += self.mu[link] <= cap
        return problem, opt_ij

    def learn(self, input_batch):
        return None

    def precompute_weights(self):
        # for each node and class, compute W_i^k i.e. the number of hops from node i to the destination of class k
        W = {}
        for node in self.env.nodes:
            continue
        pass

