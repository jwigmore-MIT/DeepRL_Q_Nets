# import
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
from copy import deepcopy
from Environments.MCMH_tools import *
import numpy as np
import itertools
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union
from gymnasium.wrappers import TimeLimit





class bern_rv:

    def __init__(self, num = 1, prob = 0.5):
        self.num = num
        self.prob = prob

    def sample(self):
        if self.prob == 1:
            return self.num
        else:
            return int(np.random.choice([0, self.num], 1, p=[1 - self.prob, self.prob]))



class MultiClassMultiHop(gym.Env):
    '''
    MAJOR CHANGE: Internal states and actions can use nested dictionaries, but all external will use
    flattened versions
    This all inputs/returns for external facing functions return flattened arrays
    '''
    def __init__(self, net_para = None, config = None,  space_para: dict = None, **kwargs):
        super(MultiClassMultiHop, self).__init__()

        # initialize from config
        if config is not None:
            attrs = [a for a in dir(config.env) if not a.startswith('__')]
            net_para = {attr: getattr(config.env, attr) for attr in attrs}

        self.name = net_para.get('name',None)
        self.t = 0

        # topology information
        self.nodes = eval(net_para['nodes'])  # nodes <list>
        self.links = [tuple(link) for link in eval(net_para['links'])] # observable links <list> of <tup>
        self.graph = self._make_graph()
        self.reachable_nodes = self.create_reachable()
        self.capacities_fcn = self._extract_capacities(net_para['capacities'])


        # classes
        self.classes = self._extract_classes(net_para['classes'])
        #net_para['classes']  # [(k,s_k, d_k, [a_k, p_k])]
        self.K = len(self.classes)

        # action format
        self.action_format = {link: {c: 0 for c in self.classes.keys()} for link in self.links}

        # initialize neighbors
        self.neighbors = defaultdict(list)  # pairs of neighbors i.e. possible next nodes for a packet
        for start, end in self.links:
            self.neighbors[start].append(end)


        if space_para:
            q_lim = space_para['q_lim']  # limit on buffer sizes
        else:
            q_lim = 10_000

        # states
        self.Arr = {node: {c: 0 for c in self.classes.keys()} for node in
                    self.nodes}  # Counter for all arrivals: {(s_k, d_k): a_k(t)}
        self.Q = {node: {c: 0 for c in self.classes.keys()} for node in
                  self.nodes}  # Buffers for all nodes Q_i^k = Q[i][k]
        self.Cap = None
        self._sim_capacities()

        # actions
        self.f = {link: {c: 0 for c in self.classes.keys()} for link in
                  self.links}  # captures the action (flows) for each node

        # sum of queues in the network
        self.backlog = 0
        self.delivered = 0

        # gym variables
        self.N = len(self.nodes)
        self.M = len(self.links)

        # initialize observation space
        Q_space = spaces.Dict(
            {n: spaces.Dict({cls: spaces.Box(low=0, high=q_lim, dtype=float) for cls in self.classes.keys()}) for n in
             self.nodes})
        Cap_space = spaces.Dict(
            {link: spaces.Box(low=0, high=bern_rv.num, dtype=float) for link, bern_rv in self.capacities_fcn.items()})


        self.observation_space = spaces.Dict({
            'Q': Q_space,
            #'Cap': Cap_space,
        })
        flat_obs_space = spaces.utils.flatten_space(self.observation_space)
        self.obs_space_size = flat_obs_space.shape[0]  # the size of the state space

        self.max_actions = self.limit_actions()

        self.action_space = spaces.Dict(
            {link: spaces.Dict({cls: spaces.Box(low=0, high=self.max_actions[link][cls], dtype=int) for cls in self.classes.keys()}) for
             link, class_dict in self.max_actions.items()})
        self.action_space_size = spaces.utils.flatten_space(self.action_space).shape[0]
        # Trick to get the number of states minus
        self.flat_qspace_size = len(self.get_flat_arrival_keys())
        if config is not None:
            setattr(config.env, "flat_state_dim", self.obs_space_size)
            setattr(config.env, "flat_action_dim", self.action_space_size)
        self._seed = config.seed
        self.reset(seed=self._seed)

    def _make_graph(self):
        graph = {}
        for node in self.nodes:
            graph[node] = []

        for link in self.links:
            graph[link[0]].append(link[1])

        return graph

    def compute_reachable_nodes(self, start_node):
        def traverse(node, visited: set):
            visited.add(node)
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    traverse(neighbor, visited)
        visited = set()
        traverse(start_node, visited)
        return visited

    def create_reachable(self):
        reachable_nodes = {}
        for node in self.nodes:
            reachable_nodes[node] = self.compute_reachable_nodes(node)
        return reachable_nodes


    """
    Actions correspond to flows on links
    In order to determine if an action f(i,j,k) is valid, we need to see if the destination node of class k is reachable from node j
    """

    def limit_actions(self):
        """
        Input: action_format
               reachable_nodes_dict
        """
        max_action = {}
        for link, class_dict in self.action_format.items():
            j = link[1]
            j_reach = self.reachable_nodes[j]
            max_action[link] = {}
            for cls, value in class_dict.items():
                cls_dest = self.classes[cls][1]
                if cls_dest not in j_reach:
                    max_action[link][cls] = 0
                else:
                    max_action[link][cls] = self.capacities_fcn[link].num
        return max_action





    def step(self, action):
        return self._step(action)
    def _step(self, action: Dict):
        info = {'action':self.flatten_action(action)}
        obs = self.get_state()

        # copy action dict, add external flows
        flows = action.copy()
        # apply flow (action + exits), update state
        self._serve(flows)
        info['flows'] = self.flatten_action(deepcopy(self.f))
        info["delivered"] = deepcopy(self.delivered)
        backlog = self.backlog = info["backlog"] =  self._get_backlog()
        reward = - backlog

        # get and record arrivals and next capacities
        self._sim_arrivals()
        info['arrivals'] = self.flatten_arrivals(deepcopy(self.Arr))
        info['queues'] = self.get_f_state()
        self._sim_capacities()
        #info['Arr'] = keys_to_strings(deepcopy(self.Arr))

        self.t +=1
        # required for gym

        terminated = False
        truncated = False

        new_obs = self.get_state()



        return new_obs, reward, terminated, truncated, info

    def reset(self, seed = None, options = None):

        # reset states, actions, flows
        self.Arr = {node: {c: 0 for c in self.classes.keys()} for node in
                    self.nodes}  # Counter for all arrivals: {(s_k, d_k): a_k(t)}
        self.Q = {node: {c: 0 for c in self.classes.keys()} for node in
                  self.nodes}  # Buffers for all nodes Q_i^k = Q[i][k]
        self.f = {link: {c: 0 for c in self.classes.keys()} for link in self.links}

        self._sim_capacities()

        self.t = 0

        # required by gym
        obs = self.get_state()
        info = {}
        return obs, info

        #return self.flatten_obs(obs), info

    # For parsing input dict
    def _extract_capacities(self, cap_dict):
        caps = {}
        if '(0,0)' in cap_dict.keys():
            # All links have the same capacity and probability
            capacity = cap_dict['(0,0)']['capacity']
            probability = cap_dict['(0,0)']['probability']
            for link in self.links:
                caps[link] = bern_rv(num=capacity, prob = probability)
        else:
            for link, l_info in cap_dict.items():
                if isinstance(link, str):
                    link = eval(link)
                rv = bern_rv(num = l_info['capacity'], prob= l_info['probability'])
                caps[link] = rv

        return caps

    def _extract_classes(self, class_dict):
        classes = {}
        for cls_num, cls_info in class_dict.items():
            rv = bern_rv(num = cls_info['arrival'], prob = cls_info['probability'])
            classes[int(cls_num)] = [cls_info['source'], cls_info['destination'], rv]

        return classes
    # internal dynamics
    def _sim_capacities(self):
        self.Cap = {key: value.sample() for key, value in self.capacities_fcn.items()}

    def _sim_arrivals(self):
        for cls, tup in self.classes.items():
            src = tup[0]  # source node
            # number of arrivals, prob of arrival
            self.Arr[src][cls] = tup[2].sample()
            self.Q[src][cls] += self.Arr[src][cls]
        # self.logger.record_env_item('arrivals', self.A, self.tt)

    def _serve(self, flows: dict):
        if isinstance(flows, np.int32):
            flows = self.action_map[flows]

        Q_old = deepcopy(self.Q)

        self.delivered = 0
        for link, link_flow in flows.items():
            start_node = link[0]
            end_node = link[1]
            c_ij = self.Cap[link]
            for cls, flow in link_flow.items():
                # Ensure not sending more than start node class queue or link capacity f_ijk <= min(Q_ik, c=C_ij)
                f_max = min(Q_old[start_node][cls],c_ij, self.Q[start_node][cls]) # f_max <= Q_ik(old)

                # Apply bound to f_ijk
                self.f[link][cls] = max(0,min(f_max, flow)) # 0<= f_ijk <= f_max <= Q_ik(old)

                # Subtract f_ijk from start node class k queue
                self.Q[start_node][cls] = max(self.Q[start_node][cls]-self.f[link][cls],0)

                # Add f_ijk to end node class k queue, if its not the destination
                if end_node ==  self.classes[cls][1]:
                    self.delivered += deepcopy(self.Q[end_node][cls])
                    self.Q[end_node][cls] = 0
                else:
                    self.Q[end_node][cls] = self.Q[end_node][cls]+self.f[link][cls]

                c_ij = c_ij- self.f[link][cls]
                if c_ij <= 0:
                    continue


        return

    def _get_backlog(self):
        cost = np.array([0])
        for i, Q_i in self.Q.items():
            for cls, Q_ic in Q_i.items():
                cost += Q_ic #- self.Arr[i][cls] # uncomment if we want to not factor in current arrivals
        # self.logger.record_env_item("cost", cost, self.tt)
        return max(cost,0)

    # Getters
    def get_state(self):
        return {'Q': deepcopy(self.Q), 'Cap': deepcopy(self.Cap)}

    def get_f_state(self):
        state = self.get_state()
        return self.flatten_obs(state)

    def get_action_format(self):
        return deepcopy(self.action_format)

    def get_links(self):
        return deepcopy(self.links)

    # Flattening
    def flatten_obs_space(self):
        return spaces.utils.flatten_space(self.observation_space)

    def flatten_obs(self, obs):
        return spaces.utils.flatten(self.observation_space, obs).reshape((1, -1))

    def flatten_action_space(self):
        return spaces.utils.flatten_space(self.action_space)

    def flatten_action(self, action):
        return spaces.utils.flatten(self.action_space, action).reshape((1,-1))

    def flatten_arrivals(self, arr):
        return spaces.utils.flatten(self.observation_space.spaces['Q'],arr).reshape((1,-1))

    def unflatten_obs(self, flat_obs):
        return spaces.utils.unflatten(self.observation_space, flat_obs)

    def unflatten_action(self, flat_action):
        return spaces.utils.unflatten(self.action_space, flat_action)

    # Generating and Mapping spaces
    def get_flat_obs_keys(self):
        obs_keys = []
        for space_name, space in self.observation_space.spaces.items():
            if space_name == 'Cap':
                for link, space2 in space.items():
                    obs_keys.append(f"C{link}")
            elif space_name == 'Q':
                for node_num, space2 in space.items():
                    for class_num, space3 in space2.items():
                        obs_keys.append(f"Q({node_num},{class_num})")
        return obs_keys

    def get_flat_action_keys(self, mod = "F"):
        action_keys = []
        for link, space in self.action_space.spaces.items():
                for class_num, space2 in space.items():
                    subs = f"({link[0]},{link[1]},{class_num})"
                    action_keys.append(f"{mod}{subs}")

        return action_keys

    def get_flat_arrival_keys(self, mod = "I"):
        arrival_keys = []
        for node_num, arr_dict in self.Arr.items():
                for class_num, arrivals in arr_dict.items():
                    subs = f"({node_num},{class_num})"
                    arrival_keys.append(f"{mod}{subs}")

        return arrival_keys

    # For getting valid actions




# longest connected queue is served for each link
def MCMH_longest_conn_q_policy(env, state):
    """
    NOT IN USE
    """
    action = deepcopy(env.action_format)
    Q = state['Q']
    Cap = state['Cap']
    for link, cap in Cap.items():
        if cap == 0:  # if the capacity is zero,
            action[link] = dict.fromkeys(action[link].keys(), 0)
        else:
            i = link[0]
            j = link[1]
            Q_i = Q[i]
            Q_ic_m = max(Q_i, key = Q_i.get)
            action[link][Q_ic_m] = min([Q_i[Q_ic_m], cap])
            Q[i][Q_ic_m] -= min([action[link][Q_ic_m],Q[i][Q_ic_m]])
    return action


class FlatActionWrapper(gym.ActionWrapper):
    """
    This action wrapper maps flattened actions <nd.array> back to dictionary
    actions of which the Base environment understands
    """
    def __init__(self, MCMH_env):
        super(FlatActionWrapper, self).__init__(MCMH_env)
        self.flattened_action_space = self.flatten_action_space()

    def action(self, action: np.ndarray):
        return self.unflatten_action(action)

class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Action wrapper that maps integers to valid "discrete" actions. For policies that utilize the
    "Discrete Set" action-space representation. This representation was used for all PG and DDQN results
    prior to 3.7.2023
    """
    def __init__(self, env):
        super(DiscreteActionWrapper, self).__init__(env)
        self.generate_action_map()
        # flat_act_space = spaces.utils.flatten_space(self.action_space)
        self.action_space_size = self.action_map.__len__()  # the size of the action space

    def action(self, action):
        action = self.action_map[action]
        return action
    def generate_action_map(self):
        # returns a dictionary, keys = int, values = corresponding action f_ijk
        action_map = {}
        flat_action_map = {}
        n_action = 0
        f_action_space = self.flatten_action_space()
        possible_actions = [list(range(0, (k + 1))) for k in f_action_space.high]
        all_actions = np.array(np.meshgrid(*possible_actions)).T.reshape(-1, len(possible_actions))
        for flat_action in all_actions:
            non_flat = self.unflatten_action(flat_action)
            valid = True
            for link, flows_dict in non_flat.items():
                cap = self.Cap[link]
                if sum(flows_dict.values()) > cap:
                    valid = False
                    break
                else:
                    valid = True
            if valid:
                action_map[n_action] = non_flat
                flat_action_map[n_action] = flat_action
                n_action += 1
        self.action_map = action_map
        self.flat_action_map = flat_action_map
        return action_map, flat_action_map


def init_continuous_env(para, train = True):
    """
    Creates a MCMH environment and applys both the TimeLimit gymnasium wrapper
    and custom FlatAction wrapper. Result it an environment that works with
    Backpressure.py and continuous neural network outputs?
    :param para: parameter dictionary obtained from JSON import
    :return: an MCMH environment for the Backpressure policy
    """

    env = MultiClassMultiHop(para['problem_instance'])
    if train:
        max_episode_steps = para['train_parameters']["episode_length"]
    else:
        max_episode_steps = para['test_parameters']["episode_length"]
    env = TimeLimit(env, max_episode_steps=max_episode_steps )
    env = FlatActionWrapper(env)
    return env

def init_discrete_env(para, train = True):
    env = MultiClassMultiHop(para['problem_instance'])
    if train:
        max_episode_steps = para['train_parameters']["episode_length"]
    else:
        max_episode_steps = para['test_parameters']["episode_length"]
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = DiscreteActionWrapper(env)
    return env

# def init_continuous_env(para, train = True):
#     env = BaseMultiClassMultiHop(para['problem_instance'])
#     if train:
#         max_episode_steps = para['train_parameters']["episode_length"]
#     else:
#         max_episode_steps = para['test_parameters']["episode_length"]
#     env = TimeLimit(env, max_episode_steps=max_episode_steps)



if __name__ == '__main__':
    # Get the environment params file
    pass





