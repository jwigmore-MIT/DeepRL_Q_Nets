# import
import gymnasium as gym
from copy import deepcopy
import numpy as np



def create_rv(rng, nums, probs):
    if isinstance(nums, int):
        return bern_rv(rng, num=nums, prob=probs)
    elif isinstance(nums, list):
        return categorical_rv(rng, nums=nums, probs=probs)

class bern_rv:

    def __init__(self, rng, num = 1, prob = 0.5):
        self.rng = rng
        self.num = num
        self.prob = prob

    def sample(self):
        if self.prob == 1:
            return self.num
        else:
            return int(self.rng.choice([0, self.num], 1, p=[1 - self.prob, self.prob]))

    def mean(self):
        return self.num * self.prob

    def max(self):
        return self.num

class categorical_rv:

    def __init__(self, rng, nums = [0,1], probs = None):
        self.rng = rng
        self.nums = nums
        self.probs = probs

    def sample(self):
        if self.probs is None:
            return self.nums
        else:
            return int(self.rng.choice(self.nums, 1, p=self.probs))

    def mean(self):
        return np.dot(self.nums, self.probs)

    def max(self):
        return np.max(self.nums)

class ServerAllocation(gym.Env):

    def __init__(self, net_para):
        super(ServerAllocation, self).__init__()
        # Seeding
        self.rng, self.rng_seed = gym.utils.seeding.np_random(net_para["seed"])
        #self.rng = np.random.default_rng()
        # Nodes/Buffers/Queues
        self.nodes = eval(net_para['nodes'])
        self.destination = max(self.nodes)
        self.n_queues = len(self.nodes) - 1
        self.buffers = {node: 0 for node in self.nodes if node != self.destination}
        # Links
        self.obs_links = net_para["obs_links"] # whether or not the link states are observable
        self.links = [tuple(link) for link in eval(net_para['links'])]  # observable links <list> of <tup>
        self.capacities_fcn = self._extract_capacities(net_para['capacities'])
        self.Cap = self._sim_capacities()
        # Arrivals/Classes
        self.classes, self.destinations = self._extract_classes(net_para['classes'])
        # Spaces
        state_low = np.concatenate((np.zeros(self.n_queues), np.zeros(self.n_queues)))
        state_high = np.concatenate((1e3 * np.ones(self.n_queues), np.array([cap.max() for cap in self.capacities_fcn])))
        self.state_space = gym.spaces.Box(low=state_low, high=state_high, shape=(2 * self.n_queues,), dtype=np.float32)
        if self.obs_links:
            self.observation_space = self.state_space
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1e3, shape=(self.n_queues,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.n_queues+1)
        # Fixed Policies
        if self.obs_links:
            self.fixed_policies = ["LCQ", "RCQ", "LRCQ"] # "Longest Connected Queue", "Random Connected Queue", "Least Reliable Connected Queue"
        else:
            self.fixed_policies = ["LQ", "RQ", "MRQ"] # "Longest Queue", "Random Queue", "Most Reliable Queue"


    def step(self, action, debug = False):
        # the action is an integer indicating the server to send the arrived packet too
        info = {}
        action = int(action)
        # Step 0: Convert action to the queue number and check if it is valid
        if debug: init_buffer = deepcopy(self.buffers)
        current_capacities = deepcopy(self.Cap)

        if action > self.n_queues:
            raise ValueError(f"Invalid action {action} for {self.n_queues} queues")

        # Step 1: Apply the action
        if action > 0:

            if self.get_obs().sum() > 0: #only apply an action if the queue is not empty
                ignore_action = False
                delivered = min(1 * self.Cap[action-1], self.buffers[action])
                self.buffers[action] -= delivered

            else:
                ignore_action = True
                delivered = 0
        else:
            ignore_action = False
            delivered = 0
        if debug: post_action_buffer = deepcopy(self.buffers)

        if (self.get_obs() < 0).any():
            raise ValueError(f"Negative queue size, something is wrong")

        # Step 2: Get the corresponding reward
        #reward = self._get_reward()

        # Step 3: Simulate New Arrivals
        n_arrivals = self._sim_arrivals()

        # Step 4: Simulate new capacities
        self._sim_capacities()


        if debug: post_arrival_buffer = deepcopy(self.buffers)
        new_capacities = deepcopy(self.Cap)

        # Step 5: Get the new state
        next_state = self.get_obs()
        backlog = self.get_backlog()

        # Step 2: Get the corresponding reward
        reward = self._get_reward()

        # Step 6: Fill out the info dict
        info = {"ignore_action": ignore_action, "current_capacities": current_capacities,"new_capacities": new_capacities, "delivered": delivered,
                "backlog": backlog, "n_arrivals": n_arrivals, "env_reward": reward}
        terminated = False
        truncated = False

        if debug: self._debug_printing(init_buffer, current_capacities, delivered,
                           ignore_action, action, post_action_buffer,
                           post_arrival_buffer, n_arrivals, reward, new_capacities)

        return next_state, reward, terminated, truncated, info

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        #np.random.seed(seed)
        self.buffers = {node: 0 for node in self.nodes if node != self.destination}
        obs = self.get_obs()
        self._sim_capacities()
        return obs, {}

    def _debug_printing(self, init_buffer, current_capacities, delivered,
                           ignore_action, action, post_action_buffer,
                           post_arrival_buffer, n_arrivals, reward, new_capacities):
        print("="*20)
        print(f"Initial Buffer: {init_buffer}")
        print(f"Current Capacities: {current_capacities}")
        print(f"Ignore Action: {ignore_action}")
        print(f"Action: {action}")
        print(f"Delivered: {delivered}")
        print(f"Post Action Buffer: {post_action_buffer}")
        print(f"Reward: {reward}")
        print("Arrivals: ", n_arrivals)
        print(f"Post Arrival Buffer: {post_arrival_buffer}")
        print(f"New Capacities: {new_capacities}")
        print("="*20)
        print("\n")

    def _sim_capacities(self):
        self.Cap = [rv.sample() for rv in self.capacities_fcn]
        # return copy of the second half of Cap.keys as np.array
        return np.array(self.Cap)


    def _get_reward(self, type = "congestion"):

        if type == "congestion":
            return -self.get_backlog()
        else:
            raise NotImplementedError

    def get_backlog(self):
        return np.sum(self.get_buffers())

    def get_buffers(self):
        return np.array(list(self.buffers.values()))

    def get_obs(self):
        if not self.obs_links:
            return np.array(self.get_buffers())
        else:
            return np.concatenate([self.get_buffers(), self.get_cap()])

    def get_cap(self):
        return np.array(self.Cap)

    def get_stable_action(self, type = "LQ"):
        q_obs = self.get_buffers()
        if type == "LQ":
            action = np.random.choice(np.where(q_obs == q_obs.max())[0]) + 1  # LQ
        elif type == "SQ":
            q_obs[q_obs == 0] = 1000000
            action = np.random.choice(np.where(q_obs == q_obs.min())[0]) + 1
        elif type == "RQ":
            action = np.random.choice(np.where(q_obs > 0)[0]) + 1
        elif type == "LCQ":
            cap = self.get_cap()
            connected_obs = cap * q_obs
            action = np.random.choice(np.where(connected_obs == connected_obs.max())[0]) + 1
        elif type == "MWQ": #Max Weighted Queue
            p_cap = self.service_rate
            valid_obs = q_obs > 0
            # choose the most reliable queue that is not empty
            valid_p_cap = p_cap * valid_obs**2
            if np.all(valid_obs == False):
                action = 0
            else:
                action = np.random.choice(np.where(valid_p_cap == valid_p_cap.max())[0]) + 1
        elif type == "MWCQ": #Max Weighted Connected Queue
            cap = self.get_cap()
            weighted_obs = cap * q_obs**2
            if np.all(weighted_obs == 0):
                action = 0
            else:
                action = np.random.choice(np.where(weighted_obs == weighted_obs.max())[0]) + 1

        elif type == "RCQ": # Random Connected Queue
            cap = self.get_cap()
            connected_obs = cap * q_obs
            if np.all(connected_obs == 0):
                action = 0
            else:
                action = np.random.choice(np.where(connected_obs > 0)[0]) + 1
        elif type == "LRCQ": # Least Reliable Connected Queue
            p_cap = self.unreliabilities
            cap = self.get_cap()
            connected_obs = cap * q_obs
            weighted_obs = p_cap * connected_obs
            if np.all(weighted_obs == 0):
                action = 0
            else:
                action = np.random.choice(np.where(weighted_obs == weighted_obs.max())[0]) + 1
        elif type == "Optimal":
            if not self.obs_links:
                p_cap = 1 - self.unreliabilities
                non_empty = q_obs > 0
                action  = np.argmax(p_cap*non_empty)+1
            else:
                # should select the connected link with the lowest success probability
                p_cap = 1 - self.unreliabilities
                non_empty = q_obs > 0
                action = np.argmax(p_cap * non_empty) + 1


        if not isinstance(action, int):
            action = action.astype(int)
        return action

    def get_serviceable_buffers(self):
        if self.obs_links:
            temp = self.get_buffers() * self.get_cap()
            return temp
    def get_mask(self, state = None):
        """
        Cases:
        Based on buffers being empty
            1) All buffers are empty -> mask all but action 0
            2) There is at least one non-empty buffer -> mask action 0 and all empty buffers
        Based on links being connected AND observed
        1.) Mask the actions corresponding to non-connected links
            - If already masked, leave masked, if not masked but not connected mask
                mask[1:] = np.logical_or(mask[1:], 1-self.get_cap())
        Returns: Boolean mask vector corresponding to actions 0 to n_queues
        -------

        """
        # returns a vector of length n_queues + 1
        mask = np.bool_(np.zeros(self.n_queues+1)) # size of the action space
        # masking based on buffers being empty
        if self.get_backlog() == 0:
            mask[1:] = True
        else: # Case 2
            mask[0] = True # mask action 0
            mask[1:] = self.get_buffers() < 1 # mask all empty buffers
        # masking based on connected links
        if self.obs_links:
            mask[1:] = np.logical_not(self.get_serviceable_buffers())
            if np.all(mask[1:]==True):
                mask[0] = False
        if np.all(mask):
            raise ValueError("Mask should not be all True")
        elif np.all(mask == False):
            raise ValueError("Mask should not be all False")
        return mask


    def _sim_arrivals(self):
        n_arrivals = 0
        for cls_num, cls_rv in enumerate(self.classes.items()):
            source = cls_num+1
            if cls_rv.sample() == 1:
                self.buffers[source] += 1
                n_arrivals += 1
        return n_arrivals

    def _extract_capacities(self, cap_dict):
        caps = {}
        if '(0,0)' in cap_dict.keys():
            # All links have the same capacity and probability
            capacity = cap_dict['(0,0)']['capacity']
            probability = cap_dict['(0,0)']['probability']
            for link in self.links:
                caps[link] = create_rv(self.rng, num=capacity, prob = probability)

        for link, l_info in cap_dict.items():
            if isinstance(link, str):
                link = eval(link)
            if link == (0,0):
                continue
            capacity = eval(l_info['capacity']) if isinstance(l_info['capacity'], str) else l_info['capacity']
            probability = eval(l_info['probability']) if isinstance(l_info['probability'], str) else l_info['probability']

            rv = create_rv(self.rng, nums = capacity, probs = probability)
            caps[link] = rv

            # generate unreliabilities
        service_rate = []
        for link in self.links:
            if link[1] == self.destination:
                service_rate.append(caps[link].mean())
        self.service_rate = np.array(service_rate)


        if (0,0) in caps.keys():
            del caps[(0,0)]

        # Convert caps to a list
        caps = np.array(list(caps.values()))

        return caps

    def _extract_classes(self, class_dict):
        classes = []
        destinations = []
        for cls_num, cls_info in class_dict.items():
            rv = bern_rv(self.rng, num = cls_info['arrival'], prob = cls_info['probability'])
            classes.append(rv)
        if len(classes) != len(list(self.links)):
            raise ValueError("Number of classes must equal number of links")

        return classes, destinations

    def _sim_arrivals(self):
        arrivals = np.zeros(len(self.classes))
        for cls_num, cls_rv in enumerate(self.classes):
            source = cls_num+1
            if cls_rv.sample() == 1:
                self.buffers[source] += 1
                arrivals[source-1] += 1
        return arrivals

    def set_state(self, state):
        if self.obs_links:
            for i in range(self.n_queues):
                self.buffers[i+1] = state[i]
            for j in range(self.n_queues):
                self.Cap[j] = state[j+self.n_queues]

        return self.get_obs()

    def estimate_transitions(self, state, action, max_samples = 1000, min_samples = 100, theta = 0.001):

        C_sas = {} # counter for transitions (s,a,s')
        P_sas = {} # probabilities for transitions (s,a,s')
        diff = {}
        for n in range(1, max_samples+1):
            self.set_state(state)
            next_state, _, _, _, _ = self.step(action)
            if tuple(next_state) in C_sas.keys():
                C_sas[tuple(next_state)] += 1# increment the counter for the visits to next state
                p_sas = C_sas[tuple(next_state)]/n # calculate the probability of the transition
                diff[tuple(next_state)] = np.abs(P_sas[tuple(next_state)] - p_sas)
                P_sas[tuple(next_state)] = p_sas
            else:
                C_sas[tuple(next_state)] = 1
                P_sas[tuple(next_state)] = 1

            # check for convergence:
            if n > min_samples and np.all(list(diff.values())) < theta:
                break
        P_sas = {key: value/n for key, value in C_sas.items()}
        if np.abs(np.sum(list(P_sas.values())) - 1) > 0.001:
            raise ValueError("Transition Probabilities do not sum to one")

        # convert to probabilities

        return P_sas, n

from DP.mdp import MDP
from DP.value_iteration import ValueIteration
from DP.tabular_value_function import TabularValueFunction
import pickle


class ServerAllocationMDP(MDP):

    def __init__(self, env, name = "", q_max = 10, discount = 0.99):
        self.actions = list(np.arange(env.action_space.n))
        self.n_queues = env.n_queues
        self.state_list = self.get_state_list(env, q_max = q_max)
        self.tx_matrix = None
        self.q_max = q_max
        self.discount = discount
        self.name = f"{name}_qmax{q_max}_discount{discount}"
        #self.env = deepcopy(env)

    class TX_Matrix:

        def __init__(self, tx_matrix, n_tx_samples, num_s_a_pairs):
            self.tx_matrix = tx_matrix
            self.n_tx_samples = n_tx_samples
            self.num_s_a_pairs = num_s_a_pairs

        def __call__(self, *args, **kwargs):
            return self.tx_matrix


    def get_state_list(self, env, q_max):
        def create_state_map(low, high):
            state_elements = []
            for l, h in zip(low, high):
                state_elements.append(list(np.arange(l, h + 1)))
            state_combos = np.array(np.meshgrid(*state_elements)).T.reshape(-1, len(state_elements))
            # convert all arrays in state_combos to lists
            state_combos = [list(state) for state in state_combos]
            return state_combos

        high_queues = np.ones(env.n_queues) * q_max
        high_links = np.array([rv.max() for rv in env.capacities_fcn])
        high = np.concatenate((high_queues, high_links))
        low = np.zeros_like(high)
        state_list = create_state_map(low, high)
        return state_list
    def estimate_tx_matrix(self, env, max_samples = 1000, min_samples = 100, theta = 0.001, save = True):

        tx_matrix, n_tx_samples = form_transition_matrix(env, self.state_list, self.actions, max_samples, min_samples, theta)
        num_s_a_pairs = np.sum([len(tx_matrix[key]) for key in tx_matrix.keys()])
        num_samples = np.array(list(n_tx_samples.values()))
        print("Transition Matrix Estimated")
        print("Mean number of samples per state-action pair: ", np.mean(num_samples))
        print("Number of state-action pairs: ", np.sum(num_s_a_pairs))
        self.tx_matrix = self.TX_Matrix(tx_matrix, n_tx_samples, num_s_a_pairs)

        if save:
            pickle.dump(self.tx_matrix, open(f"saved_mdps/{self.name}_max_samples-{max_samples}_tx_matrix.pkl", "wb"))

        return self.tx_matrix

    def load_tx_matrix(self, path):
        self.tx_matrix = pickle.load(open(path, "rb"))
        #self.n_tx_samples = {key: len(self.tx_matrix[key]) for key in self.tx_matrix.keys()}
        return self.tx_matrix,

    def get_states(self):
        # return all possible states from 0 to s_max for each server
        return [list(state)[:-1] for state in self.tx_matrix().keys()]

    def get_transitions(self, state, action):
        key = deepcopy(state)
        key.append(action)
        tx_dict = self.tx_matrix()[tuple(key)]
        transitions = list(zip(list(tx_dict.keys()), list(tx_dict.values())))

        return transitions

    def get_reward(self, state, action, next_state):
        next_buffers = next_state[:self.n_queues]
        if action >0:
            if state[action-1] == 0: # if the action for the chosen queue is empty
                return -1000
            elif state[self.n_queues+action - 1] == 0: # if the link for the chosen action has zero capacity
                return -1000
        if np.any(np.array(next_buffers) >= self.q_max):
            return -100
        else:
            return -np.sum(next_buffers)

    def get_actions(self, state):
        buffers = np.array(state[:self.n_queues])
        servers = np.array(state[self.n_queues:])
        if np.all(buffers == 0):
            return [0]
        else:
            buf_serv = buffers*servers
            if np.all(buf_serv == 0):
                return [0]
            else:
                return np.where(buf_serv > 0)[0] + 1

    def get_initial_state(self):
        return np.zeros(self.n_queues*2)

    def is_terminal(self, state):
        return False

    def get_discount_factor(self):
        return self.discount

    def get_goal_states(self):
        return None

    def do_VI(self, max_iterations = 100, theta = 0.1):
        if self.tx_matrix is None:
            raise ValueError("Transition Matrix must be estimated before running VI")
        value_table = TabularValueFunction()
        ValueIteration(self, value_table).value_iteration(max_iterations, theta)
        policy = value_table.extract_policy(self)
        policy_table = dict(policy.policy_table)
        value_table = dict(value_table.value_table)
        self.vi_policy = policy_table
        self.value_table = value_table


    def get_VI_policy(self):
        if self.value_table is None:
            raise ValueError("Value Table must be estimated before getting policy")
        policy = self.value_table.extract_policy(self)
        policy_table = dict(policy.policy_table)
        self.vi_policy = policy_table

    def save_MDP(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    def use_policy(self, state):
        if self.vi_policy is None:
            raise ValueError("Policy must be estimated before using it")
        return self.vi_policy[tuple(state)]

def form_transition_matrix(env, state_list, action_list, max_samples = 1000, min_samples = 100, theta = 0.001):
    from tqdm import tqdm
    """
    state_list: list of states as lists
    action_list: list of integers
    """
    tx_matrix = {} # keys will be tuples of the form (state, action)
    n_samples = {}
    pbar = tqdm(total = len(state_list)**len(action_list), desc = "Estimating Transition Matrix")
    n=0
    for state in state_list:
        # get valid actions
        env.set_state(state)
        mask = env.get_mask()
        action_list = np.where(mask == False)[0]
        for action in action_list:
            n+=1
            pbar.update(n)
            # create a tuple of the form (state, action)
            key = deepcopy(state)
            key.extend([action])
            tx_matrix[tuple(key)], n_samples[tuple(key)] = env.estimate_transitions(state, action, max_samples, min_samples, theta)

    return tx_matrix, n_samples



def get_state_list(env, q_max):
    state_list = []
    for i in range(env.n_queues):
        state_list.append([0])
        state_list.append([1])
    return state_list