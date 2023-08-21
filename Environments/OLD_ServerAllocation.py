# import
import gymnasium as gym
from copy import deepcopy
import numpy as np

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
class ServerAllocation(gym.Env):

    def __init__(self, net_para):
        super(ServerAllocation, self).__init__()
        # Seeding
        self.rng, self.rng_seed = gym.utils.seeding.np_random(net_para["seed"])
        # Nodes/Buffers/Queues
        self.nodes = eval(net_para['nodes'])
        self.destination = max(self.nodes)
        self.n_queues = len(self.nodes) - 1
        self.buffers = {node: 0 for node in self.nodes if node != self.destination}
        self.observation_space = gym.spaces.Box(low=0, high=1e6, shape=(self.n_queues,), dtype=np.float32)

        # Links and capacity functions
        self.links = [tuple(link) for link in eval(net_para['links'])]
        self.capacities_fcn = self._extract_capacities(net_para['capacities'])
        # Classes (really the arrival processes)
        self.classes, self.destinations = self._extract_classes(net_para['classes'])


        self.action_space = gym.spaces.Discrete(self.n_queues+1)
        self.Cap = self._sim_capacities()
    def step(self, action, debug = False):
        # the action is an integer indicating the server to send the arrived packet too
        info = {}

        # Step 0: Convert action to the queue number and check if it is valid
        if debug: init_buffer = deepcopy(self.buffers)
        current_capacities = deepcopy(self.Cap)

        if action < 0 or action > self.n_queues:
            raise ValueError(f"Invalid action {action} for {self.n_queues} queues")

        # Step 1: Apply the action
        if action > 0:

            if self.get_obs().sum() > 0: #only apply an action if the queue is not empty
                ignore_action = False
                self.buffers[action] -= min(1 * self.Cap[action-1], self.buffers[action])
                delivered = 1 * self.Cap[action-1]
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
        reward = self._get_reward()

        # Step 3: Simulate New Arrivals
        n_arrivals = self._sim_arrivals()

        # Step 4: Simulate new capacities
        self._sim_capacities()


        if debug: post_arrival_buffer = deepcopy(self.buffers)
        new_capacities = deepcopy(self.Cap)

        # Step 5: Get the new state
        next_state = self.get_obs()
        backlog = self.get_backlog()

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
        np.random.seed(seed)
        self.buffers = {node: 0 for node in self.nodes}
        state = self.get_obs()
        self._sim_capacities()
        return state, {}

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
    def _serve_step(self):
        delivered = 0
        for server in self.nodes[1:-1]:
            server_capacity = self.Cap[server,self.destination]
            # if the server has capacity, reduce the buffer by the server capacity
            delivered +=  min(self.buffers[server], server_capacity)
            self.buffers[server] = max(0, self.buffers[server] - server_capacity)
        return delivered

    def _get_reward(self, type = "congestion"):

        if type == "congestion":
            return -np.sum([self.buffers[node] for node in self.nodes])
        else:
            raise NotImplementedError

    def get_backlog(self):
        return np.sum([self.buffers[node] for node in self.nodes])

    def get_obs(self):
        return np.array([self.buffers[node] for node in self.nodes[:-1]])

    def get_cap(self):
        return np.array(self.Cap)

    def get_stable_action(self, type = "LQ"):
        obs = self.get_obs()
        if type == "LQ":
            action = np.random.choice(np.where(obs == obs.max())[0]) + 1  # LQ

        elif type == "SQ":
            obs[obs == 0] = 1000000
            action = np.random.choice(np.where(obs == obs.min())[0]) + 1
        elif type == "RQ":
            action = np.random.choice(np.where(obs > 0)[0]) + 1
        elif type == "LCQ":
            cap = self.get_cap()
            connected_obs = cap*obs
            action = np.random.choice(np.where(connected_obs == connected_obs.max())[0]) + 1
        elif type == "MWQ":
            p_cap = 1 - self.unreliabilities
            weighted_obs = p_cap * obs
            action = np.random.choice(np.where(weighted_obs == weighted_obs.max())[0]) + 1
        elif type == "Optimal":
            p_cap = 1 - self.unreliabilities
            non_empty = obs > 0
            action  = np.argmax(p_cap*non_empty)+1
        return action.astype(int)


    def get_mask(self):
        # returns a vector of length n_queues + 1
        mask = np.bool_(np.zeros(self.n_queues+1))
        if self.get_obs().sum() == 0:
            mask[1:] = True
        else:
            mask[0] = True
            mask[1:] = self.get_obs() < 1
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
                caps[link] = bern_rv(self.rng, num=capacity, prob = probability)

        for link, l_info in cap_dict.items():
            if isinstance(link, str):
                link = eval(link)
            if link == (0,0):
                continue
            rv = bern_rv(self.rng, num = l_info['capacity'], prob= l_info['probability'])
            caps[link] = rv

            # generate unreliabilities
        unrel = []
        for link in self.links:
            if link[1] == self.destination:
                unrel.append(1-caps[link].prob)
        self.unreliabilities = np.array(unrel)


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



