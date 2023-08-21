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
        # Links
        self.obs_links = net_para["obs_links"] # whether or not the link states are observable
        self.links = [tuple(link) for link in eval(net_para['links'])]  # observable links <list> of <tup>
        self.capacities_fcn = self._extract_capacities(net_para['capacities'])
        self.Cap = self._sim_capacities()
        # Arrivals/Classes
        self.classes, self.destinations = self._extract_classes(net_para['classes'])
        # Spaces
        state_low = np.concatenate((np.zeros(self.n_queues), np.zeros(self.n_queues)))
        state_high = np.concatenate((1e3 * np.ones(self.n_queues), np.ones(self.n_queues)))
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

        # Step 0: Convert action to the queue number and check if it is valid
        if debug: init_buffer = deepcopy(self.buffers)
        current_capacities = deepcopy(self.Cap)

        if action > self.n_queues:
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
        elif type == "MRQ": #Most Reliable Queue
            p_cap = 1 - self.unreliabilities
            valid_obs = q_obs > 0
            # choose the most reliable queue that is not empty
            valid_p_cap = p_cap * valid_obs
            if np.all(valid_obs == False):
                action = 0
            else:
                action = np.random.choice(np.where(valid_p_cap == valid_p_cap.max())[0]) + 1
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


    def get_mask(self):
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
            mask[1:] = np.logical_or(mask[1:], 1-self.get_cap())
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



