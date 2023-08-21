# import
import gymnasium as gym
from copy import deepcopy
import numpy as np

class bern_rv:

    def __init__(self, num = 1, prob = 0.5):
        self.num = num
        self.prob = prob

    def sample(self):
        if self.prob == 1:
            return self.num
        else:
            return int(np.random.choice([0, self.num], 1, p=[1 - self.prob, self.prob]))
class ServerAssignment(gym.Env):

    def __init__(self, net_para):
        super(ServerAssignment, self).__init__()
        # Seeding
        self.rng, self.rng_seed = gym.utils.seeding.np_random(net_para["seed"])
        # Nodes/Buffers/Queues
        self.nodes = self._init_nodes(net_para['nodes'])
        self.n_servers = len(self.nodes) - 2
        self.destination_node = max(self.nodes)
        self.arrival_node = min(self.nodes)
        self.buffers = {node: 0 for node in self.nodes if node != self.destination_node}
        # Links and capacity functions
        self.links = self._init_links(net_para['links'])  # observable links <list> of <tup>
        self.capacities_fcn = self._extract_capacities(net_para['capacities'])
        self._sim_capacities()
        # Arrivals/Classes
        self.classes, self.destinations = self._extract_classes(net_para['classes'])
        # Spaces
        self.observation_space = gym.spaces.Box(low=0, high=1e6, shape=(len(self.buffers),), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.n_servers+1)

    def step(self, action, debug = False):
        # the action is an integer indicating the server to send the arrived packet too
        info = {}
        if debug: debug_strs = []
        # Step 0: Check if action is valid
        if debug: debug_strs.append(f"Initial Buffer: {deepcopy(self.buffers)}")
        if debug: debug_strs.append(f"Mask: {self.get_mask()}")
        server_action = action
        if server_action > self.n_servers+1:
            raise ValueError(f"Invalid action {action} for {self.n_servers} servers")
        curr_capacities = self.get_cap()
        if debug: debug_strs.append(f"Capacities: {curr_capacities}")
        if debug: debug_strs.append(f"Action: {action}")
        # Step 1: Send the packet to the appropriate server
        if action > 0:
            if self.buffers[0] > 0:
                self.buffers[server_action] += 1
                self.buffers[0] -= 1
                ignore_action = False
            else:
                ignore_action = True
        else: # action == 0
            ignore_action = False
        if debug: debug_strs.append(f"Post-Action Buffer: {self.buffers}")

        # Step 2: Simulate the service process
        delivered = self._serve_step()

        if debug: debug_strs.append(f"Post-Service Buffer {deepcopy(self.buffers)}")

        # Step 3: Get the corresponding reward
        reward = self._get_reward()
        if debug: debug_strs.append(f"Reward: {reward}")


        # Step 4: Simulate New Arrivals
        n_arrivals = self._sim_arrivals()
        if debug: debug_strs.append(f"Arrivals: {n_arrivals}")
        if debug: debug_strs.append(f"Post-Arrival Buffer: {deepcopy(self.buffers)}")

        # Simulate new link capacities
        self._sim_capacities()
        new_capacities = self.get_cap()

        if debug: post_arrival_buffer = deepcopy(self.buffers)

        # Step 5: Get the new state
        next_state = self.get_obs()
        backlog = self.get_backlog()

        # Step 6: Fill out the info dict
        info = {"ignore_action": ignore_action, "capacities": curr_capacities, "delivered": delivered,
                "backlog": backlog, "n_arrivals": n_arrivals, "env_reward": reward}
        terminated = False
        truncated = False

        # if debug: self._debug_printing(init_buffer, curr_capacities, delivered, post_serve_buffer,
        #                                ignore_action, action, server_action, post_action_buffer,
        #                                post_arrival_buffer, n_arrivals, reward)

        if debug: self._debug_printing2(debug_strs)

        return next_state, reward, terminated, truncated, info

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        self.buffers = {node: 0 for node in self.nodes}
        state = self.get_obs()
        return state, {}

    def _debug_printing(self, init_buffer, capacities, delivered, post_serve_buffer,
                           ignore_action, action, server_action, post_action_buffer,
                           post_arrival_buffer, n_arrivals, reward):
        print("="*20)
        print(f"Initial Buffer: {init_buffer}")
        print(f"Capacities: {capacities}")
        print(f"Delivered: {delivered}")
        print(f"Post Serve Buffer: {post_serve_buffer}")
        print(f"Ignore Action: {ignore_action}")
        print(f"Action: {action} ; Server Action: {server_action}")
        print(f"Post Action Buffer: {post_action_buffer}")
        print(f"Reward: {reward}")
        print("Arrivals: ", n_arrivals)
        print(f"Post Arrival Buffer: {post_arrival_buffer}")
        print("="*20)
        print("\n")

    def _debug_printing2(self, debug_strs):

        for debug_str in debug_strs:
            print(debug_str)
        print("=" * 20)


    def _init_nodes(self, net_para_nodes):
        nodes = eval(net_para_nodes)
        if 0 not in nodes:
            self_nodes = [node - 1 for node in nodes]
            self.old_net_para_fmt = True
        else:
            self_nodes = nodes
            self.old_net_para_fmt = False
        return self_nodes


    def _init_links(self, net_para_links):
        if not self.old_net_para_fmt:
            self_links = [tuple(link) for link in eval(net_para_links)]
        else:
            self_links = []
            for link in eval(net_para_links):
                self_links.append((link[0]-1, link[1]-1))
        return self_links



    def _sim_capacities(self):
        self.Cap = {key: value.sample() for key, value in self.capacities_fcn.items()}
        # return copy of the second half of Cap.keys as np.array

        return np.array(list(self.Cap.values()))
    def _serve_step(self):
        delivered = 0
        cap = self.get_cap()
        for server in list(self.buffers.keys()):
            if server == self.arrival_node or server == self.destination_node:
                continue
            server_capacity = cap[server-1]
            # if the server has capacity, reduce the buffer by the server capacity
            delivered +=  min(self.buffers[server], server_capacity)
            self.buffers[server] = max(0, self.buffers[server] - server_capacity)
        return delivered

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
        # arrival node + buffers
        return self.get_buffers()

    def get_cap(self):
        return np.array(list(self.Cap.values())[self.n_servers:])

    def get_mask(self):
        """
        Cases:
        1. No arrivals -> mask all but action 0
        2. Arrival -> Mask action 0
        Returns
        -------
        """
        mask = np.bool_(np.zeros(self.n_servers+1))
        if self.get_obs()[0] == 0:
            mask[1:] = True
        else:
            mask[0] = True
        return mask


    def get_stable_action(self, type = "JSQ"):
        if type == "JSQ":
            if self.get_obs()[0] == 0:
                return 0
            else:
                return np.argmin(self.get_buffers()[1:])+1
        else:
            raise NotImplementedError
    def _sim_arrivals(self):
        n_arrivals = 0
        for cls_num, cls_info in self.classes.items():
            source = cls_info[0]
            destination = cls_info[1]
            rv = cls_info[2]
            if rv.sample() == 1:
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
                caps[link] = bern_rv(num=capacity, prob = probability)

        for link, l_info in cap_dict.items():
            if isinstance(link, str):
                link = eval(link)
            if self.old_net_para_fmt:
                old_link = link
                link = (old_link[0]-1, old_link[1]-1)
            if link == (0,0):
                continue
            rv = bern_rv(num = l_info['capacity'], prob= l_info['probability'])
            caps[link] = rv

            # generate unreliabilities
        self.unrel = []
        for link in self.links:
            if link[1] == self.destination_node:
                self.unrel.append(1-caps[link].prob)


        if (0,0) in caps.keys():
            del caps[(0,0)]
        return caps

    def _extract_classes(self, class_dict):
        classes = {}
        destinations = {}
        for cls_num, cls_info in class_dict.items():
            if self.old_net_para_fmt:
                source = cls_info['source'] - 1
                destination = cls_info['destination'] - 1
            else:
                source = cls_info['source']
                destination = cls_info['destination']
            rv = bern_rv(num = cls_info['arrival'], prob = cls_info['probability'])
            classes[int(cls_num)] = [source, destination, rv]
            destinations[int(cls_num)] = destination
        return classes, destinations













