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

class MultiStepServerAssignment(gym.Env):

    def __init__(self, net_para):
        super(MultiStepServerAssignment, self).__init__()

        self.nodes = eval(net_para['nodes'])
        self.destination = max(self.nodes)
        self.n_servers = len(self.nodes) - 2
        self.links = [tuple(link) for link in eval(net_para['links'])]  # observable links <list> of <tup>
        self.classes, self.destinations = self._extract_classes(net_para['classes'])
        self.capacities_fcn = self._extract_capacities(net_para['capacities'])
        self.observation_space = gym.spaces.Box(low=0, high=1e6, shape=(len(self.nodes),), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.n_servers)
        self.buffers = {node: 0 for node in self.nodes}
        self.Cap = self._sim_capacities()
    def step(self, action, debug = False):
        # the action is an integer indicating the server to send the arrived packet too
        info = {}
        interarrival_time = 0

        # Step 0: Convert action to the buffer number and check if it is valid
        if debug: init_buffer = deepcopy(self.buffers)
        server_action = action + 2
        if server_action < 2 or server_action > self.n_servers+1:
            raise ValueError(f"Invalid action {action} for {self.n_servers} servers")
        # Step 2: initialize capacities and delivered
        capacities = np.zeros(len(self.links))
        delivered = np.zeros(len(self.nodes))
        # Step 1: Send the packet to the appropriate server

        if self.buffers[1] > 0:
            self.buffers[server_action] += 1
            self.buffers[1] -= 1
            ignore_action = False
        else:
            ignore_action = True
        if debug: post_action_buffer = deepcopy(self.buffers)




        # Step 3: Get the corresponding reward


        # Step 4: Simulate New Arrivals
        n_arrivals = 0
        interarrival_time = 0
        reward = 0
        while n_arrivals == 0:
            interarrival_time += 1
            capacities += self._sim_capacities()
            delivered += self._serve_step()
            reward += self._get_reward()
            n_arrivals = self._sim_arrivals()




        average_capacity = capacities/interarrival_time
        if debug: post_arrival_buffer = deepcopy(self.buffers)

        # Step 5: Get the new state
        next_state = self._get_obs()
        backlog = self._get_backlog()

        # Step 6: Fill out the info dict
        info = {"ignore_action": ignore_action, "average_capacities": average_capacity, "delivered": delivered,
                "backlog": backlog, "interarrival_time": interarrival_time,"n_arrivals": n_arrivals, "env_reward": reward}
        terminated = False
        truncated = False

        if debug: self._debug_printing(init_buffer, ignore_action, action, server_action,
                        post_action_buffer, interarrival_time,
                        post_arrival_buffer, delivered, average_capacity, n_arrivals, reward)

        return next_state, reward, terminated, truncated, info

    def reset(self, seed = None):
        super().reset(seed = seed)
        self.buffers = {node: 0 for node in self.nodes}
        state = self._get_obs()
        info = {"reset": True}
        return state, info

    def _debug_printing(self, init_buffer, ignore_action, action, server_action,
                        post_action_buffer, interarrival_time,
                        post_arrival_buffer, delivered, average_capacity, n_arrivals, reward):
        print("="*20)
        print(f"Initial Buffer: {init_buffer}")
        print(f"Ignore Action: {ignore_action}")
        print(f"Action: {action} ; Server Action: {server_action}")
        print(f"Post Action Buffer: {post_action_buffer}")
        print(f"Interarrival Time: {interarrival_time}")
        print(f"Average Capacity: {average_capacity}")
        print(f"Delivered: {delivered}")
        print("Arrivals: ", n_arrivals)
        print(f"Post Arrival Buffer: {post_arrival_buffer}")
        print(f"Reward: {reward}")
        print("="*20)
        print("\n")

    def _sim_capacities(self):
        self.Cap = {key: value.sample() for key, value in self.capacities_fcn.items()}
        # return copy of the second half of Cap.keys as np.array

        return np.array(list(self.Cap.values()))
    def _serve_step(self):
        delivered = np.zeros(len(self.nodes))
        for server in self.nodes[1:-1]:
            server_capacity = self.Cap[server,self.destination]
            # if the server has capacity, reduce the buffer by the server capacity
            delivered[server] +=  min(self.buffers[server], server_capacity)
            self.buffers[server] = max(0, self.buffers[server] - server_capacity)
        return delivered

    def _get_reward(self, type = "congestion"):

        if type == "congestion":
            return -np.sum([self.buffers[node] for node in self.nodes[1:-1]])
        else:
            raise NotImplementedError

    def _get_backlog(self):
        return np.sum([self.buffers[node] for node in self.nodes[1:-1]])

    def _get_obs(self):
        return np.array([self.buffers[node] for node in self.nodes])
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
            if link == (0,0):
                continue
            rv = bern_rv(num = l_info['capacity'], prob= l_info['probability'])
            caps[link] = rv


        if (0,0) in caps.keys():
            del caps[(0,0)]

        #generate unreliabilities
        self.unrel = []
        for link in self.links:
            self.unrel.append(1-caps[link].prob)
        return caps

    def _extract_classes(self, class_dict):
        classes = {}
        destinations = {}
        for cls_num, cls_info in class_dict.items():
            rv = bern_rv(num = cls_info['arrival'], prob = cls_info['probability'])
            classes[int(cls_num)] = [cls_info['source'], cls_info['destination'], rv]
            destinations[int(cls_num)] = cls_info['destination']
        return classes, destinations

def parse_env_json(json_path, config_args = None):
    import json
    para = json.load(open(json_path))
    env_para = para["problem_instance"]
    if config_args is not None:
        if hasattr(config_args,'env'):
            for key, value in env_para.items():
                setattr(config_args.env, f"{key}", value)
        else:
            for key, value in env_para.items():
                setattr(config_args, f"env.{key}", value)
    return env_para
def generate_clean_rl_MSSA_env(config):
    config = config
    def thunk():
        env_para = parse_env_json(config.root_dir + config.env_json_path, config)
        env_para["seed"] = config.seed
        env = MultiStepServerAssignment(env_para)
        #env = gym.wrappers.TransformReward(env, lambda x: x*config.reward_scale)
        #env = gym.wrappers.TransformObservation(env, lambda x: x*config.obs_scale)
        return env
    return thunk