from functools import singledispatch
import gymnasium as gym
# from Environments.MultiClassMultiHop import MultiClassMultiHop
from stable_baselines3.common.monitor import Monitor
import numpy as np
from dataclasses import dataclass
from param_extractors import parse_env_json
from collections import defaultdict


@dataclass
class Config:
    pass



@singledispatch
def keys_to_strings(ob):
    return ob

@keys_to_strings.register
def _handle_dict(ob: dict):
    return {str(k): keys_to_strings(v) for k, v in ob.items()}

@keys_to_strings.register
def _handle_list(ob: list):
    return [keys_to_strings(v) for v in ob]

@singledispatch
def keys_to_ints(ob):
    return ob

@singledispatch
def keys_to_tup(ob):
    return ob

@keys_to_tup.register
def _handle_dict(ob: dict):
    return {tuple(k): v for k, v in ob.items}

@keys_to_ints.register
def _handle_dict(ob: dict):
    return {int(k): keys_to_ints(v) for k,v in ob.items()}

@keys_to_ints.register
def _handle_list(ob: list):
    return [keys_to_ints(v) for v in ob]


class FlatActionWrapper(gym.ActionWrapper):
    """
    This action wrapper maps flattened actions <nd.array> back to dictionary
    actions of which the Base environment understands
    """

    def __init__(self, MCMH_env):
        super(FlatActionWrapper, self).__init__(MCMH_env)
        self.action_space = self.flatten_action_space()

    def action(self, action: np.ndarray):
        return self.unflatten_action(action)


class StepLoggingWrapper(gym.Wrapper):
    "Custom wrapper to log some of the outputs of the step function"

    def __init__(self, env, log_keys = ["backlog"], filename = "log.csv"):
        super(StepLoggingWrapper, self).__init__(env)
        self.log_keys = log_keys
        self.filename = filename
        self.eps = 0
        self.log ={self.eps: {}}
        for key in self.log_keys:
            self.log[self.eps][key] = []



    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        for key in self.log_keys:
            self.log[self.eps][key].extend(info[key])
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.eps += 1
        self.log[self.eps] = {}
        for key in self.log_keys:
            self.log[self.eps][key] = []
        return self.env.reset(**kwargs)

    def save_log(self):
        import csv
        with open(self.filename, 'w') as f:
            writer = csv.writer(f)
            for key, value in self.log.items():
                writer.writerow([key, value])



def generate_env(config, max_steps = None):
    """
    Generates the environment and applies the appropriate wrappers
    """
    from Environments.MultiClassMultiHop import MultiClassMultiHop
    parse_env_json(config.root_dir + config.env.env_json_path, config)
    env = MultiClassMultiHop(config=config)
    # required wrappers
    env = FlatActionWrapper(env)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.ClipAction(env)

    # optional wrappers
    if max_steps is not None:
        env = gym.wrappers.time_limit.TimeLimit(env, max_episode_steps=max_steps)
    if config.env.wrappers.normalize_obs:
        env = gym.wrappers.NormalizeObservation(env)
    if config.env.wrappers.normalize_reward:
        env = gym.wrappers.NormalizeReward(env)
    if config.env.wrappers.record_stats:
        env = gym.wrappers.RecordEpisodeStatistics(env)
    if hasattr(config.env.wrappers, "scale_reward"):
        if isinstance(config.env.wrappers.scale_reward, float) or isinstance(config.env.wrappers.scale_reward, int):
            env = gym.wrappers.TransformReward(env, lambda x: (x- config.env.wrappers.scale_reward) / (-config.env.wrappers.scale_reward) * (2) - 1)
    if hasattr(config.env.wrappers, "norm_reward"):
        # Normalize rewards by the episode length
        if isinstance(config.env.wrappers.norm_reward, float) or isinstance(config.env.wrappers.norm_reward, int):
            env = gym.wrappers.TransformReward(env, lambda x: x/config.env.wrappers.norm_reward)
        # r \in [-50,0] -> r \in [-1,1] or r \in [a,b] -> r' \in [c,d]
        # r' = (r - a)/(b-a) * (d-c) + c


    return env


def SB3_generate_env(config: Config, max_steps = 1000, monitor_settings = None, backpressure = False):
    """
    Generates the environment and applies the wrappers
    """
    from Environments.MultiClassMultiHop import MultiClassMultiHop
    parse_env_json(config.root_dir + config.env.env_json_path, config)
    env = MultiClassMultiHop(config=config)
    # required wrappers
    env = FlatActionWrapper(env)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.ClipAction(env)
    # optional wrappers

    env = gym.wrappers.time_limit.TimeLimit(env, max_episode_steps=max_steps)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if monitor_settings is not None:
        monitor_settings["info_keywords"] = tuple(monitor_settings["info_keywords"])
        env = Monitor(env, **monitor_settings)

    if not backpressure:
        if config.env.normalize_obs:
            env = gym.wrappers.NormalizeObservation(env)
        if config.env.normalize_reward:
            env = gym.wrappers.NormalizeReward(env)
    env = StepLoggingWrapper(env)
    return env


class Heap():

    def __init__(self):
        self.array = []
        self.size = 0
        self.pos = []

    def newMinHeapNode(self, v, dist):
        minHeapNode = [v, dist]
        return minHeapNode

    # A utility function to swap two nodes
    # of min heap. Needed for min heapify
    def swapMinHeapNode(self, a, b):
        t = self.array[a]
        self.array[a] = self.array[b]
        self.array[b] = t

    # A standard function to heapify at given idx
    # This function also updates position of nodes
    # when they are swapped.Position is needed
    # for decreaseKey()
    def minHeapify(self, idx):
        smallest = idx
        left = 2 * idx + 1
        right = 2 * idx + 2

        if (left < self.size and
                self.array[left][1]
                < self.array[smallest][1]):
            smallest = left

        if (right < self.size and
                self.array[right][1]
                < self.array[smallest][1]):
            smallest = right

        # The nodes to be swapped in min
        # heap if idx is not smallest
        if smallest != idx:
            # Swap positions
            self.pos[self.array[smallest][0]] = idx
            self.pos[self.array[idx][0]] = smallest

            # Swap nodes
            self.swapMinHeapNode(smallest, idx)

            self.minHeapify(smallest)

    # Standard function to extract minimum
    # node from heap
    def extractMin(self):

        # Return NULL wif heap is empty
        if self.isEmpty() == True:
            return

        # Store the root node
        root = self.array[0]

        # Replace root node with last node
        lastNode = self.array[self.size - 1]
        self.array[0] = lastNode

        # Update position of last node
        self.pos[lastNode[0]] = 0
        self.pos[root[0]] = self.size - 1

        # Reduce heap size and heapify root
        self.size -= 1
        self.minHeapify(0)

        return root

    def isEmpty(self):
        return True if self.size == 0 else False

    def decreaseKey(self, v, dist):

        # Get the index of v in  heap array

        i = self.pos[v]

        # Get the node and update its dist value
        self.array[i][1] = dist

        # Travel up while the complete tree is
        # not heapified. This is a O(Logn) loop
        while (i > 0 and self.array[i][1] <
               self.array[(i - 1) // 2][1]):
            # Swap this node with its parent
            self.pos[self.array[i][0]] = (i - 1) // 2
            self.pos[self.array[(i - 1) // 2][0]] = i
            self.swapMinHeapNode(i, (i - 1) // 2)

            # move to parent index
            i = (i - 1) // 2;

    # A utility function to check if a given
    # vertex 'v' is in min heap or not
    def isInMinHeap(self, v):

        if self.pos[v] < self.size:
            return True
        return False


def printArr(dist, n):
    print ("Vertex\tDistance from source")
    for i in range(n):
        print(f"{i} \t\t {dist[i]}")

class Graph():

    def __init__(self, V):
        self.V = V
        self.graph = defaultdict(list)

    # Adds an edge to an undirected graph
    def addEdge(self, src, dest, weight):

        # Add an edge from src to dest.  A new node
        # is added to the adjacency list of src. The
        # node is added at the beginning. The first
        # element of the node has the destination
        # and the second elements has the weight
        newNode = [dest, 1]
        self.graph[src].insert(0, newNode)



    # The main function that calculates distances
    # of shortest paths from src to all vertices.
    # It is a O(ELogV) function
    def dijkstra(self, src):

        V = self.V  # Get the number of vertices in graph
        dist = []  # dist values used to pick minimum
        # weight edge in cut

        # minHeap represents set E
        minHeap = Heap()

        #  Initialize min heap with all vertices.
        # dist value of all vertices
        for v in range(V):
            dist.append(np.inf)
            minHeap.array.append(minHeap.
                                 newMinHeapNode(v, dist[v]))
            minHeap.pos.append(v)

        # Make dist value of src vertex as 0 so
        # that it is extracted first
        minHeap.pos[src] = src
        dist[src] = 0
        minHeap.decreaseKey(src, dist[src])

        # Initially size of min heap is equal to V
        minHeap.size = V;

        # In the following loop,
        # min heap contains all nodes
        # whose shortest distance is not yet finalized.
        while minHeap.isEmpty() == False:

            # Extract the vertex
            # with minimum distance value
            newHeapNode = minHeap.extractMin()
            u = newHeapNode[0]

            # Traverse through all adjacent vertices of
            # u (the extracted vertex) and update their
            # distance values
            for pCrawl in self.graph[u]:

                v = pCrawl[0]

                # If shortest distance to v is not finalized
                # yet, and distance to v through u is less
                # than its previously calculated distance
                if (minHeap.isInMinHeap(v) and
                        dist[u] != 1e7 and \
                        pCrawl[1] + dist[u] < dist[v]):
                    dist[v] = pCrawl[1] + dist[u]

                    # update distance value
                    # in min heap also
                    minHeap.decreaseKey(v, dist[v])

        printArr(dist, V)
        return dist

