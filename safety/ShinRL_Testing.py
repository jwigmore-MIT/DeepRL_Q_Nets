import chex
import shinrl
import gymnasium as gym
import jax.numpy as jnp
import jax
import numpy as np

from functools import reduce, singledispatch


def create_state_map(low, high):
    state_elements = []
    for l, h in zip(low, high):
        state_elements.append(np.arange(l, h + 1))
    state_combos = np.array(np.meshgrid(*state_elements)).T.reshape(-1, len(state_elements))

    return state_combos, state_combos[:, :2]

def create_transition_map(state, p_arrival, p_link):
    pass

class M2A1Config(shinrl.EnvConfig):
    servers = 2
    s_max = 10
    high_servers = s_max*np.ones([servers,1])
    high_links = np.ones([servers,1])
    high = np.concatenate([high_servers, high_links])
    low = np.zeros([servers*2,1])
    p_X = np.array([0.2, 0.1])
    p_Y = np.array([0.3, 0.95])

    int2state, int2obs = create_state_map(low, high)
    transition_map, transion_prob = create_transition_map(int2state, p_X, p_Y)

    dS: int = (servers*(s_max+1))**2
    dA = servers

    # dS: int = 484 # number of states = (number of servers * (s_max +1))**(number of server states) = (2*(10+1))*(2**2) = 44
    # dA: int = 2 # number of actions = number of servers +1
    discount: float = 0.99
    horizon: int = 1000




class M2A1ShinEnv(shinrl.ShinEnv):
    DefaultConfig = M2A1Config

    def __init__(self, config = None):
        #self.map_states()
        super().__init__(config)



    @property
    def dS(self):
        return self.config.dS

    @property
    def dA(self):
        return self.config.dA

    @property
    def observation_space(self):
        return gym.spaces.Box(low=np.array([0, 0]), high=np.array([10, 10]))
    @property
    def state_space(self):
        return gym.spaces.Box(low=np.array([0, 0, 0, 0]), high=np.array([10, 10, 1, 1]))
    @property
    def action_space(self):
        return gym.spaces.Discrete(self.dA)

    def map_states(self):
        # enumerate all states in self.observation_space
        # map it to an integer
        # store the mapping in a dictionary
        state_elements = []
        for i in range(self.state_space.shape[0]):
            state_elements.append(np.arange(self.state_space.low[i], self.state_space.high[i]+1))
        # get all combinations of state elements
        state_combos = np.array(np.meshgrid(*state_elements)).T.reshape(-1, len(state_elements))
        self.int_2_box_state = state_combos
        self.int_2_box_obs = state_combos[:, :2]

    def init_probs(self):
        return jnp.concatenate([np.array([1.0]), np.zeros(self.dS-1)])

    def transition(self, state, action):
        unflatten_state = self.int_2_box_state[state.val]
        return unflatten_state

    def observation(self, state):
        #obs = jnp.take(self.int_2_box_obs, state, axis = 1)
        #return obs
        return observation_tuple(self.config, state)



    def reward(self, state, action):
        return reward(self.config, state, action)
        # uf_state = self.int_2_box_state[state.val]
        #
        # delivered_0 = jnp.array((uf_state[:,0]>0)*(action.val[:]==0),dtype = int)
        # delivered_1 = jnp.array((uf_state[:,1]>0)*(action.val[:]==1),dtype = int)
        # return -uf_state[:,0] - uf_state[:,1] + delivered_0 + delivered_1

    # should go in calc.py

@jax.jit
def observation_tuple(config, state):
    # produces an observation tuple from the state
    tup = jnp.take(config.int2obs, state, axis = 0)
    return tup

@jax.jit
def reward(config, state, action):
    # NEED TO DO THE SAME THING FOR THE REWARD FUNCTION
    box_state = jnp.take(config.int2state, state, axis = 0)
    obs_0 = jnp.take(box_state,0, axis = 0)
    obs_1 = jnp.take(box_state,1, axis = 0)
    state_0 = jnp.take(box_state,2, axis = 0)
    state_1 = jnp.take(box_state,3, axis = 0)
    delivered_0 = jnp.min(jnp.array([obs_0,(state_0 > 0)*(action == 0)]))
    delivered_1 = jnp.min(jnp.array([state_1,(state_1 > 0)*(action == 1)]))
    return -obs_0 - obs_1 + delivered_0 + delivered_1

def transition(config, state, action):
    box_state = jnp.take(config.int2state, state, axis = 0)






if __name__ == "__main__":
    from shinrl.envs.cartpole.env import CartPole
    #env1 = CartPole()
    env = M2A1ShinEnv()