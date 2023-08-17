import chex
import shinrl
import gymnasium as gym
import jax.numpy as jnp
import jax
import numpy as np

from functools import reduce, singledispatch


@chex.dataclass
class M2A1Config(shinrl.EnvConfig):
    dS: int = 484 # number of states = (number of servers * (s_max +1))**(number of server states) = (2*(10+1))*(2**2) = 44
    dA: int = 2 # number of actions = number of servers +1
    discount: float = 0.99
    horizon: int = 1000

class M2A1ShinEnv(shinrl.ShinEnv):
    DefaultConfig = M2A1Config

    def __init__(self, config = None):
        self.map_states()
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
        return jnp.array(self.int_2_box_obs[state.val])

    def reward(self, state, action):
        uf_state = self.int_2_box_state[state.val]

        delivered_0 = jnp.array((uf_state[:,0]>0)*(action.val[:]==0),dtype = int)
        delivered_1 = jnp.array((uf_state[:,1]>0)*(action.val[:]==1),dtype = int)
        return -uf_state[:,0] - uf_state[:,1] + delivered_0 + delivered_1


if __name__ == "__main__":
    env = M2A1ShinEnv()