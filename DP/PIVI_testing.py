import numpy as np
import random
from collections import defaultdict
from itertools import product
from mdp import MDP
from value_iteration import ValueIteration
import pickle


class ServerAllocationMDP(MDP):
    def __init__(self):
        self.A = np.arange(3)
        self.Servers = 2
        self.P_X = np.array([0.2, 0.1])
        self.P_Y = np.array([0.3, 0.95])
        self.s_max = 20

        self.discount = 0.99

    def get_states(self):
        # return all possible states from 0 to s_max for each server
        return create_state_map(np.zeros(self.Servers), self.s_max * np.ones(self.Servers))
    def get_transitions(self, state, action):
        i = action - 1
        if state[i] < 1:
            i = -1
        # enumerate all next_states that are +-1 from the current state
        next_states = perturb_array_combinations(state, -1, 1)
        p = []
        for ns in next_states:
            # if ns is a valid state
            if np.any(ns < 0):
                p.append(0)
            else:
                delta = ns-state
                p_elements = []
                for j ,element in enumerate(ns):
                    if j == i:
                        p_elements.append(self.P_Y[j] * (1 - self.P_X[j]) * (delta[j] == -1) + (delta[j] == 0) * (self.P_Y[j] * self.P_X[j] + (1 - self.P_Y[j]) * (1 - self.P_X[j])) + (delta[j] == 1) * (1 - self.P_Y[j]) * self.P_X[j]) #packet was delivered and no arrival
                    else:
                        p_elements.append(self.P_X[j] * (delta[j] == 1) + (1 - self.P_X[j]) * (delta[j] == 0))
                p.append(np.prod(p_elements))
        # drop all states and probabilities where p= 0
        p = np.array(p)
        next_states = next_states[p != 0]
        p = p[p != 0]
        p = p / np.sum(p)
        # make p a column array
        p = p.reshape(-1, 1)
        # merge into a single list
        transitions = list(zip(next_states.tolist(), p))

        return transitions

    def get_reward(self, state, action, next_state):
        if np.any(np.array(next_state) >=  self.s_max):
            return -100
        else:
            return -np.sum(next_state)

    def get_actions(self, state):
        return self.A

    def get_initial_state(self):
        return np.zeros(self.Servers)

    def is_terminal(self, state):
        return False

    def get_discount_factor(self):
        return self.discount

    def get_goal_states(self):
        return None



# Example usage
# input_array = np.array([5, 6])
# a = -1
# b = 1
# result = perturb_array_combinations(input_array, a, b)
# print(result)


def perturb_array_combinations(arr, a, b):
    perturbed_combinations = []
    perturbation_range = range(a, b + 1)

    for perturbations in product(perturbation_range, repeat=len(arr)):
        perturbed_combinations.append(arr + np.array(perturbations))

    return np.vstack(perturbed_combinations)

def create_state_map(low, high):
        state_elements = []
        for l, h in zip(low, high):
            state_elements.append(np.arange(l, h + 1))
        state_combos = np.array(np.meshgrid(*state_elements)).T.reshape(-1, len(state_elements))

        return state_combos

from tabular_value_function import TabularValueFunction

mdp = ServerAllocationMDP()
values = TabularValueFunction()
ValueIteration(mdp, values).value_iteration(max_iterations=1000)

policy = values.extract_policy(mdp)
# convert policy.policy_table to normal dictionary
policy_table = dict(policy.policy_table)


pickle.dump(policy_table, open("M2A1_policy_table.p", "wb"))


