from collections import defaultdict

from DP.policy import DeterministicPolicy


class TabularPolicy(DeterministicPolicy):
    def __init__(self, default_action=None):
        self.policy_table = defaultdict(lambda: default_action)

    def select_action(self, state):
        if not isinstance(state, tuple):
            state = tuple(state)
        return self.policy_table[state]

    def update(self, state, action):
        if not isinstance(state, tuple):
            state = tuple(state)
        self.policy_table[state] = action
