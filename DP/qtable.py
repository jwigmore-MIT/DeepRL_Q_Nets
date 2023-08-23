from collections import defaultdict

from DP.qfunction import QFunction


class QTable(QFunction):
    def __init__(self, default=0.0):
        self.qtable = defaultdict(lambda: default)

    def update(self, state, action, delta):
        if not isinstance(state, tuple):
            state = tuple(state)
        self.qtable[(state, action)] = self.qtable[(state, action)] + delta

    def get_q_value(self, state, action):
        if not isinstance(state, tuple):
            state = tuple(state)
        return self.qtable[(state, action)]
