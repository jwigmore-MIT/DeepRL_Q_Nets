import torch.nn as nn
import torch
import numpy as np
from NonDRLPolicies.Backpressure import MCMHBackPressurePolicy




class Interventioner:

    def __init__(self, safe_policy, trigger_state = 0):
        super().__init__()
        self.safe_policy = safe_policy
        self.trigger_state = trigger_state



    def check_safety(self, state):
        # In unsafe state, return False and int. prob = 1, else, return true and int prob = 0
        if np.sum(state) > self.trigger_state:
            return False, 1
        else:
            return True, 0

    def act(self, state, device = None):
        return self.safe_policy.act(state)


class ProbabilisticInterventioner(nn.Module):
    """
    Allow for some probability of intervention in unsafe states near the boundary
    """
    def __init__(self, safe_actor, trigger_state = 0, omega = 1.0):
        super().__init__()
        self.safe_actor = safe_actor
        self.trigger_state = trigger_state
        self.omega = omega

    def check_safety(self, state):
        # In unsafe state, return False, otherwise True
        gap = self.trigger_state - np.sum(state)
        prob = min(1,np.exp(-self.omega * gap))  # probability of intervention
        if np.random.rand() < prob:
            return (False, prob)
        else:
            return (True, prob)


    def act(self, state, device = None):
        return self.safe_actor.act(state)

class SafeAgent(nn.Module):
    def __init__(self, neural_agent, safe_actor):
        """
        params:
        neural_agent: the neural agent (e.g. PPO-Agent)
        safe_actor: the safe actor/policy (e.g. Backpressure)
        """
        super().__init__()
        self.safe_actor = safe_actor
        self.neural_agent = neural_agent
        if hasattr(neural_agent,"obs_normalizer"):
            self.obs_normalizer = neural_agent.obs_normalizer
        self.force_nn = False # to make it so the agent uses the neural agent even if it is in an unsafe state

    def check_safety(self, state):
        # In unsafe state, return False, otherwise True
        return self.safe_actor.check_safety(state)

    def act(self, state, device = None):
        # check if state is safe
        safe, int_prob = self.check_safety(state)

        if safe and not self.force_nn:
            action, nn_obs =  self.neural_agent.act(state)
        else:
            action = self.safe_actor.act(state)
            nn_obs = self.neural_agent.obs_normalizer.normalize(state, update=True)
        return action, nn_obs, 1-safe, int_prob

    def update(self, batch):
        # Need to factor in the intervention
        return self.neural_agent.update(batch)

    def get_log_prob(self, obs, action):
        return self.neural_agent.get_log_prob(obs, action)




def init_safe_agent(safety_config, neural_agent, env):
    """
    Initialize a safe agent from a config file
    """
    # Create Safe Policy
    if safety_config.safe_policy == "BP":
        safe_policy = MCMHBackPressurePolicy(env, **safety_config.args.toDict())
    else:
        raise NotImplementedError("Safe policy not implemented")

    # Check the trigger state
    if safety_config.trigger_state is None:
        raise ValueError("Trigger state not specified")

    # Create the safe actor
    safe_actor = Interventioner(safe_policy, trigger_state = safety_config.trigger_state)

    # Create the safe agent
    safe_agent = SafeAgent(neural_agent, safe_actor)

    return safe_agent



