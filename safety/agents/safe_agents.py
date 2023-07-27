import torch.nn as nn
import torch
import numpy as np
from NonDRLPolicies.Backpressure import MCMHBackPressurePolicy
from NonDRLPolicies.StaticPolicies import JoinTheShortestQueuePolicy, JoinARandomQueuePolicy
from copy import deepcopy




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


class ShrinkingInterventioner:

    def __init__(self, safe_policy, trigger_state = 0, shrink_rate = 1e-4):
        super().__init__()
        self.safe_policy = safe_policy
        self.trigger_state = trigger_state
        self.shrink_rate = shrink_rate # we increase the trigger state by this amount each time we intervene
        self.triggers = 0
        self._init_trigger_state = deepcopy(trigger_state)


    def check_safety(self, state):
        # In unsafe state, return False and int. prob = 1, else, return true and int prob = 0

        if np.sum(state) > self.trigger_state:
            self.triggers +=1
            self.trigger_state += self.shrink_rate
            return False, 1
        else:
            return True, 0

    def act(self, state, device = None):
        return self.safe_policy.act(state)


class ProbabilisticInterventioner:
    """
    Allow for some probability of intervention in unsafe states near the boundary
    """
    def __init__(self, safe_actor, trigger_state = 0, omega = 1.0, arrival_mod = -2):
        super().__init__()
        self.safe_actor = safe_actor
        self.trigger_state = trigger_state
        self.omega = omega
        self.arrival_mod = arrival_mod

    def check_safety(self, state):
        # In unsafe state, return False, otherwise True
        gap = self.trigger_state - np.sum(state) - 2
        prob = min(1,np.exp(-self.omega * gap))  # probability of intervention
        if np.random.rand() < prob:
            return False, prob
        else:
            return True, prob


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
        self.force_safe = False

    def change_mode(self, mode = 'train'):
        if mode == 'train':
            self.neural_agent.actor.training = True
        elif mode == 'test':
            self.neural_agent.actor.training = False
        else:
            Exception(f"mode ({mode}) does not exist")
    def check_safety(self, state):
        # In unsafe state, return False, otherwise True
        return self.safe_actor.check_safety(state)

    def act(self, state, device = None):
        # check if state is safe
        safe, int_prob = self.check_safety(state)

        # if the state is safe, and we are not forcing the neural agent then use the neural agent
        if safe and not self.force_nn and not self.force_safe:
            action, nn_obs =  self.neural_agent.act(state)
        elif self.force_safe:
            safe = 2
            action = self.safe_actor.act(state)
            nn_obs = self.neural_agent.obs_normalizer.normalize(state, update=True)
        else:
            action = self.safe_actor.act(state)
            nn_obs = self.neural_agent.obs_normalizer.normalize(state, update=True)
        return action, nn_obs, 1-safe, int_prob

    def update(self, batch, pretrain = False):
        # Need to factor in the intervention
        return self.neural_agent.update(batch, pretrain)

    def get_log_prob(self, obs, action):
        return self.neural_agent.get_log_prob(obs, action)




def init_safe_agent(safety_config, neural_agent, env):
    """
    Initialize a safe agent from a config file
    """
    # Create Safe Policy
    if safety_config.safe_policy == "BP":
        safe_policy = MCMHBackPressurePolicy(env, **safety_config.args.toDict())
    elif safety_config.safe_policy == "JSQ":
        safe_policy = JoinTheShortestQueuePolicy(env)
    elif safety_config.safe_policy == "JRQ":
        safe_policy = JoinARandomQueuePolicy(env)
    else:
        raise NotImplementedError("Safe policy not implemented")

    # Check the trigger state
    if safety_config.trigger_state is None:
        raise ValueError("Trigger state not specified")

    # Create the safe actor
    if hasattr(safety_config, "mod") and safety_config.mod is not None:
        if safety_config.mod == "prob":
            safe_actor = ProbabilisticInterventioner(safe_policy, trigger_state = safety_config.trigger_state, **safety_config.mod_args.toDict())
        elif safety_config.mod == "shrink":
            safe_actor = ShrinkingInterventioner(safe_policy, trigger_state = safety_config.trigger_state, **safety_config.mod_args.toDict())
    else:
        safe_actor = Interventioner(safe_policy, trigger_state = safety_config.trigger_state)

    # Create the safe agent
    safe_agent = SafeAgent(neural_agent, safe_actor)

    return safe_agent



