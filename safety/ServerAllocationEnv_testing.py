from clean_rl_utils import generate_clean_rl_env
import numpy as np
from safety.utils import clean_rl_ppo_parse_config
from Environments.ServerAllocation import ServerAllocationMDP
from tqdm import tqdm
import random
import pickle
"""
TODO:
1. Implement and test optimal policy for ConnectedServerAllocation Problem
2. Run test on LCQ, RCQ, and OCQ policies
3. Go through IA AR PPO code to verify it will work with connected
    - Try to make the code agnostic to the problem type i.e. ServerAllocation, ConnectedServerAllocation, ServerAssignment
    - Will have to implement fake masking on ServerAssignment for this
4. Run IA AR PPO on ConnectedServerAllocation Problem
5. Implement other version of Bai's problems with more nodes

"""
config_file = "clean_rl/ServerAllocation/M3/M3A1-O_IA_AR_PPO.yaml"
#config_file = "clean_rl/MSSA_N2S1_config1.yaml"
args = clean_rl_ppo_parse_config(config_file)
env = generate_clean_rl_env(args, env_type= "ServerAllocation", normalize = False)()
debug = False
random.seed(args.seed)
np.random.seed(args.seed)
env.reset(seed = args.seed)

type = "MWCQ" # Longest Queue (LQ), Random Queue (RQ), Longest Connected Queue (LCQ), Max Weighted Queue (MWQ)
#DP_policy = pickle.load(open("DP/M2A2_policy_table.p", "rb"))
# mdp = ServerAllocationMDP(env, 5)
# mdp.estimate_tx_matrix(env, max_samples = 100)
# mdp.do_VI(max_iterations=10)
# #mdp.get_VI_policy()
# mdp.save_MDP(f"saved_MDPs/M2A2_O_MDP.p")
if type == "DP":
    mdp = pickle.load(open("saved_MDPs/M2A3_O_MDP.p", "rb"))




arrivals = 0
cap = 0
delivered = 0
rewards = 0
backlogs = 0
test_length = 10000
pbar = tqdm(range(int(test_length)))
actions = np.zeros(test_length)
observations = np.zeros((test_length, env.observation_space.shape[0]))
masks = np.zeros((test_length, env.get_mask().shape[0]))


for t in pbar:
    #action = env.action_space.sample() # JRQ
    # obs = env.get_obs()
    # clip_obs = np.clip(obs, 0, 10)
    # state_tup = tuple(clip_obs)
    #
    # #action = policy[state_tup]
    # action = 2 if obs[1] > 0 else 1
    obs = env.get_obs()
    mask = env.get_mask()
    if env.get_backlog() == 0:
        action = 0
    elif type == "DP":
        clip_obs = np.clip(obs, 0, mdp.q_max-2)
        action = mdp.use_policy(clip_obs)
    else:
        action = env.get_stable_action(type = type)
    if mask[action] == 1:
        Exception("Invalid Action")

        # if type == "LQ":
        #     action = np.argmax(env.get_obs()) +1 # LQ
        # elif type == "RQ":
        #     action = np.random.choice(np.where(env.get_obs() > 0)[0]) +1
        # elif type == "LCQ":
        #     cap = env.get_cap()
        #     obs = env.get_obs()
        #     connected_obs = cap*obs[:-1]
        #     action = np.argmax(connected_obs) +1
        # elif type == "MWQ":
        #     p_cap = env.unreliabilities
        #     obs = env.get_obs()
        #     weighted_obs = p_cap*obs[:-1]
        #     action = np.argmax(weighted_obs) +1
        # LQ_action = np.argmax(env.get_obs()) +1
        # if LQ_action == action:
        #     pass
        # else:
        #     print(f"LQ action: {LQ_action} action:  {action}")
    # if t > 10000000:
    #     debug = True
    # else:
    #     debug = False
    step = env.step(action, debug=debug)
    actions[t] = action
    arrivals += step[4]['n_arrivals']
    delivered += step[4]['delivered']
    cap += np.array(step[4]['current_capacities'])
    observations[t] = env.get_obs()
    masks[t] = mask
    rewards+=step[1]
    backlogs += step[4]['backlog']

average_reward = np.sum(rewards)/test_length
lta_average_rewards = np.cumsum(rewards)/np.arange(1, test_length+1)
average_backlog = np.sum(backlogs)/test_length
average_arrivals = arrivals/test_length
average_sum_arrivals = np.sum(arrivals)/test_length
average_reliability = cap/test_length
action_dist = np.bincount(actions.astype(int))/test_length
mode_action = np.argmax(np.bincount(actions.astype(int)))




