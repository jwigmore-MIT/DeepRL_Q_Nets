from clean_rl_utils import generate_clean_rl_env
import numpy as np
from safety.utils import clean_rl_ppo_parse_config
from tqdm import tqdm
import random
import pickle

config_file = "clean_rl/ServerAllocation/M2/M2A1_IA_AR_PPO.yaml"
#config_file = "clean_rl/MSSA_N2S1_config1.yaml"
args = clean_rl_ppo_parse_config(config_file)
env = generate_clean_rl_env(args, env_type= "Allocation", normalize = False)()

random.seed(args.seed)
np.random.seed(args.seed)
env.reset(seed = args.seed)

type = "LQ" # Longest Queue (LQ), Random Queue (RQ), Longest Connected Queue (LCQ), Max Weighted Queue (MWQ)

arrivals = 0
cap = 0
delivered = 0
rewards = 0
test_length = 100000
pbar = tqdm(range(int(test_length)))
actions = np.zeros(test_length)
observations = np.zeros((test_length, 2))

policy = pickle.load(open("../DP/M2A1_policy_table.p", "rb"))

for t in pbar:
    #action = env.action_space.sample() # JRQ
    obs = env.get_obs()
    clip_obs = np.clip(obs, 0, 10)
    state_tup = tuple(clip_obs)

    #action = policy[state_tup]
    action = 2 if obs[1] > 0 else 1
    # if env.get_obs().sum() == 0:
    #     action = 0
    # else:
    #     action = env.get_stable_action(type = type)

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
    if t > 10000000:
        debug = True
    else:
        debug = False
    step = env.step(action, debug=debug)
    actions[t] = action
    arrivals += step[4]['n_arrivals']
    delivered += step[4]['delivered']
    cap += np.array(step[4]['current_capacities'])
    observations[t] = obs
    rewards+=step[1]

average_reward = np.sum(rewards)/test_length
average_arrivals = arrivals/test_length
average_sum_arrivals = np.sum(arrivals)/test_length
average_reliability = cap/test_length
action_dist = np.bincount(actions.astype(int))/test_length
mode_action = np.argmax(np.bincount(actions.astype(int)))




