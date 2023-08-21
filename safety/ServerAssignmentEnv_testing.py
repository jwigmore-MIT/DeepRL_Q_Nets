import numpy as np
from safety.utils import clean_rl_ppo_parse_config
from clean_rl_utils import generate_clean_rl_env
from tqdm import tqdm

config_file = "clean_rl/N4S3/N4S3_IA_ARPPO1.yaml"
#config_file = "clean_rl/MSSA_N2S1_config1.yaml"
args = clean_rl_ppo_parse_config(config_file)
env = generate_clean_rl_env(args, env_type="ServerAssignment", normalize = False)()


arrivals = 0
delivered = 0
rewards = 0
test_length = 100
pbar = tqdm(range(int(test_length)))
actions = np.zeros(test_length)
observations = np.zeros((test_length, env.observation_space.shape[0]))
masks = np.zeros((test_length, env.get_mask().shape[0]))
for t in pbar:
    #action = env.action_space.sample() # JRQ
    mask = env.get_mask()
    action = env.get_stable_action(type = "JSQ") # JSQ

    step = env.step(action, debug=True)
    actions[t] = action
    arrivals += step[4]['n_arrivals']
    delivered += step[4]['delivered']
    observations[t] = env.get_obs()
    masks[t] = mask
    rewards+=step[1]

average_reward = np.sum(rewards)/test_length






