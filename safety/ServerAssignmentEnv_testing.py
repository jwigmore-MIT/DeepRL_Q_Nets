from Environments.ServerAssignment import ServerAssignment, generate_clean_rl_env
import numpy as np
from safety.utils import clean_rl_ppo_parse_config
from tqdm import tqdm

config_file = "clean_rl/N4S3_PPO1.yaml"
#config_file = "clean_rl/MSSA_N2S1_config1.yaml"
args = clean_rl_ppo_parse_config(config_file)
env = generate_clean_rl_env(args, normalize = False)()


arrivals = 0
delivered = 0
rewards = 0
test_length = 1e6
pbar = tqdm(range(int(test_length)))
for t in pbar:
    #action = env.action_space.sample() # JRQ

    action = np.argmin(list(env.buffers.values())[1:-1]) # JSQ

    step = env.step(action, debug=False)
    arrivals += step[4]['n_arrivals']
    delivered += step[4]['delivered']
    rewards+=step[1]

average_reward = np.sum(rewards)/test_length






