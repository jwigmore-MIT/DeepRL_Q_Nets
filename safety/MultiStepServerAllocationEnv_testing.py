from Environments.ServerAllocation import ServerAllocation, generate_clean_rl_env, generate_clean_rl_MSSA_env
from safety.utils import clean_rl_ppo_parse_config
import numpy as np

#config_file = "clean_rl/N2S2_config1.yaml"
config_file = "clean_rl/MSSA_N2S1_config1.yaml"
args = clean_rl_ppo_parse_config(config_file)
env = generate_clean_rl_MSSA_env(args)()


arrivals = 0
delivered = 0
rewards = 0
test_steps = 100000
total_time = 0
for t in range(test_steps):
    # join a random server
    action = env.action_space.sample()
    # join the shortest queue
    #action = np.argmin(list(env.buffers.values())[1:-1])
    step = env.step(action, debug=False)
    arrivals += step[4]['n_arrivals']
    delivered += step[4]['delivered']
    total_time += step[4]['interarrival_time']
    rewards += step[1]

time_average_backlog = rewards/total_time




