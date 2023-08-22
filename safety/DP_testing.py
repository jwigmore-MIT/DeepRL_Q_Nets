from safety.utils import clean_rl_ppo_parse_config
from safety.clean_rl_utils import clean_rl_ppo_parse_config, generate_clean_rl_env
import numpy as np
from Environments.ServerAllocation import ServerAllocationMDP
import pickle




max_samples = 1000
min_samples = 200
theta = 0.001
config_file = "clean_rl/ServerAllocation/M2/M2A2-O_IA_AR_PPO.yaml"
args = clean_rl_ppo_parse_config(config_file)
env = generate_clean_rl_env(args, env_type= "ServerAllocation", normalize = False)()

mdp = ServerAllocationMDP(env, q_max = 15, discount = 0.99)
mdp.estimate_tx_matrix(max_samples = max_samples, min_samples = min_samples, theta = theta)
# q_max = 10
# high_queues = np.ones(env.n_queues)*q_max
# high_links = np.array([rv.max() for rv in env.capacities_fcn])
# high = np.concatenate((high_queues, high_links))
# low = np.zeros_like(high)
# # state_list = create_state_map(low, high)
# #state_list = [[1,1,0,1]]
# action_list = list(range(0, env.n_queues+1))
# tx_matrix, n_samples = form_transition_matrix(env, state_list, action_list, max_samples = max_samples, min_samples = min_samples, theta = 0.001)
# pickle.dump(tx_matrix, open(f"M2A2-O_tx_matrix_{max_samples}.pkl", 'wb'))

# tx_matrix = form_transition_matrix(env, state_list, action_list, num_samples = 10000)
# tx_matrix2 = form_transition_matrix(env, state_list, action_list, num_samples = 5000)
# diff = {}
# for sa in tx_matrix.keys():
#     for sp in tx_matrix[sa].keys():
#         try:
#             diff[sp] = np.abs(tx_matrix[sa][sp] - tx_matrix2[sa][sp])
#         except KeyError:
#             diff[sp] = 1
#
# total_error = np.sum(list(diff.values()))