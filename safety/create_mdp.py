from clean_rl_utils import generate_clean_rl_env
import numpy as np
from safety.utils import clean_rl_ppo_parse_config
from Environments.ServerAllocation import ServerAllocationMDP
import random

config_file = "clean_rl/ServerAllocation/M2/M2A2-O_IA_AR_PPO.yaml"
args = clean_rl_ppo_parse_config(config_file)
env = generate_clean_rl_env(args, env_type= "ServerAllocation", normalize = False)()
random.seed(args.seed)
np.random.seed(args.seed)
env.reset(seed = args.seed)


# MDP Settings
q_max = 15
max_samples = 1000
max_vi_iterations = 50

# Create MDP
mdp = ServerAllocationMDP(env, name = "M2A2_O_MDP", q_max = q_max)
mdp.estimate_tx_matrix(env, max_samples = max_samples)
mdp.do_VI(max_iterations=max_vi_iterations)
#mdp.load_tx_matrix('saved_mdps/M2A2_O_MDP_qmax5_discount0.99_max_samples-100_tx_matrix.pkl')
mdp.save_MDP(f"saved_MDPs/M2A2_O_MDP.p")




