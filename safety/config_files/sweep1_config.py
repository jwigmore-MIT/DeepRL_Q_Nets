sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize',
        'name': 'rollout_summary/mean_reward'
        },
    'parameters': {
        'actor_learning_rate': {'values', [3e-4, 1e-4, 3e-5, 5e-3]},
        'std_learning_rate': {'values': [3e-4, 1e-4, 3e-5, 5e-3]},
        'batch_size': {'values': [64, 128, 256, 512]},
        'num_episodes': {'values': [1000, 2000]},
        'imit_coef': {'values': [0.0, 0.2, 1.0, 3]},
        'int_coef': {'values': [0.0, 1, 5, 10]},
        'grad_clip': {'values': [None, 0.5, 1.0, 5, 10, 50]},
        'norm_factor': {"values": [20, 40, 80, 160]},
     }
}