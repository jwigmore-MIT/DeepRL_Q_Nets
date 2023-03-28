
import json

class Args(object):
    pass



def parse_jsons(training_json_path, env_json_path, test_json_path):
    config_args = Args()
    training_args = parse_training_json(training_json_path, config_args)
    env_para = parse_env_json(env_json_path, config_args)
    test_args = parse_test_json(test_json_path, config_args)
    return config_args, training_args, env_para, test_args

def parse_training_json(json_path, config_args):

    para = json.load(open(json_path))
    args = Args()
    algo_para = para["algo_parameters"]
    for key, value in algo_para.items():
        setattr(args, key, value)
        setattr(config_args, f"train/{key}", value)
    args.num_steps = int(args.num_steps_per_env*args.num_envs)
    args.reset_steps = int(args.reset_steps_per_env*args.num_envs)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_resets = args.total_timesteps /args.reset_steps # total number of new trajectories seen
    return args

def parse_env_json(json_path, config_args):
    para = json.load(open(json_path))
    env_para = para["problem_instance"]
    for key, value in env_para.items():
        setattr(config_args, f"env/{key}", value)
    return env_para

def parse_test_json(json_path, config_args):
    para = json.load(open(json_path))
    test_args = Args()
    test_para = para["test_parameters"]
    for key, value in test_para.items():
        setattr(test_args, key, value)
        setattr(config_args, f"test/{key}", value)
    return test_args