
import json

class Args(object):
    pass

def parse_json(json_path: str):

    para = json.load(open(json_path))
    args = Args()
    test_args = Args()
    env_para = para["problem_instance"]
    algo_para = para["algo_parameters"]
    test_para = para["test_parameters"]

    for key, value in algo_para.items():
        setattr(args, key, value)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    for key, value in test_para.items():
        setattr(test_args, key, value)
    return env_para, args, test_args

def parse_training_json(json_path):

    para = json.load(open(json_path))
    args = Args()
    algo_para = para["algo_parameters"]
    for key, value in algo_para.items():
        setattr(args, key, value)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_resets = args.total_timesteps /args.reset_steps # total number of new trajectories seen
    return args

def parse_env_json(json_path):
    para = json.load(open(json_path))
    env_para = para["problem_instance"]
    return env_para

def parse_test_json(json_path):
    para = json.load(open(json_path))
    test_args = Args()
    test_para = para["test_parameters"]
    for key, value in test_para.items():
        setattr(test_args, key, value)
    return test_args