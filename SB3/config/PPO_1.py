from SB3.config.base import *


@dataclass
class PPOkwargs:
    learning_rate: float = field(default= 3e-4),
    n_steps = 2048,
    batch_size = 64,
    n_epochs = 10,
    gamma = 0.99,
    gae_lambda = 0.95,
    clip_range = 0.2,
    clip_range_vf = None,
    normalize_advantage = True,
    ent_coef = 0.0,
    vf_coef = 0.5,
    max_grad_norm = 0.5,
    use_sde = False,
    sde_sample_freq = -1,
    target_kl = None,
    stats_window_size = 100,
    tensorboard_log = None,
    policy_kwargs = None,
    verbose = 0,
    seed = None,
    device = "auto",
    _init_setup_model = True

    def asdict(self):
        return asdict(self)


@dataclass
class PPOAgentConfig:
    policy_name: str = "PPO" # PPO, TD3, etc
    policy: Union[str, Type[ActorCriticPolicy]] = "MlpPolicy"
    policy_kwargs: PPOkwargs = field(default_factory=PPOkwargs)


@dataclass
class TrainingConfig:
    '''
    total_timesteps – The total number of samples (env steps) to train on
    callback – callback(s) called at every step with state of the algorithm.
    log_interval – The number of episodes before logging.
    tb_log_name – the name of the run for TensorBoard logging
    reset_num_timesteps – whether or not to reset the current timestep number (used in logging)
    progress_bar – Display a progress bar using tqdm and rich.
    '''
    total_timesteps: int = 100000
    callback = None
    log_interval: int = 1




# Evironment
config = Config()
config.env.env_json_path = project_root + "/JSON/Environment/Env1/Env1a.json"
config.agent = PPOAgentConfig()