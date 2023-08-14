from safety.clean_rl_tuner import Tuner
import optuna



tuner = Tuner(
    script="Sweep_Clean_RL_IA_ARPPO.py",
    metric="window_average_backlog",
    metric_last_n_average_window=50,
    direction="minimize",
    aggregation_type="average",
    target_scores= {"N2S3":[0,10]},
    params_fn=lambda trial: {
        "num-steps": trial.suggest_categorical("num-steps", [64, 128, 256, 512]),
        "learning-rate": trial.suggest_loguniform("learning-rate", 1e-5, 1e-3),
        "norm-adv": trial.suggest_categorical("norm-adv", [True, False]),
        "reward-scale": trial.suggest_categorical("reward-scale", [0.0005, 0.001, 0.1]),
        "alpha": trial.suggest_categorical("alpha", [0.1, 0.2]),
        "nu": trial.suggest_categorical("nu", [0.0, -0.1, -0.3, -1.0]),
        # Not Swept
        "exp-name": "N4S3-sweep",
        "env-json-path": "/JSON/Environment/ServerAllocation/N4/N4S3.json",
        "track": True,
        "wandb-project-name": "N4S3-sweep",

        "total-timesteps": 1000000,
        "obs-scale": 30,
        "int-thresh": 20,

    },
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    sampler=optuna.samplers.TPESampler(),
)
tuner.tune(
    num_trials=100,
    num_seeds=1,
)