from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from NonDRLPolicies.Backpressure import MCMHBackPressurePolicy
from stable_baselines3.common.callbacks import BaseCallback
from wandb.integration.sb3 import WandbCallback
import wandb
import os
from copy import deepcopy



def generate_agent(config, env, ):
    """
    Generates the agent
    """
    ac = config.agent # ac = agent_config
    if ac.policy_name == "PPO":
        agent = PPO(
            policy= ac.policy,
            env=env,
            **ac.kwargs.toDict(),
        )
    elif ac.policy_name == "Backpressure":
        agent = MCMHBackPressurePolicy(env, M = ac.modified)
    else:
        raise NotImplementedError
    return agent

class TrainingWandbCallback(BaseCallback):
    """
    Custom callback for plotting additional values in wandb.
    """
    def __init__(self, verbose=0, model_save_path = None):
        super().__init__(verbose)
        if wandb.run is None:
            raise wandb.Error("You must call wandb.init() before CustomWandbCallback()")
        self.sum_backlog = 0
        self.rollout_steps = 0

        if model_save_path is not None:
            os.makedirs(model_save_path, exist_ok=True)
            self.model_save_path = model_save_path
            self.best_lta_backlog = -float('inf')

    def _on_rollout_end(self) -> bool:
        # Log additional tensor
        lta_backlog = self.sum_backlog / self.rollout_steps
        wandb.log({"rollout/lta_backlog": lta_backlog})
        if self.model_save_path is not None:
            if self.best_lta_backlog < lta_backlog:
                self.save_model()

        # Reset values
        self.sum_backlog = 0
        self.rollout_steps = 0
        return True

    def _on_step(self) -> bool:
        self.sum_backlog += sum(self.locals["infos"][i]["backlog"] for i in range(len(self.locals["infos"])))/ len(self.locals["infos"])
        self.rollout_steps += 1
        return True


    def save_model(self) -> None:
        self.model.save(self.model_save_path + "/best_model.zip")
        wandb.save(self.model_save_path + "/best_model.zip", base_path = self.model_save_path)


class CustomEvalCallback(BaseCallback):
    def __init__(self, verbose=0, best_model_save_path = None):
        super().__init__(verbose)
        self.best_model_save_path = best_model_save_path
        best_final_reward = -float('inf')

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # get the final reward
        final_reward = self.locals["infos"][0]["episode"]["r"]
        if final_reward > self.best_final_reward:
            self.best_final_reward = final_reward
            # save the new best model



class EvalWandbLogger(BaseCallback):
    """
    Custom callback for plotting additional values in wandb while evaluating an agent
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        if wandb.run is None:
            raise wandb.Error("You must call wandb.init() before CustomWandbCallback()")
        self.sum_backlog = 0
        self.rollout_steps = 0
        self.n = 0
        self.logger = {self.n: []}

    def next_env(self):
        self.sum_backlog= 0
        self.rollout_steps = 0
        self.n +=1
        self.logger[self.n] = []

    def _on_step(self, locals, globals) -> bool:
        self.sum_backlog += sum(locals["infos"][i]["backlog"] for i in range(len(locals["infos"])))/ len(locals["infos"])
        self.rollout_steps += 1
        self.logger[self.n].append(deepcopy(self.sum_backlog / self.rollout_steps))

        if locals["done"]:
            self.next_env()

    def write_log(self):
        all_data = []
        keys = []
        for n,data in self.logger.items():
            all_data.append(data)
            keys.append("env_" + str(n))
        x = list(range(len(all_data[0])))
        wandb.log({"eval/lta_backlog": wandb.plot_line_series(xs = x, ys = all_data,
                                                                keys = keys, title = "Test LTA Backlog")})



