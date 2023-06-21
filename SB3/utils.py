import pandas as pd
from stable_baselines3 import PPO, TD3
from stable_baselines3.ppo import MlpPolicy
from NonDRLPolicies.Backpressure import MCMHBackPressurePolicy
from stable_baselines3.common.callbacks import BaseCallback
from wandb.integration.sb3 import WandbCallback
import wandb
import os
from copy import deepcopy
import numpy as np
import plotly.graph_objects as go



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
    elif ac.policy_name == "TD3":
        agent = TD3(
            policy= ac.policy,
            env=env,
            **ac.kwargs.toDict(),
        )
    else:
        raise NotImplementedError
    return agent

def load_agent(config, env):
    if config.agent.policy_name == "PPO":
        agent = PPO.load(config.load.zip_path)
    if config.agent.policy_name == "Backpressure":
        agent = MCMHBackPressurePolicy(env, M = config.agent.modified)
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
        self.num_rollouts =0

        if model_save_path is not None:
            os.makedirs(model_save_path, exist_ok=True)
            self.model_save_path = model_save_path
            self.best_lta_backlog = -float('inf')

    def old_on_rollout_end(self) -> bool:
        # Log additional tensor
        self.num_rollouts +=1
        lta_backlog = self.sum_backlog / self.rollout_steps
        wandb.log({"rollout/lta_backlog": lta_backlog, "rollout/num_rollouts": self.num_rollouts})
        if self.model_save_path is not None:
            if self.best_lta_backlog < lta_backlog:
                self.save_model()

        # Reset values
        self.sum_backlog = 0
        self.rollout_steps = 0
        return True

    def reset_log(self):
        self.num_rollouts += 1
        lta_backlog = self.sum_backlog / self.rollout_steps
        wandb.log({"rollout/lta_backlog": lta_backlog, "rollout/num_rollouts": self.num_rollouts})
        if self.model_save_path is not None:
            if self.best_lta_backlog < lta_backlog:
                self.save_model()

        # Reset values
        self.sum_backlog = 0
        self.rollout_steps = 0

    def _on_step(self) -> bool:
        # log information
        self.sum_backlog += sum(self.locals["infos"][i]["backlog"] for i in range(len(self.locals["infos"])))/ len(self.locals["infos"])
        self.rollout_steps += 1
        # check to see if the trajectory is over
        if self.locals["dones"].any():
            self.reset_log()

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
        self.logger[self.n].append(deepcopy(self.sum_backlog / self.rollout_steps)[0])

        if locals["done"] or locals["info"]["TimeLimit.truncated"]:
            self.next_env()

    def write_log(self):
        data_list = list(self.logger.values())
        data_array = np.array(data_list[:-1])
        df = pd.DataFrame(data_array.T, columns = list(range(data_array.shape[0])))
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(y=df.mean(axis=1), mode='lines', name='mean'))
        fig1.update_xaxes(title_text="Time Step")
        fig1.update_yaxes(title_text="Backlog")
        wandb.log({"test/Mean Backlog vs Time": fig1})

        # plot the final LTA backlog as box and whisker plot
        fig2 = go.Figure()
        fig2.add_trace(go.Box(y=df.iloc[-1,:], name='Final LTA'))
        fig2.update_xaxes(title_text="Environment")
        fig2.update_yaxes(title_text="Backlog")
        wandb.log({"test/Final LTA Backlog": fig2})

        # plot all lta backlogs vs time as line plot
        fig3 = go.Figure()
        for i in range(data_array.shape[0]):
            fig3.add_trace(go.Scatter(y=df.iloc[:,i], mode='lines', name='env {}'.format(i)))
        fig3.update_xaxes(title_text="Time Step")
        fig3.update_yaxes(title_text="Backlog")
        wandb.log({"test/All Backlogs vs Time": fig3})






