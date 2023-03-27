import os
import numpy as np
import torch
import wandb
import pandas as pd
pd.options.plotting.backend = "plotly"
import plotly.graph_objects as go
import logging
logging.getLogger().setLevel(logging.INFO)

from utils import get_stats

class CheckpointSaver:
    def __init__(self, dirpath, env_string, algo_string, decreasing=False, top_n=5):
        """
        dirpath: Directory path where to store all model weights
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value

        Models should be saved to Saved_Models\\{env_para['name']}\\{training_args.name}\\
        And have the file_name '{self.env_string}_{self.algo_string}_s{metric_val}.pt'

        Only 5 total models should be saved for each environment/training args combination

        Each artifact should include the metric score and epoch in the metadata

        """
        if not os.path.exists(dirpath): os.makedirs(dirpath)
        self.dirpath = dirpath
        self.top_n = top_n
        self.decreasing = decreasing
        self.top_model_paths = {}
        self.best_metric_vals = -np.ones(top_n)*np.inf
        self.env_string = env_string
        self.algo_string = algo_string
        self.get_best_metrics()
        self.cleanup()

    def get_best_metrics(self):
        if os.listdir(self.dirpath):
            for file_name in os.listdir(self.dirpath):
                if file_name.endswith('pt') and "manual" not in file_name:
                    file_path = os.path.join(self.dirpath,file_name)
                    score_str = file_name.split("score")[1].replace(".pt","")
                    score = float(score_str)
                    self.top_model_paths[file_path] = score
            self.sort_best_model_dict()
        else:
            self.top_model_paths = {}

    def __call__(self, model, epoch, metric_val):
        model_path = os.path.join(self.dirpath, model.__class__.__name__ + f'_epoch{epoch}_score{metric_val:.2e}.pt')
        info_string = None
        if self.best_metric_vals.shape[0] < self.top_n:
            save = True
            info_str =  f"Less than {self.top_n} models saved for the Environenment/Training parameters, saving model at {model_path}, & logging model weights to W&B."
        elif self.decreasing:
            n_th_best = np.min(self.best_metric_vals)
            save = metric_val< n_th_best
            info_str = f"Current metric value better than {metric_val} better than best {n_th_best}, saving model at {model_path}, & logging model weights to W&B."
        else:
            n_th_best = np.min(self.best_metric_vals)
            save = metric_val > n_th_best
            info_str = f"Current metric value better than {metric_val} better than best {n_th_best}, saving model at {model_path}, & logging model weights to W&B."

        if save:
            #logging.info(info_str)
            torch.save(model.state_dict(), model_path)
            self.log_artifact(f'{self.env_string}_{self.algo_string}.pt', model_path, metric_val, epoch)
            self.top_model_paths[model_path] = metric_val
            self.sort_best_model_dict()
        if len(self.top_model_paths.keys()) > self.top_n:
            self.cleanup()
        return self.best_metric_vals, info_str

    def log_artifact(self, filename, model_path, metric_val, epoch):
        artifact = wandb.Artifact(filename, type='model', metadata={'Validation score': metric_val, 'Epoch': epoch})
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact)

    def sort_best_model_dict(self):
        self.top_model_paths = dict(sorted(self.top_model_paths.items(), key=lambda x:x[1], reverse = not self.decreasing))
        self.best_metric_vals = np.array([x for x in self.top_model_paths.values()])

    def cleanup(self):
        all_model_paths = list(tuple(self.top_model_paths))
        to_remove = all_model_paths[self.top_n:]
        #logging.info(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            os.remove(o)
            del self.top_model_paths[o]
        #self.top_model_paths = self.top_model_paths[:self.top_n]


def wandb_plot_rewards_vs_time(rewards_vs_time, policy_name):
    # NOT FINISHED NOR USED
    df = pd.DataFrame(rewards_vs_time, columns = [f"Env {e}" for e in range(rewards_vs_time.shape[1])])
    fig = df.plot(title = "Test Rewards vs Time", labels=dict(index="Time", value="Rewards"))
    # fig.show()
    wandb.log({"test_rewards_vs_time": fig})

    df["LTA_Rewards"], df["LTA_Error"] = get_stats(rewards_vs_time.T)


    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x = df.index,
        y = df["LTA_Rewards"],
        name = "LTA Rewards",
        line=dict(color='rgb(31, 119, 180)')
    ))
    fig2.add_trace(go.Scatter(
        name = "Upper Error",
        x = df.index,
        y = df["LTA_Rewards"]+df["LTA_Error"],
        mode = "lines",
        line = dict(width = 0),
        showlegend= False
    ))
    fig2.add_trace(go.Scatter(
        name="Lower Error",
        x=df.index,
        y=df["LTA_Rewards"] - df["LTA_Error"],
        mode="lines",
        marker=dict(color="#444"),
        line=dict(width=0),
        fillcolor= 'rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False
    )) # TODO: Figure out why error shading is not showing in weights and biases

    fig2.update_layout(
        title="Long-Time Average Rewards vs Time",
        xaxis_title="Time",
        yaxis_title="LTA")
    #fig2.show()
    wandb.log({"LTA_vs_Time": fig2})
    wandb.define_metric("test/step")
    wandb.define_metric("test/LTA", summary = "mean", step = "test/step")

    for i in range(df.shape[0]):
        log_dict = {
            "test/step": i,
            "test/LTA": df["LTA_Rewards"][i],
        }
        wandb.log(log_dict)


    table = wandb.Table(dataframe = df) # columns = [f"Env {e}" for e in range(data.shape[0])])
    wandb.log({"test_rewards_table": table})
    return df


def wandb_test_qs_vs_time(test_history, merge= True):
    q_dfs = []
    for key, value in test_history.items():
        if key is 'Env_seeds':
            continue
        else:
            df = value
            q_cols = [x for x in df.columns if "Q" in x]
            qi_df = df.loc[:, q_cols]
            q_dfs.append(qi_df)
    if merge:
        q_df = pd.concat(q_dfs).groupby(level=0, axis='columns').mean()
        fig = q_df.plot()
        #fig.show()
        wandb.log({"Average_Qs":fig})
        return q_dfs, q_df
    else:
        return q_dfs, None