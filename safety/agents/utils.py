import torch
import numpy as np
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def mlp_init(input_dim: int, output_dim: int, hidden_layers: int, hidden_dim: int, act_func: str):
    layers = []
    in_dim = input_dim
    for i in range(hidden_layers):
        layers.append(layer_init(torch.nn.Linear(in_dim, hidden_dim)))
        layers.append(get_activation(act_func))
        in_dim = hidden_dim
    layers.append(layer_init(torch.nn.Linear(in_dim, output_dim), std=0.01))
    return torch.nn.Sequential(*layers)

def get_activation(act_func: str):
    if act_func == "relu":
        return torch.nn.ReLU()
    elif act_func == "tanh":
        return torch.nn.Tanh()
    elif act_func == "sigmoid":
        return torch.nn.Sigmoid()
    else:
        raise NotImplementedError