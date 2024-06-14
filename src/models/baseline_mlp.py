from typing import List
from torch import nn


class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List, out_dim=None):
        super().__init__()
        len_hidden_layers = len(hidden_dims)
        linear_layers = [nn.Linear(input_dim, hidden_dims[0]),
                         nn.ReLU()]
        for i in range(len_hidden_layers - 1):
            linear_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            linear_layers.append(nn.LayerNorm(hidden_dims[i + 1]))
            linear_layers.append(nn.ReLU())
            linear_layers.append(nn.Dropout(0.2))
        if out_dim is None:
            linear_layers.append(nn.Linear(hidden_dims[-1], input_dim))
        else:
            linear_layers.append(nn.Linear(hidden_dims[-1], out_dim))
        self.mlp = nn.Sequential(*linear_layers)

    def forward(self, x):
        return self.mlp(x)
