from typing import List
import torch
from torch import nn
import torch.nn.functional as F
from utils import getNegativeDocumentEmbedding, getRandomDocumentEmbedding


class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List):
        super().__init__()
        len_hidden_layers = len(hidden_dims)
        linear_layers = [nn.Linear(input_dim, hidden_dims[0]),
                         nn.ReLU()]
        for i in range(len_hidden_layers - 1):
            linear_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            linear_layers.append(nn.ReLU())
        linear_layers.append(nn.Linear(hidden_dims[-1], input_dim))
        self.mlp = nn.Sequential(*linear_layers)

    def forward(self, x):
        return self.mlp(x)




def CrossEntropySimilarityLoss(x, y, device, K=128):
    pass
    # CrossEntropy 

