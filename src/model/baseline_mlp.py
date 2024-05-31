from typing import List
import torch
from torch import nn
import torch.nn.functional as F
from utils import getRandomDocumentEmbedding


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


def CosineSimilarityLoss(x, y, device, K=64, documents=None):
    sampledDocumentEmbedding = getRandomDocumentEmbedding(documents=documents, device=device, K=K)
    return (1 - F.cosine_similarity(x.unsqueeze(dim=0), y, dim=-1)).sum() / y.size(0) + \
        (F.cosine_similarity(x.unsqueeze(dim=0), sampledDocumentEmbedding)).sum() / K

        