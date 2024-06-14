from typing import List
from torch import nn
from src.models import BaselineMLP


class BYOL(nn.Module):
    def __init__(self, input_dim, hidden_dims: List[int],  out_dim=None):
        super().__init__()
        if out_dim is not None:
            out_dim = input_dim
        self.backbone = BaselineMLP(
            input_dim=input_dim, hidden_dims=hidden_dims, out_dim=out_dim)
        self.projector = nn.Sequential(
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.projector(x)
