from torch import nn
from src.models.baseline_mlp import BaselineMLP


class Projector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.ReLU(),
            nn.Linear(2 * dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    def __init__(self, input_dim, hidden_dims, out_dim):
        super().__init__()
        self.backbone = BaselineMLP(input_dim, hidden_dims, out_dim=out_dim)
        self.projector = Projector(dim=out_dim)

    def forward(self, x, return_feature=False):
        features = self.backbone(x)
        z = self.projector(features)
        if return_feature:
            return features
        return z
