import torch
import torch.nn as nn
from dlca import DLCALayer

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x.float())

class DLCAMLP(nn.Module):
    def __init__(self, input_dim=40, num_levels=10, hidden_dim=256, output_dim=10):
        super().__init__()
        self.dlca = DLCALayer(num_levels=num_levels)
        self.net = nn.Sequential(
            nn.Linear(2 * num_levels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        features = self.dlca(x.float())
        return self.net(features)

class DLCAAugmentedMLP(nn.Module):
    def __init__(self, input_dim=40, num_levels=10, hidden_dim=256, output_dim=10):
        super().__init__()
        self.dlca = DLCALayer(num_levels=num_levels)
        self.net = nn.Sequential(
            nn.Linear(input_dim + 2 * num_levels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x_f = x.float()
        features = self.dlca(x_f)
        combined = torch.cat([x_f, features], dim=1)
        return self.net(combined)
