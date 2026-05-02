import torch
import torch.nn as nn
from higuchi import HiguchiFractalDimension

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
        return self.net(x)

class HFDNet(nn.Module):
    def __init__(self, input_dim=40, k_max=10, hidden_dim=256, output_dim=10):
        super().__init__()
        self.hfd = HiguchiFractalDimension(k_max=k_max)
        self.mlp = nn.Sequential(
            nn.Linear(k_max, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        hfd_features = self.hfd(x) # (B, k_max)
        # Handle potential NaNs or Infs if any k subseries was too short (not expected for MNIST-1D)
        hfd_features = torch.nan_to_num(hfd_features)
        # Use log scale as features because HFD is power-law
        hfd_features = torch.log(hfd_features + 1e-8)
        return self.mlp(hfd_features)

class HFDAugmentedMLP(nn.Module):
    def __init__(self, input_dim=40, k_max=10, hidden_dim=256, output_dim=10):
        super().__init__()
        self.hfd = HiguchiFractalDimension(k_max=k_max)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + k_max, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        hfd_features = self.hfd(x)
        hfd_features = torch.nan_to_num(hfd_features)
        hfd_features = torch.log(hfd_features + 1e-8)

        combined = torch.cat([x, hfd_features], dim=1)
        return self.mlp(combined)
