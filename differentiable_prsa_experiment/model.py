import torch
import torch.nn as nn
try:
    from .layer import DPRSALayer
except ImportError:
    from layer import DPRSALayer

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        return self.net(x)

class DPRSANet(nn.Module):
    def __init__(self, in_channels=1, num_anchors=8, window_size=20, hidden_dim=256, output_dim=10):
        super().__init__()
        self.prsa = DPRSALayer(in_channels, num_anchors, window_size)
        prsa_out_dim = num_anchors * in_channels * window_size
        self.mlp = nn.Sequential(
            nn.Linear(prsa_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (batch, 1, 40)
        x = self.prsa(x)
        return self.mlp(x)

class DPRSAAugmentedMLP(nn.Module):
    def __init__(self, input_dim=40, in_channels=1, num_anchors=4, window_size=20, hidden_dim=256, output_dim=10):
        super().__init__()
        self.prsa = DPRSALayer(in_channels, num_anchors, window_size)
        prsa_out_dim = num_anchors * in_channels * window_size
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + prsa_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (batch, 1, 40)
        feat_prsa = self.prsa(x)
        feat_raw = x.view(x.size(0), -1)
        combined = torch.cat([feat_raw, feat_prsa], dim=1)
        return self.mlp(combined)
