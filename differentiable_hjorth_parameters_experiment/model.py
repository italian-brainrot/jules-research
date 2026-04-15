import torch
import torch.nn as nn
import torch.nn.functional as F

class HjorthLayer(nn.Module):
    """
    Computes Hjorth parameters: Activity, Mobility, and Complexity.
    Input shape: (batch, length) or (batch, channels, length)
    Output shape: (batch, channels * 3)
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Activity: Variance of the signal
        activity = torch.var(x, dim=-1)

        # First derivative
        dx = torch.diff(x, dim=-1)
        var_dx = torch.var(dx, dim=-1)

        # Mobility: Square root of variance of first derivative divided by variance of signal
        mobility = torch.sqrt(var_dx / (activity + self.eps))

        # Second derivative
        ddx = torch.diff(dx, dim=-1)
        var_ddx = torch.var(ddx, dim=-1)

        # Complexity: Mobility of first derivative divided by mobility of signal
        mobility_dx = torch.sqrt(var_ddx / (var_dx + self.eps))
        complexity = mobility_dx / (mobility + self.eps)

        # Concatenate features: (batch, channels, 3)
        features = torch.stack([activity, mobility, complexity], dim=-1)
        return features.flatten(1)

class HjorthMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hjorth = HjorthLayer()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = x.to(torch.float32)
        features = self.hjorth(x)
        return self.mlp(features)

class HjorthAugmentedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hjorth = HjorthLayer()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = x.to(torch.float32)
        features = self.hjorth(x)
        combined = torch.cat([x, features], dim=-1)
        return self.mlp(combined)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = x.to(torch.float32)
        return self.mlp(x)
