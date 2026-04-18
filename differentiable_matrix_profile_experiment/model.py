import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableMatrixProfile(nn.Module):
    def __init__(self, window_size=5, temperature=0.1, learn_temperature=True):
        super().__init__()
        self.window_size = window_size
        if learn_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        m = self.window_size
        num_windows = seq_len - m + 1

        # Extract windows: (batch_size, num_windows, m)
        windows = x.unfold(1, m, 1)

        # Z-normalize windows
        # Calculate mean and std for each window
        means = windows.mean(dim=-1, keepdim=True)
        stds = windows.std(dim=-1, keepdim=True) + 1e-8
        z_windows = (windows - means) / stds

        # Compute pairwise distances between z-normalized windows
        # (batch_size, num_windows, 1, m) - (batch_size, 1, num_windows, m)
        # Using the expansion: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        # Since windows are z-normalized, ||a||^2 = m (roughly, if using biased std)
        # Actually, let's just do it directly for clarity and to avoid issues with std definition

        dist_sq = torch.sum((z_windows.unsqueeze(2) - z_windows.unsqueeze(1))**2, dim=-1)
        dist = torch.sqrt(dist_sq + 1e-8)

        # Mask out diagonal (self-match)
        mask = torch.eye(num_windows, device=x.device).unsqueeze(0) * 1e9

        # Also mask out trivial matches (windows that overlap significantly)
        # Standard in Matrix Profile is to mask out windows within m/2 of each other
        exclusion_zone = m // 2
        for i in range(-exclusion_zone, exclusion_zone + 1):
            if i == 0: continue
            diag_mask = torch.diag(torch.ones(num_windows - abs(i), device=x.device), diagonal=i).unsqueeze(0) * 1e9
            mask = mask + diag_mask

        dist = dist + mask

        # Softmin across the j dimension (nearest neighbor)
        tau = torch.clamp(self.temperature, min=1e-3)
        mp = -tau * torch.logsumexp(-dist / tau, dim=2)

        return mp

class MPAugmentedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, window_size=5):
        super().__init__()
        self.mp_layer = DifferentiableMatrixProfile(window_size=window_size)
        mp_output_dim = input_dim - window_size + 1
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + mp_output_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        mp_features = self.mp_layer(x)
        combined = torch.cat([x, mp_features], dim=1)
        return self.mlp(combined)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)
