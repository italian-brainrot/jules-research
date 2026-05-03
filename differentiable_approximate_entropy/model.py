import torch
import torch.nn as nn
import torch.nn.functional as F

class DSampEnLayer(nn.Module):
    def __init__(self, m=2, r=0.2, gamma=10.0, learnable=True):
        super().__init__()
        self.m = m
        if learnable:
            self.r = nn.Parameter(torch.tensor(r))
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.register_buffer('r', torch.tensor(r))
            self.register_buffer('gamma', torch.tensor(gamma))

    def _get_matches(self, x, m):
        # x shape: (batch, seq_len)
        batch_size, seq_len = x.shape
        if seq_len < m + 1:
            return torch.zeros(batch_size, device=x.device)

        # Extract subsequences of length m
        # windows shape: (batch, num_windows, m)
        windows = x.unfold(1, m, 1)

        # Compute pairwise distances between all windows in each batch
        # windows: (B, N, M)
        # diff: (B, N, N, M)
        diff = windows.unsqueeze(1) - windows.unsqueeze(2)
        # Using L2 distance as a differentiable proxy for Chebyshev distance
        dists = torch.norm(diff, p=2, dim=-1) # (B, N, N)

        # Soft-count matches: sigma(gamma * (r - dist))
        # Mask out self-matches (diagonal)
        mask = torch.eye(dists.shape[1], device=x.device).unsqueeze(0)

        matches = torch.sigmoid(self.gamma * (self.r - dists))
        matches = matches * (1 - mask)

        # Sum matches
        return matches.sum(dim=(1, 2))

    def forward(self, x):
        # x: (batch, channels, seq_len)
        if x.dim() == 3:
            # Process each channel and average?
            # For MNIST-1D, it's usually (B, 1, 40)
            x = x.squeeze(1)

        # B = matches for length m
        # A = matches for length m + 1
        B = self._get_matches(x, self.m)
        A = self._get_matches(x, self.m + 1)

        # SampEn = -log(A/B) = log(B) - log(A)
        # Add epsilon to avoid log(0)
        eps = 1e-8
        samp_en = torch.log(B + eps) - torch.log(A + eps)

        # Handle cases where A or B might be zero or A might be very small compared to B
        # Actually Sample Entropy is non-negative because A <= B always in discrete case.
        # Here with soft-counts, it might not be strictly A <= B, but it should be close.
        samp_en = torch.clamp(samp_en, min=0.0)

        return samp_en.unsqueeze(-1)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
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

class DSampEnAugmentedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, m=2, r=0.2, gamma=10.0):
        super().__init__()
        self.dsampen = DSampEnLayer(m=m, r=r, gamma=gamma)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x shape: (batch, input_dim)
        feat = self.dsampen(x.unsqueeze(1))
        combined = torch.cat([x, feat], dim=1)
        return self.mlp(combined)
