import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DCDLayer(nn.Module):
    """
    Differentiable Correlation Dimension Layer.
    Approximates the correlation dimension D2 by computing a soft correlation integral
    and performing a differentiable linear regression in log-log space.
    """
    def __init__(self, m=3, tau=1, r_min=0.1, r_max=0.9, num_r=8, gamma=10.0, learnable_params=True):
        super().__init__()
        self.m = m
        self.tau = tau
        self.num_r = num_r

        # r values to sample the correlation integral
        r_vals = torch.linspace(r_min, r_max, num_r)
        self.register_buffer('r_vals', r_vals)
        self.register_buffer('log_r', torch.log(r_vals))

        if learnable_params:
            self.gamma = nn.Parameter(torch.tensor(gamma))
            # We can also make tau learnable, but it requires interpolation.
            # For now, let's keep it fixed or use soft-selection if needed.
            # But let's start simple.
        else:
            self.register_buffer('gamma', torch.tensor(gamma))

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, L)
        Returns:
            D2: Correlation dimension estimate (B, 1)
            C_r: Correlation integral values (B, num_r)
        """
        B, L = x.shape

        # 1. Phase space reconstruction (Delay Embedding)
        # We'll use a fixed tau for now.
        # indices for embedding: [0, tau, ..., (m-1)*tau]
        # We can use unfold.
        window_size = (self.m - 1) * self.tau + 1
        if L < window_size:
            # Fallback if L is too small
            x_emb = x.unsqueeze(-1).expand(-1, -1, self.m)
        else:
            # x: (B, L) -> (B, 1, L)
            # unfolded: (B, window_size, num_windows)
            unfolded = x.unsqueeze(1).unfold(2, window_size, 1) # (B, 1, num_windows, window_size)
            unfolded = unfolded.squeeze(1) # (B, num_windows, window_size)

            indices = torch.arange(0, window_size, self.tau, device=x.device)
            x_emb = unfolded[:, :, indices] # (B, N, m) where N = L - window_size + 1

        # 2. Pairwise distances
        # x_emb: (B, N, m)
        dist_sq = torch.cdist(x_emb, x_emb, p=2) # (B, N, N)

        # 3. Soft Correlation Integral C(r)
        # C(r) = (2 / (N*(N-1))) * sum_{i < j} sigmoid(gamma * (r - dist_ij))
        N = x_emb.shape[1]
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1)

        # r_vals: (num_r)
        # dist_sq: (B, N, N)
        # We want (B, num_r)

        # Expand for broadcasting
        # dist_sq: (B, 1, N, N)
        # r_vals: (1, num_r, 1, 1)
        r_expanded = self.r_vals.view(1, self.num_r, 1, 1)
        dists = dist_sq.unsqueeze(1)

        # Soft step: sigmoid(gamma * (r - dist))
        soft_step = torch.sigmoid(self.gamma * (r_expanded - dists))

        # Apply mask and sum
        C_r = (soft_step * mask).sum(dim=(2, 3)) * (2.0 / (N * (N - 1) + 1e-8))

        # 4. Differentiable Linear Regression to find D2
        # log(C_r) = D2 * log(r) + constant
        # We use a small epsilon to avoid log(0)
        log_C = torch.log(C_r + 1e-8)

        # Simple least squares for slope:
        # slope = cov(x, y) / var(x)
        # Here x = log_r, y = log_C

        log_r = self.log_r # (num_r)
        mean_r = log_r.mean()
        mean_C = log_C.mean(dim=1, keepdim=True)

        num = ((log_r - mean_r) * (log_C - mean_C)).sum(dim=1, keepdim=True)
        den = ((log_r - mean_r)**2).sum()

        D2 = num / (den + 1e-8)

        return D2, C_r

class DCDAugmentedMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10, m=3, tau=1):
        super().__init__()
        self.dcd = DCDLayer(m=m, tau=tau)
        # DCD returns D2 (1) and C_r (num_r=8)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + 1 + 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        d2, cr = self.dcd(x)
        combined = torch.cat([x, d2, cr], dim=1)
        return self.mlp(combined)

class DCDMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10, m=3, tau=1):
        super().__init__()
        self.dcd = DCDLayer(m=m, tau=tau)
        self.mlp = nn.Sequential(
            nn.Linear(1 + 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        d2, cr = self.dcd(x)
        features = torch.cat([d2, cr], dim=1)
        return self.mlp(features)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.mlp(x)
