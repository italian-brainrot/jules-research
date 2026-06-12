import torch
import torch.nn as nn
import torch.nn.functional as F

class DDELayer(nn.Module):
    """
    Differentiable Dispersion Entropy (DDE) Layer.
    Computes dispersion entropy of 1D signals using soft-binning.
    """
    def __init__(self, c=3, m=2, tau=1, sigma=0.1, learnable=True):
        super().__init__()
        self.c = c
        self.m = m
        self.tau = tau

        # Centers for soft-binning (initialized uniformly between -1 and 1)
        centers = torch.linspace(-1.0, 1.0, c)
        if learnable:
            self.centers = nn.Parameter(centers)
            self.sigma = nn.Parameter(torch.tensor(sigma))
        else:
            self.register_buffer('centers', centers)
            self.register_buffer('sigma', torch.tensor(sigma))

    def forward(self, x):
        """
        x: (B, L) or (B, 1, L)
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        B, L = x.shape

        # 1. Soft-binning
        # x: (B, L) -> (B, L, 1)
        # centers: (c) -> (1, 1, c)
        # dists: (B, L, c)
        dists = (x.unsqueeze(-1) - self.centers.view(1, 1, -1))**2

        # Softmax to get membership weights
        # w: (B, L, c)
        w = F.softmax(-dists / (2 * self.sigma**2 + 1e-8), dim=-1)

        # 2. Embedding
        # We need to compute pattern weights for windows of size m with delay tau
        # num_windows = L - (m - 1) * tau
        window_indices = torch.arange(0, self.m * self.tau, self.tau, device=x.device)

        # Use unfold to get windows of membership weights
        # w: (B, L, c) -> (B, c, L) for unfold
        w_p = w.permute(0, 2, 1)

        # We want to pick m points with delay tau.
        # Unfold gives sliding windows of size (m-1)*tau + 1
        span = (self.m - 1) * self.tau + 1
        # unfolded: (B, c * span, num_windows)
        unfolded = F.unfold(w_p.unsqueeze(-1), (span, 1))
        num_windows = unfolded.shape[-1]

        # unfolded: (B, c, span, num_windows)
        unfolded = unfolded.view(B, self.c, span, num_windows)

        # selected: (B, c, m, num_windows)
        selected = unfolded[:, :, window_indices, :]

        # 3. Pattern probabilities
        # A pattern is defined by a choice of one class for each of the m positions.
        # There are c^m patterns.
        # We can compute the joint probability by taking the product of memberships.

        # selected: (B, c, m, num_windows) -> (B, num_windows, m, c)
        selected = selected.permute(0, 3, 2, 1)

        # We want to compute the product across m for all combinations of c.
        # This can be done efficiently using log-sum-exp trick or just product if small.
        # Let's use a recursive approach or broadcasted product to generate all c^m patterns.

        # Initialize pattern_weights with first position's memberships
        # pattern_weights: (B, num_windows, c)
        pattern_weights = selected[:, :, 0, :]

        for i in range(1, self.m):
            # pattern_weights: (B, num_windows, c^i)
            # next_w: (B, num_windows, c)
            next_w = selected[:, :, i, :]
            # Combine: (B, num_windows, c^i, 1) * (B, num_windows, 1, c) -> (B, num_windows, c^i, c)
            pattern_weights = pattern_weights.unsqueeze(-1) * next_w.unsqueeze(-2)
            # Flatten last two: (B, num_windows, c^{i+1})
            pattern_weights = pattern_weights.reshape(B, num_windows, -1)

        # pattern_weights: (B, num_windows, c^m)
        # Average over windows to get pattern distribution
        # p: (B, c^m)
        p = pattern_weights.mean(dim=1)

        # 4. Shannon Entropy
        # H = -sum(p * log(p))
        eps = 1e-9
        entropy = -torch.sum(p * torch.log(p + eps), dim=-1)

        # Normalize by log(c^m) so it's between 0 and 1
        max_entropy = torch.log(torch.tensor(float(self.c**self.m), device=x.device))
        entropy = entropy / (max_entropy + eps)

        return entropy.unsqueeze(-1)

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
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.net(x)

class DDEMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10, c=3, m=2, tau=1):
        super().__init__()
        self.dde = DDELayer(c=c, m=m, tau=tau)
        # DDE output is a single value (entropy)
        # That's probably not enough for classification on its own if we just take one entropy value.
        # Usually one might use different m or tau.
        # Let's say we use a few combinations? No, let's just stick to the requested experiment.
        # But for classification, a single feature is likely to fail.
        # I'll use a few m, tau pairs or just a larger c^m if I were to use pattern distribution as features.

        # Actually, maybe the distribution itself is more informative than the entropy?
        # Standard DE experiment often looks at the distribution.
        # But "Dispersion Entropy" refers to the scalar value.

        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        feat = self.dde(x)
        return self.net(feat)

class DDEAugmentedMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10, c=3, m=2, tau=1):
        super().__init__()
        self.dde = DDELayer(c=c, m=m, tau=tau)
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if x.dim() == 2:
            x_raw = x
        else:
            x_raw = x.squeeze(1)
        feat = self.dde(x)
        combined = torch.cat([x_raw, feat], dim=1)
        return self.net(combined)
