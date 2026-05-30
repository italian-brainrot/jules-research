import torch
import torch.nn as nn
import torch.nn.functional as F

class Differentiable1DPersistence(nn.Module):
    """
    Computes 0D persistence of a 1D signal using sublevel set filtration.
    Vectorized implementation for PyTorch.
    """
    def __init__(self, top_k=10):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        # x shape: (batch, length)
        B, N = x.shape
        device = x.device

        # Add small epsilon to ensure unique values and handle flat regions
        x_eps = x + torch.linspace(0, 1e-6, N, device=device).unsqueeze(0)

        # 1. Identify local minima
        max_val = x_eps.max().detach() + 10.0
        padded_x = torch.cat([x_eps[:, :1] + max_val, x_eps, x_eps[:, -1:] + max_val], dim=1)
        is_min = (x_eps < padded_x[:, :-2]) & (x_eps < padded_x[:, 2:]) # (B, N)

        # 2. For each point, find L_i and R_i (nearest indices with smaller values)
        val_i = x_eps.unsqueeze(2) # (B, N, 1)
        val_j = x_eps.unsqueeze(1) # (B, 1, N)
        smaller = val_j < val_i # (B, N, N)

        indices = torch.arange(N, device=device).reshape(1, 1, N)
        i_indices = torch.arange(N, device=device).reshape(1, N, 1)

        left_mask = (indices < i_indices) & smaller
        right_mask = (indices > i_indices) & smaller

        # L_i is the max index in left_mask. Use -1e6 for "not found"
        L_i = (indices * left_mask.float() - (1 - left_mask.float()) * 1e6).max(dim=2).values
        # R_i is the min index in right_mask. Use 1e6 for "not found"
        R_i = (indices * right_mask.float() + (1 - right_mask.float()) * 1e6).min(dim=2).values

        # 3. Compute Range Maxima M_L and M_R
        j_indices = torch.arange(N, device=device).reshape(1, 1, N)

        in_range_L = (j_indices >= L_i.unsqueeze(2)) & (j_indices <= i_indices)
        M_L = (x_eps.unsqueeze(1) * in_range_L.float() - (1 - in_range_L.float()) * 1e6).max(dim=2).values

        in_range_R = (j_indices >= i_indices) & (j_indices <= R_i.unsqueeze(2))
        M_R = (x_eps.unsqueeze(1) * in_range_R.float() - (1 - in_range_R.float()) * 1e6).max(dim=2).values

        # 4. Compute death values
        has_L = (L_i > -0.5)
        has_R = (R_i < N - 0.5)

        death = torch.full_like(x_eps, -1e6)

        mask_both = has_L & has_R & is_min
        death[mask_both] = torch.min(M_L[mask_both], M_R[mask_both])

        mask_L = has_L & (~has_R) & is_min
        death[mask_L] = M_L[mask_L]

        mask_R = (~has_L) & has_R & is_min
        death[mask_R] = M_R[mask_R]

        # 5. Calculate persistence and sort
        persistence = death - x_eps
        persistence = F.relu(persistence)

        persistence, _ = torch.sort(persistence, dim=1, descending=True)

        if persistence.shape[1] < self.top_k:
            padding = torch.zeros(B, self.top_k - persistence.shape[1], device=device)
            persistence = torch.cat([persistence, padding], dim=1)

        return persistence[:, :self.top_k]

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, output_dim=10):
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

class PersistenceAugmentedMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, output_dim=10, top_k=10):
        super().__init__()
        self.sub_persistence = Differentiable1DPersistence(top_k=top_k)
        self.upper_persistence = Differentiable1DPersistence(top_k=top_k)
        self.net = nn.Sequential(
            nn.Linear(input_dim + 2 * top_k, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        p_sub = self.sub_persistence(x)
        p_upper = self.upper_persistence(-x)
        combined = torch.cat([x, p_sub, p_upper], dim=1)
        return self.net(combined)

class PersistenceMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, output_dim=10, top_k=10):
        super().__init__()
        self.sub_persistence = Differentiable1DPersistence(top_k=top_k)
        self.upper_persistence = Differentiable1DPersistence(top_k=top_k)
        self.net = nn.Sequential(
            nn.Linear(2 * top_k, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        p_sub = self.sub_persistence(x)
        p_upper = self.upper_persistence(-x)
        combined = torch.cat([p_sub, p_upper], dim=1)
        return self.net(combined)
