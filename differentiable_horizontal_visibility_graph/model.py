import torch
import torch.nn as nn
import torch.nn.functional as F

class DHVGLayer(nn.Module):
    """
    Differentiable Horizontal Visibility Graph (DHVG) Layer.
    Computes a soft adjacency matrix based on the horizontal visibility criterion:
    Two nodes i and j are connected if x_k < min(x_i, x_j) for all i < k < j.
    """
    def __init__(self, L=40, initial_scale=10.0, learnable_scale=True):
        super().__init__()
        self.L = L
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(initial_scale))
        else:
            self.register_buffer('scale', torch.tensor(initial_scale))

        # Precompute masks to avoid re-creating them every forward pass
        i_idx = torch.arange(L).view(L, 1, 1)
        j_idx = torch.arange(L).view(1, L, 1)
        k_idx = torch.arange(L).view(1, 1, L)

        # mask_k is True for k such that i < k < j
        self.register_buffer('mask_k', (i_idx < k_idx) & (k_idx < j_idx))
        # upper_tri_mask is True for i < j
        self.register_buffer('upper_tri_mask', (torch.arange(L).view(L, 1) < torch.arange(L).view(1, L)))

    def forward(self, x):
        """
        x: (B, L)
        returns: (B, L, L) soft adjacency matrix
        """
        B, L = x.shape
        xi = x.view(B, L, 1, 1)
        xj = x.view(B, 1, L, 1)
        xk = x.view(B, 1, 1, L)

        # Horizontal visibility criterion: xk < min(xi, xj)
        # We approximate this using a soft-thresholding on min(xi, xj) - xk

        # V_ijk = min(x_i, x_j) - x_k
        V = torch.min(xi, xj) - xk # (B, L, L, L)

        # Soft visibility score S_ijk = sigmoid(scale * V_ijk)
        # If V_ijk > 0, S_ijk -> 1. If V_ijk < 0, S_ijk -> 0.
        S = torch.sigmoid(self.scale * V)

        # Apply mask for k in (i, j).
        # For k NOT in (i, j), we want S to be 1.0 so it doesn't affect the product.
        S_masked = S.masked_fill(~self.mask_k.unsqueeze(0), 1.0)

        # Product over k: A_ij = prod_{k=i+1}^{j-1} S_ijk
        # For j = i + 1, the product is over an empty set, which is 1.0.
        # This correctly represents that adjacent nodes are always horizontally visible.
        A = torch.prod(S_masked, dim=-1)

        # Apply upper triangle mask (i < j)
        A = A * self.upper_tri_mask.unsqueeze(0).float()

        # Symmetrize
        A = A + A.transpose(1, 2)

        return A

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

class DHVGAugmentedMLP(nn.Module):
    def __init__(self, L=40, hidden_dim=256, output_dim=10, initial_scale=10.0):
        super().__init__()
        self.dhvg = DHVGLayer(L=L, initial_scale=initial_scale)
        # Flattened adjacency matrix is L*L
        self.input_proj = nn.Linear(L + L*L, hidden_dim)
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        A = self.dhvg(x)
        A_flat = A.view(x.shape[0], -1)
        combined = torch.cat([x, A_flat], dim=1)
        return self.net(self.input_proj(combined))

class DHVGGNN(nn.Module):
    def __init__(self, L=40, hidden_dim=256, output_dim=10, initial_scale=10.0):
        super().__init__()
        self.dhvg = DHVGLayer(L=L, initial_scale=initial_scale)
        self.node_proj = nn.Linear(1, hidden_dim)
        # Simple GCN layers
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        B, L = x.shape
        A = self.dhvg(x)

        # Add identity for self-loops
        I = torch.eye(L, device=x.device).unsqueeze(0)
        A_hat = A + I

        # Node features: signal value
        h = self.node_proj(x.unsqueeze(-1)) # (B, L, hidden_dim)

        # GCN passes: X' = activation(A_hat @ X @ W)
        h = F.relu(torch.matmul(A_hat, self.w1(h)))
        h = F.relu(torch.matmul(A_hat, self.w2(h)))

        # Global average pooling
        h_pool = h.mean(dim=1)
        return self.classifier(h_pool)
