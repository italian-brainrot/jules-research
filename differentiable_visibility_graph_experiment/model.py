import torch
import torch.nn as nn
import torch.nn.functional as F

class DVGLayer(nn.Module):
    def __init__(self, L=40, initial_scale=10.0, learnable_scale=True):
        super().__init__()
        self.L = L
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(initial_scale))
        else:
            self.register_buffer('scale', torch.tensor(initial_scale))

        self.register_buffer('t', torch.linspace(0, 1, L))

        # Precompute masks and weights to avoid re-creating them every forward pass
        i_idx = torch.arange(L).view(L, 1, 1)
        j_idx = torch.arange(L).view(1, L, 1)
        k_idx = torch.arange(L).view(1, 1, L)

        self.register_buffer('mask_k', (i_idx < k_idx) & (k_idx < j_idx))
        self.register_buffer('upper_tri_mask', (torch.arange(L).view(L, 1) < torch.arange(L).view(1, L)))

        ti = self.t.view(L, 1, 1)
        tj = self.t.view(1, L, 1)
        tk = self.t.view(1, 1, L)
        W = (tk - ti) / (tj - ti + 1e-9) # (L, L, L)
        self.register_buffer('W', W)

    def forward(self, y):
        # y: (B, L)
        B, L = y.shape

        yi = y.view(B, L, 1, 1)
        yj = y.view(B, 1, L, 1)
        yk = y.view(B, 1, 1, L)

        # Visibility criterion: yk < yi + (yj - yi) * (tk - ti) / (tj - ti)
        # V = yi + (yj - yi) * W - yk

        V = yi + (yj - yi) * self.W.unsqueeze(0) - yk # (B, L, L, L)

        S = torch.sigmoid(self.scale * V)

        # Apply mask for k in (i, j)
        S_masked = S.masked_fill(~self.mask_k.unsqueeze(0), 1.0)

        # Product over k
        # To avoid numerical issues with prod, use sum of logs
        # A = torch.exp(torch.sum(torch.log(S_masked + 1e-9), dim=-1))
        # Actually, for L=40, prod might be fine.
        A = torch.prod(S_masked, dim=-1)

        # A currently has shape (B, L, L). Only the upper triangle (i < j) is valid.
        A = A * self.upper_tri_mask.unsqueeze(0).float()

        # Symmetrize
        A = A + A.transpose(1, 2)

        return A

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

class DVGMLP(nn.Module):
    def __init__(self, L=40, hidden_dim=128, output_dim=10, initial_scale=10.0):
        super().__init__()
        self.dvg = DVGLayer(L=L, initial_scale=initial_scale)
        # Adjacency matrix is L*L. Since it's symmetric and diagonal is 0,
        # we could only take the upper triangle, but flattening is easier for now.
        self.input_proj = nn.Linear(L + L*L, hidden_dim)
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (B, L)
        A = self.dvg(x) # (B, L, L)
        A_flat = A.view(x.shape[0], -1)
        combined = torch.cat([x, A_flat], dim=1)
        return self.net(self.input_proj(combined))

class DVGGNN(nn.Module):
    def __init__(self, L=40, hidden_dim=128, output_dim=10, initial_scale=10.0):
        super().__init__()
        self.dvg = DVGLayer(L=L, initial_scale=initial_scale)
        self.node_proj = nn.Linear(1, hidden_dim)
        # Simple GCN-like layer: X' = (A + I) X W
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (B, L)
        B, L = x.shape
        A = self.dvg(x) # (B, L, L)

        # Add identity
        I = torch.eye(L, device=x.device).unsqueeze(0)
        A_hat = A + I

        # Node features: just the signal value
        h = self.node_proj(x.unsqueeze(-1)) # (B, L, hidden_dim)

        # Layer 1
        h = F.relu(torch.matmul(A_hat, self.w1(h)))
        # Layer 2
        h = F.relu(torch.matmul(A_hat, self.w2(h)))

        # Global average pooling
        h_pool = h.mean(dim=1)
        return self.classifier(h_pool)
