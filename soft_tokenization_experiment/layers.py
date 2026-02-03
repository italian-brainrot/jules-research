import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryGuidedSoftTokenization(nn.Module):
    def __init__(self, input_len, output_len, dim, sigma=0.5):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.dim = dim
        self.sigma = sigma

        # Network to predict "density" deltas
        self.delta_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape

        # Predict deltas
        # We use softplus to ensure they are positive
        deltas = F.softplus(self.delta_net(x).squeeze(-1)) # (B, L)

        # Cumulative sum to get "soft positions"
        c = torch.cumsum(deltas, dim=1) # (B, L)

        # Normalize so each sequence maps its characters to [0, M-1]
        c_min = c[:, 0:1]
        c_max = c[:, -1:]
        # Avoid division by zero
        c_norm = (c - c_min) / (c_max - c_min + 1e-6) * (self.output_len - 1)

        # Create target slots [0, 1, ..., M-1]
        slots = torch.arange(self.output_len, device=x.device, dtype=x.dtype) # (M,)

        # Compute weights using Gaussian kernel
        # distance: (B, M, L)
        dist = (c_norm.unsqueeze(1) - slots.unsqueeze(0).unsqueeze(2)).pow(2)
        weights = torch.exp(-dist / (2 * self.sigma**2))

        # Normalize weights so each SLOT should sum to 1 to be a proper pooling
        weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-6)

        # Aggregate
        pooled = torch.bmm(weights, x) # (B, M, D)
        return pooled

class AttentionPooling(nn.Module):
    def __init__(self, output_len, dim, num_heads=4):
        super().__init__()
        self.output_len = output_len
        self.dim = dim
        self.queries = nn.Parameter(torch.randn(output_len, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):
        # x: (B, L, D)
        # queries: (B, M, D)
        B = x.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        # MultiheadAttention expects (query, key, value)
        out, _ = self.attn(q, x, x)
        return out

class UniformPooling(nn.Module):
    def __init__(self, input_len, output_len):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len

    def forward(self, x):
        # x: (B, L, D)
        # Simple interpolation / resizing
        # We use interpolate which expects (B, C, L)
        x = x.transpose(1, 2)
        out = F.interpolate(x, size=self.output_len, mode='linear', align_corners=True)
        return out.transpose(1, 2)
