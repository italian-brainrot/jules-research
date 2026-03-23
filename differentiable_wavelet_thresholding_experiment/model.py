import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DAWTLayer(nn.Module):
    """Differentiable Adaptive Wavelet Thresholding Layer using Haar Wavelets."""
    def __init__(self, levels=1, input_len=40):
        super().__init__()
        self.levels = levels
        self.input_len = input_len

        # Haar filters
        # Approximation (Low-pass): [1, 1] / sqrt(2)
        # Detail (High-pass): [1, -1] / sqrt(2)
        self.register_buffer('h_low', torch.tensor([1.0, 1.0]).view(1, 1, 2) / math.sqrt(2))
        self.register_buffer('h_high', torch.tensor([1.0, -1.0]).view(1, 1, 2) / math.sqrt(2))

        # Learnable thresholds for each level (one threshold per level for all detail coefficients)
        # Initializing with a small positive value
        self.thresholds = nn.Parameter(torch.full((levels,), 0.01))

    def soft_thresholding(self, x, tau):
        return torch.sign(x) * torch.relu(torch.abs(x) - tau)

    def dwt(self, x):
        # x: (B, 1, L)
        low = F.conv1d(x, self.h_low, stride=2)
        high = F.conv1d(x, self.h_high, stride=2)
        return low, high

    def idwt(self, low, high):
        # low, high: (B, 1, L/2)
        # IDWT for Haar: x[2n] = (low[n] + high[n])/sqrt(2), x[2n+1] = (low[n] - high[n])/sqrt(2)
        # Equivalent to conv_transpose1d
        combined = torch.cat([low, high], dim=1) # (B, 2, L/2)
        # For conv_transpose1d, weight shape is (in_channels, out_channels/groups, kW)
        # in_channels=2 (low, high), out_channels=1
        weights = torch.cat([self.h_low, self.h_high], dim=0) # (2, 1, 2)
        return F.conv_transpose1d(combined, weights, stride=2)

    def forward(self, x):
        # x: (B, L)
        original_shape = x.shape
        x = x.unsqueeze(1) # (B, 1, L)

        # Multi-level decomposition
        low = x
        details = []
        for i in range(self.levels):
            low, high = self.dwt(low)
            # Apply learnable soft-thresholding to detail coefficients
            high_thresh = self.soft_thresholding(high, torch.abs(self.thresholds[i]))
            details.append(high_thresh)

        # Reconstruction
        res = low
        for i in reversed(range(self.levels)):
            res = self.idwt(res, details[i])

        # Handle padding if L was not divisible by 2^levels
        # For MNIST-1D, L=40, so up to 3 levels (40 -> 20 -> 10 -> 5) is fine.
        return res.squeeze(1)[:, :original_shape[1]]

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

class DAWTMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10, levels=2):
        super().__init__()
        self.dawt = DAWTLayer(levels=levels, input_len=input_dim)
        self.mlp = BaselineMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    def forward(self, x):
        x_denoised = self.dawt(x)
        return self.mlp(x_denoised)
