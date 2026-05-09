import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DZTLayer(nn.Module):
    def __init__(self, input_len, num_points):
        """
        Differentiable Z-Transform Layer.
        Evaluates the Z-transform at num_points learnable locations in the complex plane.
        z_k = exp(gamma_k + i * omega_k)
        """
        super().__init__()
        self.input_len = input_len
        self.num_points = num_points

        # Initialize points near the unit circle (gamma near 0)
        # and distributed in frequency (omega in [0, pi])
        self.gamma = nn.Parameter(torch.randn(num_points) * 0.01)
        self.omega = nn.Parameter(torch.linspace(0, np.pi, num_points) + torch.randn(num_points) * 0.1)

    def forward(self, x):
        """
        x: (batch, input_len)
        returns: (batch, num_points * 2) - concatenated real and imaginary parts
        """
        batch_size = x.shape[0]
        n = torch.arange(self.input_len, device=x.device, dtype=x.dtype) # (N,)

        # Compute -n * (gamma_k + i * omega_k)
        # Use broadcasting to get (num_points, input_len)
        exponent_real = -self.gamma.unsqueeze(1) * n.unsqueeze(0) # (K, N)
        exponent_imag = -self.omega.unsqueeze(1) * n.unsqueeze(0) # (K, N)

        # To prevent exponential explosion, we can clip or use a softer approach.
        # However, for N=40, it's likely manageable if initialized small.
        # We'll use complex exp: exp(a + ib) = exp(a) * (cos(b) + i sin(b))

        mag = torch.exp(exponent_real) # (K, N)

        real_part = mag * torch.cos(exponent_imag) # (K, N)
        imag_part = mag * torch.sin(exponent_imag) # (K, N)

        # x is (B, N)
        # Compute X(z_k) = sum_n x_n * z_k^{-n}
        # Result real: sum_n x_n * real_part_kn
        # Result imag: sum_n x_n * imag_part_kn

        res_real = torch.matmul(x, real_part.t()) # (B, K)
        res_imag = torch.matmul(x, imag_part.t()) # (B, K)

        # Concatenate real and imaginary parts
        res = torch.cat([res_real, res_imag], dim=-1) # (B, 2K)
        return res

class DZTAugmentedMLP(nn.Module):
    def __init__(self, input_dim, num_points, hidden_dim, output_dim):
        super().__init__()
        self.dzt = DZTLayer(input_dim, num_points)

        # input_dim + 2 * num_points
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + 2 * num_points, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        dzt_features = self.dzt(x)
        combined = torch.cat([x, dzt_features], dim=-1)
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
        return self.mlp(x)

class DZTNet(nn.Module):
    def __init__(self, input_dim, num_points, hidden_dim, output_dim):
        super().__init__()
        self.dzt = DZTLayer(input_dim, num_points)
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_points, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        dzt_features = self.dzt(x)
        return self.mlp(dzt_features)
