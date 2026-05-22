import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def gaussian_derivative_kernels(sigma, kernel_size, device):
    """
    Returns 0th, 1st, 2nd, and 3rd derivative kernels of a Gaussian.
    """
    x = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size).to(device)

    # 0th derivative (Gaussian)
    g0 = torch.exp(-0.5 * (x / sigma)**2)
    g0 = g0 / (sigma * np.sqrt(2 * np.pi) + 1e-6)

    # 1st derivative: -x/sigma^2 * g0
    g1 = -(x / (sigma**2 + 1e-6)) * g0

    # 2nd derivative: (x^2/sigma^4 - 1/sigma^2) * g0
    g2 = (x**2 / (sigma**4 + 1e-6) - 1 / (sigma**2 + 1e-6)) * g0

    # 3rd derivative: (-x^3/sigma^6 + 3x/sigma^4) * g0
    g3 = (-x**3 / (sigma**6 + 1e-6) + 3 * x / (sigma**4 + 1e-6)) * g0

    return g0, g1, g2, g3

class DifferentiableFrenetSerret(nn.Module):
    def __init__(self, kernel_size=31, init_sigma=1.0, init_tau=2.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = nn.Parameter(torch.tensor([float(init_sigma)]))
        self.tau_delay = nn.Parameter(torch.tensor([float(init_tau)]))

    def forward(self, x):
        # x: (B, L)
        B, L = x.shape
        device = x.device

        # Ensure sigma is positive
        sigma = torch.clamp(self.sigma, min=0.1, max=self.kernel_size / 4.0)

        g0, g1, g2, g3 = gaussian_derivative_kernels(sigma, self.kernel_size, device)

        # Reshape for conv1d: (1, 1, K)
        g0 = g0.view(1, 1, -1)
        g1 = g1.view(1, 1, -1)
        g2 = g2.view(1, 1, -1)
        g3 = g3.view(1, 1, -1)

        x_reshaped = x.view(B, 1, L)

        # Compute derivatives
        padding = self.kernel_size // 2
        # Use constant padding with 0 for derivatives might be bad if we want offset invariance.
        # But derivatives of a constant are 0 anyway.
        # Wait, if we pad with 0, a constant signal will have non-zero derivatives at edges.
        # Use replicate padding.
        x_padded = F.pad(x_reshaped, (padding, padding), mode='replicate')

        x1 = F.conv1d(x_padded, g1)[:, 0, :]
        x2 = F.conv1d(x_padded, g2)[:, 0, :]
        x3 = F.conv1d(x_padded, g3)[:, 0, :]

        def get_delayed_diff(signal, delay):
            # signal: (B, L)
            # delay: tensor scalar
            B, L = signal.shape

            d_floor = torch.floor(delay)
            alpha = delay - d_floor

            # Use replicate padding for delay embedding too
            pad_size = int(torch.ceil(torch.abs(delay)).item()) + 2
            padded_signal = F.pad(signal, (pad_size, pad_size), mode='replicate')

            base_indices = torch.arange(L, device=device) + pad_size

            idx_floor = base_indices - d_floor.long()
            idx_ceil = idx_floor - 1

            val_floor = torch.gather(padded_signal, 1, idx_floor.unsqueeze(0).expand(B, -1))
            val_ceil = torch.gather(padded_signal, 1, idx_ceil.unsqueeze(0).expand(B, -1))

            return (1 - alpha) * val_floor + alpha * val_ceil

        tau = torch.clamp(self.tau_delay, min=0.0, max=L/4.0)

        p1_0 = x1
        p1_1 = get_delayed_diff(x1, tau)
        p1_2 = get_delayed_diff(x1, 2*tau)

        p2_0 = x2
        p2_1 = get_delayed_diff(x2, tau)
        p2_2 = get_delayed_diff(x2, 2*tau)

        p3_0 = x3
        p3_1 = get_delayed_diff(x3, tau)
        p3_2 = get_delayed_diff(x3, 2*tau)

        def cross_product(a, b):
            res = torch.zeros_like(a)
            res[:, 0, :] = a[:, 1, :] * b[:, 2, :] - a[:, 2, :] * b[:, 1, :]
            res[:, 1, :] = a[:, 2, :] * b[:, 0, :] - a[:, 0, :] * b[:, 2, :]
            res[:, 2, :] = a[:, 0, :] * b[:, 1, :] - a[:, 1, :] * b[:, 0, :]
            return res

        def norm(a):
            return torch.sqrt(torch.sum(a**2, dim=1) + 1e-8)

        P1 = torch.stack([p1_0, p1_1, p1_2], dim=1)
        P2 = torch.stack([p2_0, p2_1, p2_2], dim=1)
        P3 = torch.stack([p3_0, p3_1, p3_2], dim=1)

        P1_cross_P2 = cross_product(P1, P2)
        kappa = norm(P1_cross_P2) / (norm(P1)**3 + 1e-8)

        numerator = torch.sum(P1_cross_P2 * P3, dim=1)
        torsion = numerator / (torch.sum(P1_cross_P2**2, dim=1) + 1e-8)

        # We should probably ignore edges where padding effects are strong
        edge = padding
        kappa = kappa[:, edge:-edge]
        torsion = torsion[:, edge:-edge]

        features = torch.stack([
            kappa.mean(dim=1),
            kappa.max(dim=1)[0],
            kappa.std(dim=1),
            torsion.mean(dim=1),
            torsion.max(dim=1)[0],
            torsion.std(dim=1)
        ], dim=1)

        return features

class FrenetSerretAugmentedMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fs_layer = DifferentiableFrenetSerret()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        fs_features = self.fs_layer(x)
        combined = torch.cat([x, fs_features], dim=1)
        return self.mlp(combined)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 258),
            nn.ReLU(),
            nn.Linear(258, 258),
            nn.ReLU(),
            nn.Linear(258, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)
