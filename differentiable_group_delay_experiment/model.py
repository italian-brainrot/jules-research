import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class DifferentiableGroupDelay(nn.Module):
    """
    Computes the Group Delay of a 1D signal in a differentiable way.
    The group delay is defined as the negative derivative of the phase with respect to frequency.
    Formula used: tau_g(w) = Re{ FFT(n * x[n]) / FFT(x[n]) }
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x shape: (batch, seq_len)
        batch_size, seq_len = x.shape
        device = x.device

        # n = [0, 1, ..., seq_len-1]
        n = torch.arange(seq_len, device=device, dtype=x.dtype)
        nx = n * x

        # FFTs
        X = torch.fft.rfft(x, dim=-1)
        NX = torch.fft.rfft(nx, dim=-1)

        # Group delay: Re(NX / X)
        # Using a small epsilon for stability in the denominator
        # We can also use: Re(NX * conj(X) / (|X|^2 + eps))
        denom = X.abs().pow(2) + self.eps
        gd = (NX * X.conj()).real / denom

        return gd

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class GroupDelayAugmentedMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, output_dim=10):
        super().__init__()
        self.gd_layer = DifferentiableGroupDelay()
        # rfft of length 40 gives floor(40/2) + 1 = 21 features
        gd_dim = (input_dim // 2) + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim + gd_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        gd = self.gd_layer(x)
        x_combined = torch.cat([x, gd], dim=1)
        return self.net(x_combined)

class GroupDelayMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, output_dim=10):
        super().__init__()
        self.gd_layer = DifferentiableGroupDelay()
        gd_dim = (input_dim // 2) + 1
        self.net = nn.Sequential(
            nn.Linear(gd_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        gd = self.gd_layer(x)
        return self.net(gd)
