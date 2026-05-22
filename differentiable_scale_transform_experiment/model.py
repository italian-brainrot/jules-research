import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DifferentiableScaleTransform(nn.Module):
    def __init__(self, input_size, num_scale_bins=None, t_min=1.0, t_max=None):
        super().__init__()
        self.input_size = input_size
        self.num_scale_bins = num_scale_bins or input_size
        self.t_min = t_min
        self.t_max = t_max or (input_size - 1.0)

        # Precompute the log-sampling grid
        u = torch.linspace(np.log(self.t_min), np.log(self.t_max), self.num_scale_bins)
        self.register_buffer("t_grid", torch.exp(u))
        self.register_buffer("sqrt_t", torch.sqrt(self.t_grid))

    def forward(self, x):
        # x: (B, N)
        B, N = x.shape

        # Linear interpolation
        x_reshaped = x.view(B, 1, 1, N)

        # Normalize t_grid to [-1, 1] for grid_sample
        # 0 maps to -1, N-1 maps to 1
        grid_t = 2.0 * self.t_grid / (N - 1) - 1.0

        grid = torch.zeros(B, 1, self.num_scale_bins, 2, device=x.device, dtype=x.dtype)
        grid[:, 0, :, 0] = grid_t # x-coordinate

        sampled = F.grid_sample(x_reshaped, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled = sampled.view(B, self.num_scale_bins)

        # Weight by sqrt(t) for isometry
        weighted = sampled * self.sqrt_t

        # FFT to get the Scale Transform
        spectral = torch.fft.rfft(weighted)
        magnitude = torch.abs(spectral)

        return magnitude

class FourierMellinLayer(nn.Module):
    def __init__(self, input_size, num_scale_bins=None):
        super().__init__()
        self.st_layer = DifferentiableScaleTransform(input_size // 2 + 1, num_scale_bins=num_scale_bins, t_min=1.0)

    def forward(self, x):
        # 1. FFT magnitude (Shift Invariance)
        x_fft = torch.fft.rfft(x)
        x_mag = torch.abs(x_fft)

        # 2. Scale Transform (Scale Invariance)
        # Note: x_mag has size (input_size // 2 + 1)
        res = self.st_layer(x_mag)
        return res

class BaselineMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.net(x)

class ScaleTransformMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_scale_bins=None):
        super().__init__()
        self.st_layer = DifferentiableScaleTransform(input_size, num_scale_bins=num_scale_bins)
        st_out_size = (self.st_layer.num_scale_bins // 2) + 1
        self.net = nn.Sequential(
            nn.Linear(st_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        st_features = self.st_layer(x)
        return self.net(st_features)

class FourierMellinMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_scale_bins=None):
        super().__init__()
        self.fm_layer = FourierMellinLayer(input_size, num_scale_bins=num_scale_bins)
        st_out_size = (self.fm_layer.st_layer.num_scale_bins // 2) + 1
        self.net = nn.Sequential(
            nn.Linear(st_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        fm_features = self.fm_layer(x)
        return self.net(fm_features)

class ScaleTransformAugmentedMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_scale_bins=None):
        super().__init__()
        self.st_layer = DifferentiableScaleTransform(input_size, num_scale_bins=num_scale_bins)
        st_out_size = (self.st_layer.num_scale_bins // 2) + 1
        self.net = nn.Sequential(
            nn.Linear(input_size + st_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        st_features = self.st_layer(x)
        combined = torch.cat([x, st_features], dim=1)
        return self.net(combined)
