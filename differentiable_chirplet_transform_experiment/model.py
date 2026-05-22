import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChirpletLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Learnable parameters
        # frequencies between 0 and 5
        self.frequencies = nn.Parameter(torch.rand(out_channels, in_channels, 1) * 5.0)
        # sigmas between 0.1 and 0.5 (relative to the [-1, 1] grid)
        self.sigmas = nn.Parameter(torch.rand(out_channels, in_channels, 1) * 0.4 + 0.1)
        # phases between 0 and 2*pi
        self.phases = nn.Parameter(torch.rand(out_channels, in_channels, 1) * 2 * math.pi)
        # chirp rates between -2 and 2
        self.chirp_rates = nn.Parameter(torch.randn(out_channels, in_channels, 1) * 1.0)

        # Grid for the kernel
        grid = torch.linspace(-1, 1, kernel_size)
        self.register_buffer("grid", grid)

    def get_kernels(self):
        # grid: (K,)
        # frequencies, sigmas, phases, chirp_rates: (Out, In, 1)

        # Reshape grid for broadcasting
        t = self.grid.view(1, 1, self.kernel_size) # (1, 1, K)

        # Chirplet formula: exp(-t^2 / (2*sigma^2)) * cos(2*pi*(f*t + 0.5*c*t^2) + phi)
        # Avoid division by zero for sigma
        sigmas = torch.clamp(self.sigmas, min=1e-3)

        # Gaussian envelope
        gaussian = torch.exp(-(t**2) / (2 * sigmas**2))

        # Sinusoidal carrier with linear frequency sweep (chirp)
        # instantaneous frequency = f + c*t
        # phase = 2*pi*(f*t + 0.5*c*t^2) + phi
        phase = 2 * math.pi * (self.frequencies * t + 0.5 * self.chirp_rates * t**2) + self.phases
        sinusoid = torch.cos(phase)

        kernels = gaussian * sinusoid
        return kernels

    def forward(self, x):
        kernels = self.get_kernels() # (Out, In, K)
        return F.conv1d(x, kernels, stride=self.stride, padding=self.padding)

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
        # x shape: (B, 40)
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.mlp(x)

class ConvMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=15, num_filters=16, use_chirplet=False, fixed_chirp=None):
        super().__init__()
        if use_chirplet:
            self.conv = ChirpletLayer(1, num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
            if fixed_chirp is not None:
                with torch.no_grad():
                    self.conv.chirp_rates.fill_(fixed_chirp)
                    self.conv.chirp_rates.requires_grad = False
        else:
            self.conv = nn.Conv1d(1, num_filters, kernel_size=kernel_size, padding=kernel_size // 2)

        # Calculate dimension after conv
        # For mnist1d, input_dim is 40. With padding=kernel_size//2 and stride=1, output length is 40.
        conv_out_dim = num_filters * input_dim

        self.mlp = nn.Sequential(
            nn.Linear(conv_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x shape: (B, 40)
        if x.dim() == 2:
            x = x.unsqueeze(1) # (B, 1, 40)
        x = self.conv(x) # (B, num_filters, 40)
        x = x.view(x.size(0), -1)
        return self.mlp(x)
