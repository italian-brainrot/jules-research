import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GammatoneLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, n=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.n = n

        # Learnable parameters
        # frequencies between 0 and 10 (normalized units)
        self.frequencies = nn.Parameter(torch.rand(out_channels, in_channels, 1) * 10.0)
        # bandwidths b between 0.1 and 5.0
        self.bandwidths = nn.Parameter(torch.rand(out_channels, in_channels, 1) * 4.9 + 0.1)
        # phases between 0 and 2*pi
        self.phases = nn.Parameter(torch.rand(out_channels, in_channels, 1) * 2 * math.pi)

        # Grid for the kernel (t >= 0)
        grid = torch.linspace(0, 1, kernel_size)
        self.register_buffer("grid", grid)

    def get_kernels(self):
        # grid: (K,)
        # frequencies, bandwidths, phases: (Out, In, 1)

        t = self.grid.view(1, 1, self.kernel_size) # (1, 1, K)

        # Gammatone formula: t^(n-1) * exp(-2*pi*b*t) * cos(2*pi*f*t + phi)
        # Envelope: t^(n-1) * exp(-2*pi*b*t)
        # Use a small epsilon for t if n-1 < 0, but here n=4 so it's fine.
        envelope = (t ** (self.n - 1)) * torch.exp(-2 * math.pi * self.bandwidths * t)

        # Sinusoidal carrier
        sinusoid = torch.cos(2 * math.pi * self.frequencies * t + self.phases)

        kernels = envelope * sinusoid

        # Normalize kernels to have unit L2 norm per filter for stability
        kernels_norm = torch.norm(kernels, p=2, dim=2, keepdim=True)
        kernels = kernels / (kernels_norm + 1e-8)

        return kernels

    def forward(self, x):
        kernels = self.get_kernels() # (Out, In, K)
        return F.conv1d(x, kernels, stride=self.stride, padding=self.padding)

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
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.net(x)

class GammatoneConvMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10, kernel_size=15, num_filters=16):
        super().__init__()
        self.gammatone = GammatoneLayer(1, num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        conv_out_dim = num_filters * input_dim
        self.mlp = nn.Sequential(
            nn.Linear(conv_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.gammatone(x)
        x = x.view(x.size(0), -1)
        return self.mlp(x)

class StandardConvMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10, kernel_size=15, num_filters=16):
        super().__init__()
        self.conv = nn.Conv1d(1, num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        conv_out_dim = num_filters * input_dim
        self.mlp = nn.Sequential(
            nn.Linear(conv_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.mlp(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    gammatone_mlp = GammatoneConvMLP()
    conv_mlp = StandardConvMLP()
    base_mlp = BaselineMLP()
    print(f"GammatoneConvMLP parameters: {count_parameters(gammatone_mlp)}")
    print(f"StandardConvMLP parameters: {count_parameters(conv_mlp)}")
    print(f"BaselineMLP parameters: {count_parameters(base_mlp)}")
