import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableMorphology1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, soft_type='lse', tau=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.soft_type = soft_type
        self.tau = nn.Parameter(torch.tensor(tau))
        self.kernel = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))

    def forward(self, x, mode='dilation'):
        # x: (B, C_in, L)
        # kernel: (C_out, C_in, K)

        batch_size, _, L = x.shape
        padding = self.kernel_size // 2
        # Use replicate padding for morphology to avoid boundary artifacts
        x_padded = F.pad(x, (padding, padding), mode='replicate')

        # Adjust for even kernel sizes if necessary, but we'll stick to odd for simplicity
        if self.kernel_size % 2 == 0:
            x_padded = x_padded[:, :, :-1]

        x_unfolded = x_padded.unfold(2, self.kernel_size, 1) # (B, C_in, L, K)

        # Expand for broadcasting
        # x_unfolded: (B, 1, C_in, L, K)
        # kernel: (1, C_out, C_in, 1, K)
        x_expanded = x_unfolded.unsqueeze(1)
        k_expanded = self.kernel.unsqueeze(0).unsqueeze(3)

        # Ensure tau is positive and not too small for stability
        tau = torch.clamp(self.tau, min=1e-3)

        if mode == 'dilation':
            vals = x_expanded + k_expanded # (B, C_out, C_in, L, K)
            if self.soft_type == 'lse':
                # Sum over input channels and kernel width
                res = tau * torch.logsumexp(vals / tau, dim=(2, 4))
            else:
                # Weighted Softmax
                weights = F.softmax(vals / tau, dim=(2, 4))
                res = torch.sum(vals * weights, dim=(2, 4))
        elif mode == 'erosion':
            vals = x_expanded - k_expanded
            if self.soft_type == 'lse':
                res = -tau * torch.logsumexp(-vals / tau, dim=(2, 4))
            else:
                weights = F.softmax(-vals / tau, dim=(2, 4))
                res = torch.sum(vals * weights, dim=(2, 4))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return res

class Dilation1d(DifferentiableMorphology1d):
    def forward(self, x):
        return super().forward(x, mode='dilation')

class Erosion1d(DifferentiableMorphology1d):
    def forward(self, x):
        return super().forward(x, mode='erosion')

class Opening1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, tau=0.1):
        super().__init__()
        self.erosion = Erosion1d(in_channels, out_channels, kernel_size, tau=tau)
        self.dilation = Dilation1d(out_channels, out_channels, kernel_size, tau=tau)

    def forward(self, x):
        return self.dilation(self.erosion(x))

class Closing1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, tau=0.1):
        super().__init__()
        self.dilation = Dilation1d(in_channels, out_channels, kernel_size, tau=tau)
        self.erosion = Erosion1d(out_channels, out_channels, kernel_size, tau=tau)

    def forward(self, x):
        return self.erosion(self.dilation(x))

class MorphologyNet(nn.Module):
    def __init__(self, input_dim=40, num_classes=10, num_kernels=8, kernel_size=5, tau=0.1):
        super().__init__()
        self.opening = Opening1d(1, num_kernels, kernel_size, tau=tau)
        self.closing = Closing1d(1, num_kernels, kernel_size, tau=tau)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_kernels * 2 * input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        o = self.opening(x)
        c = self.closing(x)
        features = torch.cat([o, c], dim=1) # (B, 2*num_kernels, L)
        return self.classifier(features)

class MLPBaseline(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class Conv1dBaseline(nn.Module):
    def __init__(self, in_channels=1, num_filters=32, kernel_size=5, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, num_filters, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),
            nn.Conv1d(num_filters, num_filters, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),
            nn.AdaptiveMaxPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.classifier(self.conv(x))
