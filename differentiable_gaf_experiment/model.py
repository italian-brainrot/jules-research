import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableGAF(nn.Module):
    """
    Differentiable Gramian Angular Field (GAF) layer.
    Converts a 1D signal into a 2D Gramian Angular Summation Field (GASF)
    or Gramian Angular Difference Field (GADF).
    """
    def __init__(self, method='summation'):
        super().__init__()
        self.method = method

    def forward(self, x):
        """
        x: (B, L) signal
        Returns: (B, 1, L, L) GAF image
        """
        # 1. Scale to [-1, 1]
        # Using min-max scaling per sample
        min_val = x.min(dim=1, keepdim=True)[0]
        max_val = x.max(dim=1, keepdim=True)[0]
        x_scaled = 2 * (x - min_val) / (max_val - min_val + 1e-8) - 1

        # Clip to ensure values are in [-1, 1] for acos
        x_scaled = torch.clamp(x_scaled, -1.0 + 1e-7, 1.0 - 1e-7)

        # 2. Polar transformation
        phi = torch.acos(x_scaled) # (B, L)

        # 3. GAF generation
        # GASF: cos(phi_i + phi_j) = cos(phi_i)cos(phi_j) - sin(phi_i)sin(phi_j)
        # GADF: sin(phi_i - phi_j) = sin(phi_i)cos(phi_j) - cos(phi_i)sin(phi_j)

        phi_i = phi.unsqueeze(2) # (B, L, 1)
        phi_j = phi.unsqueeze(1) # (B, 1, L)

        if self.method == 'summation':
            # cos(phi_i + phi_j)
            gaf = torch.cos(phi_i + phi_j)
        elif self.method == 'difference':
            # sin(phi_i - phi_j)
            gaf = torch.sin(phi_i - phi_j)
        else:
            raise ValueError("Method must be 'summation' or 'difference'")

        return gaf.unsqueeze(1) # (B, 1, L, L)

class GAFNet(nn.Module):
    def __init__(self, input_len=40, num_classes=10, method='summation'):
        super().__init__()
        self.gaf = DifferentiableGAF(method=method)

        # 2D CNN on the GAF image
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Calculate flat features: 40x40 -> 20x20 -> 10x10
        flat_size = 32 * (input_len // 4) * (input_len // 4)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2:
            pass # (B, L)
        elif x.dim() == 3:
            x = x.squeeze(1) # (B, 1, L) -> (B, L)

        gaf_img = self.gaf(x)
        features = self.conv(gaf_img)
        logits = self.classifier(features)
        return logits

class CNN1D(nn.Module):
    def __init__(self, input_len=40, num_classes=10):
        super().__init__()
        # Roughly matching parameters of GAFNet (~210k)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        flat_size = 128 * (input_len // 4) # 128 * 10 = 1280

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 145), # Adjusted to get close to 210k total params
            nn.ReLU(),
            nn.Linear(145, num_classes)
        )
        # Total params check:
        # Conv1: 64*3 + 64 = 256
        # Conv2: 128*64*3 + 128 = 24704
        # FC1: 1280*145 + 145 = 185745
        # FC2: 145*10 + 10 = 1460
        # Total: 212,165. GAFNet has ~210,330. Very close.

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) # (B, L) -> (B, 1, L)

        features = self.conv(x)
        logits = self.classifier(features)
        return logits
