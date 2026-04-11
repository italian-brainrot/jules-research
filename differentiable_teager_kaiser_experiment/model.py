import torch
import torch.nn as nn
import torch.nn.functional as F

class TeagerKaiserLayer(nn.Module):
    def __init__(self, in_channels=1, smooth_kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        if smooth_kernel_size > 1:
            # We use a learnable smoothing kernel
            self.smooth = nn.Conv1d(in_channels, in_channels, kernel_size=smooth_kernel_size,
                                    padding=smooth_kernel_size//2, groups=in_channels, bias=False)
            nn.init.constant_(self.smooth.weight, 1.0 / smooth_kernel_size)
        else:
            self.smooth = nn.Identity()

    def forward(self, x):
        """
        x shape: (batch, in_channels, length)
        """
        # Replicate padding to handle boundaries for TKEO
        x_padded = F.pad(x, (1, 1), mode='replicate')

        x_curr = x_padded[:, :, 1:-1]
        x_prev = x_padded[:, :, 0:-2]
        x_next = x_padded[:, :, 2:]

        # TKEO formula: psi(x[n]) = x[n]^2 - x[n-1]*x[n+1]
        out = x_curr**2 - x_prev * x_next

        return self.smooth(out)

class TKEOMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super().__init__()
        self.tkeo = TeagerKaiserLayer(in_channels=1)
        # TKEO(x) has same shape as x. We concatenate them.
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x shape: (batch, 40)
        if x.dim() == 2:
            x = x.unsqueeze(1) # (batch, 1, 40)

        x_energy = self.tkeo(x)

        x_combined = torch.cat([x, x_energy], dim=1) # (batch, 2, 40)
        x_combined = x_combined.view(x_combined.size(0), -1)
        return self.mlp(x_combined)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.mlp(x)

if __name__ == "__main__":
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    tkeo_mlp = TKEOMLP()
    baseline_mlp = BaselineMLP()
    print(f"TKEOMLP parameters: {count_parameters(tkeo_mlp)}")
    print(f"BaselineMLP parameters: {count_parameters(baseline_mlp)}")
