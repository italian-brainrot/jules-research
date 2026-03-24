import torch
import torch.nn as nn
import torch.nn.functional as F

class SFALayer(nn.Module):
    """
    A layer that encourages 'slow' features by penalizing the variance of the first-order
    differences of the features along the signal dimension.
    """
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        # x shape: (batch, in_channels, length)
        y = self.conv(x)
        # y shape: (batch, out_channels, length)

        # Calculate slowness penalty
        # Delta y: first-order difference along the length dimension
        dy = y[:, :, 1:] - y[:, :, :-1]

        # SFA objective is typically E[dy^2] / E[y^2]
        # We'll use a simplified version for regularization
        # We want to minimize the variance of the differences relative to the variance of the features

        # Batch and length dimensions are both used for variance calculation
        # We'll compute it per channel
        var_y = torch.var(y, dim=(0, 2), keepdim=True) + 1e-6
        var_dy = torch.mean(dy**2, dim=(0, 2), keepdim=True)

        slowness = torch.mean(var_dy / var_y)

        return y, slowness

class SFAMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, output_dim=10, num_layers=2):
        super().__init__()

        # Treat input as 1 channel signal
        self.sfa_layer = SFALayer(1, 16, kernel_size=5)

        # Flattened dimension after SFA
        sfa_out_dim = 16 * input_dim

        layers = []
        curr_dim = sfa_out_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, 40)
        x = x.unsqueeze(1) # (batch, 1, 40)
        y, slowness_penalty = self.sfa_layer(x)
        y = y.view(y.size(0), -1) # Flatten
        logits = self.mlp(y)
        return logits, slowness_penalty

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, output_dim=10, num_layers=3):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x), torch.tensor(0.0, device=x.device)
