import torch
import torch.nn as nn
import torch.nn.functional as F

class DOPLayer(nn.Module):
    """
    Differentiable Ordinal Patterns Layer.
    Extracts local ordinal information by computing soft comparisons between
    elements in a sliding window.
    """
    def __init__(self, d=3, tau=1, stride=1, temperature=10.0, learnable_temperature=True):
        super().__init__()
        self.d = d
        self.tau = tau
        self.stride = stride
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, L)
        Returns:
            Output tensor of shape (B, C * num_pairs, num_windows)
            where num_pairs = d * (d - 1) // 2
        """
        B, C, L = x.shape
        window_span = (self.d - 1) * self.tau + 1

        if L < window_span:
            raise ValueError(f"Input length {L} is too short for window span {window_span}")

        # Unfold to get windows
        # x: (B, C, L) -> (B, C * window_span, num_windows)
        # F.unfold expects 4D (B, C, H, W). We use (window_span, 1) for (H, W).
        unfolded = F.unfold(x.unsqueeze(-1), (window_span, 1), stride=(self.stride, 1))

        # Reshape to (B, C, window_span, num_windows)
        num_windows = unfolded.shape[-1]
        unfolded = unfolded.view(B, C, window_span, num_windows)

        # Select indices based on tau
        indices = torch.arange(0, window_span, self.tau, device=x.device)
        windows = unfolded[:, :, indices, :] # (B, C, d, num_windows)

        # Compute all-pairs soft comparisons
        # windows: (B, C, d, num_windows)
        # We want pairs (i, j) with 0 <= i < j < d
        i_idx, j_idx = torch.triu_indices(self.d, self.d, offset=1, device=x.device)

        # x_i, x_j: (B, C, num_pairs, num_windows)
        x_i = windows[:, :, i_idx, :]
        x_j = windows[:, :, j_idx, :]

        diff = x_i - x_j
        # Soft comparison using sigmoid
        comparisons = torch.sigmoid(self.temperature * diff)

        # Reshape to (B, C * num_pairs, num_windows)
        num_pairs = self.d * (self.d - 1) // 2
        # comparisons: (B, C, num_pairs, num_windows) -> (B, num_pairs, C, num_windows) -> (B, num_pairs * C, num_windows)
        out = comparisons.permute(0, 2, 1, 3).reshape(B, num_pairs * C, num_windows)
        return out

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

class DOPMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10, d=3, tau=1):
        super().__init__()
        self.dop = DOPLayer(d=d, tau=tau)
        window_span = (d - 1) * tau + 1
        num_windows = (input_dim - window_span) // 1 + 1
        num_pairs = d * (d - 1) // 2
        dop_out_dim = num_pairs * num_windows

        self.mlp = nn.Sequential(
            nn.Linear(dop_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x_dop = self.dop(x)
        x_dop = torch.flatten(x_dop, 1)
        return self.mlp(x_dop)

class DOPAugmentedMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10, d=3, tau=1):
        super().__init__()
        self.dop = DOPLayer(d=d, tau=tau)
        window_span = (d - 1) * tau + 1
        num_windows = (input_dim - window_span) // 1 + 1
        num_pairs = d * (d - 1) // 2
        dop_out_dim = num_pairs * num_windows

        self.mlp = nn.Sequential(
            nn.Linear(input_dim + dop_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if x.dim() == 2:
            x_raw = x
            x = x.unsqueeze(1)
        else:
            x_raw = x.squeeze(1)

        x_dop = self.dop(x)
        x_dop = torch.flatten(x_dop, 1)
        x_combined = torch.cat([x_raw, x_dop], dim=1)
        return self.mlp(x_combined)
