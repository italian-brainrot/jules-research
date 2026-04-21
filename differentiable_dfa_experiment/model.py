import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableDFALayer(nn.Module):
    """
    Differentiable Detrended Fluctuation Analysis (DFA) Layer.
    Computes the scaling exponents or fluctuation values for different scales.
    """
    def __init__(self, scales=[4, 8, 10, 20], include_slope=True, include_fluctuations=True):
        super().__init__()
        self.scales = scales
        self.include_slope = include_slope
        self.include_fluctuations = include_fluctuations

    def forward(self, x):
        # x: (B, L)
        if x.dim() == 3:
            x = x.squeeze(1)
        B, L = x.shape
        device = x.device

        # 1. Integrate the signal
        x_mean = x.mean(dim=-1, keepdim=True)
        y = torch.cumsum(x - x_mean, dim=-1) # (B, L)

        fluctuations = []

        for n in self.scales:
            # 2. Divide into windows of size n
            # We might drop the last few samples if L is not divisible by n
            num_windows = L // n
            y_cut = y[:, :num_windows * n]
            y_windows = y_cut.view(B, num_windows, n) # (B, num_windows, n)

            # 3. Local linear fit in each window
            # t = [0, 1, ..., n-1]
            t = torch.arange(n, dtype=x.dtype, device=device).view(1, 1, n)

            # Linear regression: y = at + b
            # a = Cov(t, y) / Var(t)
            t_mean = (n - 1) / 2.0
            t_var = (n**2 - 1) / 12.0 # Variance of [0, ..., n-1]

            y_window_mean = y_windows.mean(dim=-1, keepdim=True)

            covariance = ((t - t_mean) * (y_windows - y_window_mean)).mean(dim=-1, keepdim=True)
            a = covariance / t_var
            b = y_window_mean - a * t_mean

            y_fit = a * t + b

            # 4. RMS deviation in each window
            rms_sq = ((y_windows - y_fit)**2).mean(dim=-1) # (B, num_windows)

            # F(n) is the mean of rms across windows
            f_n = torch.sqrt(rms_sq.mean(dim=-1) + 1e-8) # (B,)
            fluctuations.append(f_n)

        fluctuations = torch.stack(fluctuations, dim=1) # (B, len(scales))

        output = []
        if self.include_fluctuations:
            output.append(fluctuations)

        if self.include_slope:
            # 5. Compute scaling exponent alpha (slope of log F(n) vs log n)
            log_n = torch.log(torch.tensor(self.scales, dtype=x.dtype, device=device))
            log_f = torch.log(fluctuations + 1e-8)

            # Linear regression on log-log plot
            log_n_mean = log_n.mean()
            log_n_var = ((log_n - log_n_mean)**2).mean()

            log_f_mean = log_f.mean(dim=-1, keepdim=True)
            covariance_log = ((log_n - log_n_mean) * (log_f - log_f_mean)).mean(dim=-1, keepdim=True)

            alpha = covariance_log / (log_n_var + 1e-8) # (B, 1)
            output.append(alpha)

        return torch.cat(output, dim=1)

class DFANet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, scales=[4, 8, 10, 20]):
        super().__init__()
        self.dfa = DifferentiableDFALayer(scales=scales)
        dfa_out_dim = len(scales) + 1
        self.mlp = nn.Sequential(
            nn.Linear(dfa_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = x.to(torch.float32)
        features = self.dfa(x)
        return self.mlp(features)

class DFAAugmentedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, scales=[4, 8, 10, 20]):
        super().__init__()
        self.dfa = DifferentiableDFALayer(scales=scales)
        dfa_out_dim = len(scales) + 1
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + dfa_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = x.to(torch.float32)
        features = self.dfa(x)
        combined = torch.cat([x, features], dim=-1)
        return self.mlp(combined)

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
        x = x.to(torch.float32)
        return self.mlp(x)
