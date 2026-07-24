import torch
import torch.nn as nn
import torch.nn.functional as F

class ZeroOneChaosLayer(nn.Module):
    """
    Differentiable 0-1 Test for Chaos Layer.
    Computes the chaos statistic K for multiple learnable frequencies c.
    Uses the modified Mean Square Displacement D_c(n) to account for drift.
    """
    def __init__(self, num_freqs=8, n_max=10):
        super().__init__()
        self.num_freqs = num_freqs
        # Frequencies c should be in (0, pi). Standard choice is often in (0.1*pi, 0.9*pi)
        self.c = nn.Parameter(torch.rand(num_freqs) * 0.8 * torch.pi + 0.1 * torch.pi)
        self.n_max = n_max

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, L) or (B, 1, L)
        Returns:
            K: Chaos statistic for each frequency, shape (B, num_freqs)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, C, L = x.shape

        n_max = self.n_max if self.n_max is not None else L // 10
        n_max = max(2, min(n_max, L - 2)) # Ensure n_max is at least 2 for correlation

        # j = 1...L
        j = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, 1, 1, L)
        # c: (num_freqs,) -> (1, 1, num_freqs, 1)
        c = self.c.view(1, 1, self.num_freqs, 1)

        # x_expanded: (B, C, 1, L)
        x_expanded = x.unsqueeze(2)

        # phase: (1, 1, num_freqs, L)
        phase = j * c

        # Modulated signal: (B, C, num_freqs, L)
        real_part = x_expanded * torch.cos(phase)
        imag_part = x_expanded * torch.sin(phase)

        # Mean of x for drift correction: (B, C, 1, 1)
        x_mean = x.mean(dim=2, keepdim=True).unsqueeze(2)

        k_values = []
        for n in range(1, n_max + 1):
            # (B*C*num_freqs, 1, L)
            r = real_part.reshape(-1, 1, L)
            i = imag_part.reshape(-1, 1, L)

            # Sliding window sum of length n
            # F.avg_pool1d gives mean, multiply by n to get sum
            sum_r = F.avg_pool1d(r, kernel_size=n, stride=1) * n
            sum_i = F.avg_pool1d(i, kernel_size=n, stride=1) * n

            # M_c(n) = mean over j of [ (p(j+n)-p(j))^2 + (q(j+n)-q(j))^2 ]
            # (B*C*num_freqs, L-n+1)
            msd_n_per_j = sum_r.squeeze(1)**2 + sum_i.squeeze(1)**2
            msd_n = msd_n_per_j.mean(dim=1) # (B*C*num_freqs,)

            # Drift correction: V_osc = (E[x])^2 * (1 - cos(nc)) / (1 - cos(c))
            xm = x_mean.expand(B, C, self.num_freqs, 1).reshape(-1)
            cur_c = c.expand(B, C, self.num_freqs, 1).reshape(-1)

            v_osc = xm**2 * (1 - torch.cos(torch.tensor(n, device=x.device, dtype=x.dtype) * cur_c)) / (1 - torch.cos(cur_c) + 1e-8)

            d_n = msd_n - v_osc
            k_values.append(d_n)

        # d_msd: (B*C*num_freqs, n_max)
        d_msd = torch.stack(k_values, dim=1)

        # Correlation between n (1...n_max) and d_msd
        n_axis = torch.arange(1, n_max + 1, device=x.device, dtype=x.dtype)

        # Pearson correlation
        x_vec = n_axis
        y_vec = d_msd

        x_mu = x_vec.mean()
        y_mu = y_vec.mean(dim=1, keepdim=True)
        x_std = x_vec.std()
        y_std = y_vec.std(dim=1, keepdim=True)

        cov = ((x_vec - x_mu) * (y_vec - y_mu)).mean(dim=1)
        # Use squeeze(1) to avoid broadcasting errors
        corr = cov / (x_std * y_std.squeeze(1) + 1e-8)

        # corr: (B*C*num_freqs,) -> (B, C * self.num_freqs)
        return corr.view(B, C * self.num_freqs)

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

class ChaosAugmentedMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10, num_freqs=8):
        super().__init__()
        self.chaos_layer = ZeroOneChaosLayer(num_freqs=num_freqs)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + num_freqs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if x.dim() == 2:
            x_raw = x
            x_in = x.unsqueeze(1)
        else:
            x_raw = x.squeeze(1)
            x_in = x

        k_features = self.chaos_layer(x_in)
        combined = torch.cat([x_raw, k_features], dim=1)
        return self.mlp(combined)
