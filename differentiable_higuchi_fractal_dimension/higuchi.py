import torch
import torch.nn as nn
import numpy as np

class HiguchiFractalDimension(nn.Module):
    """
    Differentiable implementation of Higuchi Fractal Dimension features.
    Computes the curve lengths L(k) for a range of k values.
    """
    def __init__(self, k_max=10):
        super().__init__()
        self.k_max = k_max

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, N) or (B, 1, N)
        Returns:
            L_k: Curve lengths of shape (B, k_max)
        """
        if x.dim() == 3:
            x = x.squeeze(1)

        B, N = x.shape
        L_k = []

        for k in range(1, self.k_max + 1):
            # For each k, we construct k sub-series
            # L_m(k) = [ (sum_{i=1 to floor((N-m)/k)} |X(m+ik) - X(m+(i-1)k)|) * (N-1) ] / [ floor((N-m)/k) * k ]

            l_m_sums = []
            for m in range(1, k + 1):
                # Indices for sub-series
                # m-1, m-1+k, m-1+2k, ...
                indices = torch.arange(m - 1, N, k, device=x.device)
                if len(indices) < 2:
                    l_m_sums.append(torch.zeros(B, device=x.device))
                    continue

                sub_series = x[:, indices]
                # Absolute differences
                abs_diffs = torch.abs(sub_series[:, 1:] - sub_series[:, :-1])
                sum_diffs = torch.sum(abs_diffs, dim=1)

                num_steps = len(indices) - 1
                normalization = (N - 1) / (num_steps * k)
                l_m = sum_diffs * normalization / k
                l_m_sums.append(l_m)

            # L(k) is the average of L_m(k)
            l_k = torch.stack(l_m_sums, dim=1).mean(dim=1)
            L_k.append(l_k)

        return torch.stack(L_k, dim=1)

def compute_hfd(x, k_max=10):
    """
    Computes a single HFD value by fitting a line in log-log space.
    """
    B, N = x.shape
    hfd_layer = HiguchiFractalDimension(k_max=k_max)
    L_k = hfd_layer(x) # (B, k_max)

    k_values = torch.arange(1, k_max + 1, device=x.device, dtype=x.dtype)
    log_k = torch.log(1.0 / k_values)
    log_L = torch.log(L_k)

    # Simple linear regression for each batch item: log_L = D * log_k + c
    # We want D.
    # log_k: (k_max,)
    # log_L: (B, k_max)

    log_k_mean = log_k.mean()
    log_L_mean = log_L.mean(dim=1, keepdim=True)

    num = torch.sum((log_k - log_k_mean) * (log_L - log_L_mean), dim=1)
    den = torch.sum((log_k - log_k_mean)**2)

    return num / den
