import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def compute_pwm_weights(n, max_r):
    """
    Precompute weights for PWMs b_r = 1/n * sum(weights[j, r] * x_(j))
    weights[j, r] = comb(j-1, r) / comb(n-1, r)
    """
    weights = torch.zeros(n, max_r + 1)
    for r in range(max_r + 1):
        for j in range(1, n + 1):
            if j - 1 >= r:
                # Use log gamma for stability if n is large, but n=40 is small.
                # However, n might be larger in other use cases.
                # comb(j-1, r) / comb(n-1, r)
                # = [(j-1)! / (r! (j-1-r)!)] / [(n-1)! / (r! (n-1-r)!)]
                # = [(j-1)! (n-1-r)!] / [(j-1-r)! (n-1)!]
                val = math.exp(math.lgamma(j) + math.lgamma(n - r) - math.lgamma(j - r) - math.lgamma(n))
                weights[j-1, r] = val
    return weights

class LMoments1d(nn.Module):
    def __init__(self, max_order=4):
        super().__init__()
        self.max_order = max_order
        # Coefficients to convert PWMs (b0, b1, ..., b_{max_order-1}) to L-moments (L1, L2, ..., L_{max_order})
        # L_{r+1} = sum_{k=0}^r (-1)^{r-k} * comb(r, k) * comb(r+k, k) * b_k
        coeffs = torch.zeros(max_order, max_order)
        for r in range(max_order):
            for k in range(r + 1):
                val = ((-1)**(r - k)) * math.comb(r, k) * math.comb(r + k, k)
                coeffs[r, k] = val
        self.register_buffer('coeffs', coeffs)

    def forward(self, x):
        """
        x: (batch, channels, length) or (batch, length)
        Returns: (batch, channels, max_order)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch, channels, n = x.shape
        x_sorted, _ = torch.sort(x, dim=-1)

        weights = compute_pwm_weights(n, self.max_order - 1).to(x.device)

        # b_r: (batch, channels, max_order)
        # x_sorted: (batch, channels, n)
        # weights: (n, max_order)
        pwms = torch.matmul(x_sorted, weights) / n

        # l_moments: (batch, channels, max_order)
        l_moments = torch.matmul(pwms, self.coeffs.t())

        return l_moments

class LMomentSlidingWindow1d(nn.Module):
    def __init__(self, window_size, stride=None, max_order=4):
        super().__init__()
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        self.max_order = max_order
        self.l_moments_layer = LMoments1d(max_order)

    def forward(self, x):
        """
        x: (batch, channels, length)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # unfold: (batch, channels, num_windows, window_size)
        x_unfolded = x.unfold(-1, self.window_size, self.stride)
        batch, channels, num_windows, window_size = x_unfolded.shape

        # Reshape for LMoments1d: (batch * channels * num_windows, 1, window_size)
        x_reshaped = x_unfolded.reshape(-1, 1, window_size)

        l_moments = self.l_moments_layer(x_reshaped) # (B*C*W, 1, max_order)

        # Reshape back: (batch, channels, num_windows, max_order)
        l_moments = l_moments.view(batch, channels, num_windows, self.max_order)

        # Final shape: (batch, channels * num_windows * max_order) or similar
        return l_moments

class LMomentAugmentedMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10, max_order=4, window_size=20, stride=10):
        super().__init__()
        self.l_moments_sliding = LMomentSlidingWindow1d(window_size, stride, max_order)

        # Calculate number of L-moment features
        # MNIST-1D has length 40.
        # With window_size=20, stride=10, we have windows [0:20], [10:30], [20:40] -> 3 windows.
        # If channels=1, and max_order=4, we get 3 * 4 = 12 features.
        dummy_x = torch.zeros(1, 1, input_dim)
        dummy_l = self.l_moments_sliding(dummy_x)
        self.num_l_features = dummy_l.numel()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim + self.num_l_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if x.dim() == 2:
            x_in = x.unsqueeze(1)
        else:
            x_in = x

        l_features = self.l_moments_sliding(x_in)
        l_features = l_features.reshape(x.size(0), -1)

        combined = torch.cat([x.reshape(x.size(0), -1), l_features], dim=1)
        return self.mlp(combined)

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
        x = x.reshape(x.size(0), -1)
        return self.mlp(x)
