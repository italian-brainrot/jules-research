import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

class DifferentiableSTLayer(nn.Module):
    """
    Computes the S-Transform of a 1D signal.
    The output is a complex-valued time-frequency representation.
    """
    def __init__(self, n_fft=None, sigma=1.0, learnable_sigma=False):
        super().__init__()
        if learnable_sigma:
            self.sigma = nn.Parameter(torch.tensor(float(sigma)))
        else:
            self.register_buffer("sigma", torch.tensor(float(sigma)))
        self.n_fft = n_fft

    def forward(self, x):
        """
        x shape: (batch, channels, length)
        Returns: (batch, channels, freq, time) complex tensor
        """
        B, C, N = x.shape
        if self.n_fft is None:
            n_fft = N
        else:
            n_fft = self.n_fft
            if n_fft > N:
                x = F.pad(x, (0, n_fft - N))
                N = n_fft
            elif n_fft < N:
                x = x[:, :, :n_fft]
                N = n_fft

        device = x.device
        X = torch.fft.fft(x, dim=-1) # (B, C, N)

        # n: frequency index, 1 to N-1
        n = torch.arange(1, N, device=device, dtype=torch.float32).view(1, 1, N-1, 1)
        # m: frequency offset index
        m = torch.arange(N, device=device, dtype=torch.float32).view(1, 1, 1, N)
        # center m to handle periodic convolution correctly
        m = torch.where(m <= N // 2, m, m - N)

        # G: (1, 1, N-1, N)
        # sigma can be learnable
        G = torch.exp(-2 * (torch.pi ** 2) * (m ** 2) * (self.sigma ** 2) / (n ** 2))

        # X_shifted[b, c, n-1, m] = X[b, c, (n+m) % N]
        n_idx = torch.arange(1, N, device=device).view(N-1, 1)
        m_idx = torch.arange(N, device=device).view(1, N)
        indices = (n_idx + m_idx) % N
        X_shifted = X[:, :, indices] # (B, C, N-1, N)

        # Multiply and IFFT along the m dimension
        st_nonzero = torch.fft.ifft(X_shifted * G, dim=-1) # (B, C, N-1, N)

        # n=0 case: S[j, 0] is the DC component (average of signal)
        # S[j, 0] = mean(x) for all j
        st_zero = torch.mean(x, dim=-1, keepdim=True).unsqueeze(2).expand(B, C, 1, N)

        st = torch.cat([st_zero.to(st_nonzero.dtype), st_nonzero], dim=2)
        return st

class STAugmentedMLP(nn.Module):
    def __init__(self, input_dim=40, in_channels=1, hidden_dim=256, output_dim=10, learnable_sigma=True):
        super().__init__()
        self.st_layer = DifferentiableSTLayer(sigma=1.0, learnable_sigma=learnable_sigma)
        # ST features: magnitude of S-transform. (B, C, N, N)
        # For mnist1d, N=40. flattened ST is 40*40 = 1600.
        st_feature_dim = in_channels * input_dim * input_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim * in_channels + st_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (B, 40)
        if x.dim() == 2:
            x = x.unsqueeze(1) # (B, 1, 40)

        st = self.st_layer(x) # (B, 1, 40, 40)
        st_mag = torch.abs(st).view(x.size(0), -1)

        x_flat = x.view(x.size(0), -1)
        combined = torch.cat([x_flat, st_mag], dim=1)

        return self.mlp(combined)

class STMLP(nn.Module):
    def __init__(self, input_dim=40, in_channels=1, hidden_dim=256, output_dim=10, learnable_sigma=True):
        super().__init__()
        self.st_layer = DifferentiableSTLayer(sigma=1.0, learnable_sigma=learnable_sigma)
        st_feature_dim = in_channels * input_dim * input_dim

        self.mlp = nn.Sequential(
            nn.Linear(st_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        st = self.st_layer(x)
        st_mag = torch.abs(st).view(x.size(0), -1)
        return self.mlp(st_mag)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, in_channels=1, hidden_dim=256, output_dim=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * in_channels, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        return self.mlp(x)
