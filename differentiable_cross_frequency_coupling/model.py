import torch
import torch.nn as nn
import torch.fft
import numpy as np

class DCFCLayer(nn.Module):
    """
    Differentiable Cross-Frequency Coupling (DCFC) Layer.
    Computes Phase-Amplitude Coupling (PAC) using learnable frequency filters.
    """
    def __init__(self, signal_len, num_pairs=8):
        super().__init__()
        self.signal_len = signal_len
        self.num_pairs = num_pairs

        # Learnable center frequencies (normalized 0 to 0.5)
        self.f_low = nn.Parameter(torch.rand(num_pairs) * 0.09 + 0.01)
        self.f_high = nn.Parameter(torch.rand(num_pairs) * 0.3 + 0.1)

        # Learnable bandwidths
        self.bw_low = nn.Parameter(torch.full((num_pairs,), 0.02))
        self.bw_high = nn.Parameter(torch.full((num_pairs,), 0.05))

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.to(torch.float32)

        X_full = torch.fft.fft(x, n=self.signal_len)
        full_freqs = torch.fft.fftfreq(self.signal_len, d=1.0).to(x.device)

        # Gaussian windows
        # Using a broader bandwidth or check filter overlap
        w_low = torch.exp(-0.5 * ((full_freqs.unsqueeze(0) - self.f_low.unsqueeze(1)) / self.bw_low.unsqueeze(1))**2)
        w_high = torch.exp(-0.5 * ((full_freqs.unsqueeze(0) - self.f_high.unsqueeze(1)) / self.bw_high.unsqueeze(1))**2)

        # Hilbert transform filter
        h = torch.zeros(self.signal_len, device=x.device)
        h[full_freqs > 0] = 2.0
        h[full_freqs == 0] = 1.0

        Z_low = torch.fft.ifft(X_full.unsqueeze(1) * w_low.unsqueeze(0) * h.unsqueeze(0).unsqueeze(0), n=self.signal_len)
        Z_high = torch.fft.ifft(X_full.unsqueeze(1) * w_high.unsqueeze(0) * h.unsqueeze(0).unsqueeze(0), n=self.signal_len)

        # Adding a small epsilon before angle/abs
        phi_low = torch.angle(Z_low + 1e-8)
        amp_high = torch.abs(Z_high)

        # MVL PAC
        coupling_complex = amp_high * torch.exp(1j * phi_low)
        mvl = torch.abs(torch.mean(coupling_complex, dim=2))

        mean_amp = torch.mean(amp_high, dim=2) + 1e-6
        pac = mvl / mean_amp

        return pac

class DCFCAugmentedMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10, num_pairs=8):
        super().__init__()
        self.dcfc = DCFCLayer(signal_len=input_dim, num_pairs=num_pairs)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + num_pairs, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        pac_features = self.dcfc(x)
        combined = torch.cat([x, pac_features], dim=1)
        return self.mlp(combined)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.mlp(x)
