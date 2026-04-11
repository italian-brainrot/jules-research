import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class DifferentiableHilbertLayer(nn.Module):
    """
    Computes the analytic signal, instantaneous envelope, and instantaneous frequency of a 1D signal.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        """
        x shape: (batch, channels, length)
        """
        batch_size, channels, length = x.shape
        if length < 2:
            return x.repeat(1, 3, 1)

        # 1. FFT
        X = torch.fft.fft(x, dim=-1)

        # 2. Construct the Hilbert mask in frequency domain
        # H[0] = 1
        # H[1:N/2] = 2
        # H[N/2] = 1 (if N is even)
        # H[N/2+1:] = 0
        h = torch.zeros(length, device=x.device, dtype=X.dtype)
        if length % 2 == 0:
            h[0] = 1
            h[1:length//2] = 2
            h[length//2] = 1
        else:
            h[0] = 1
            h[1:(length+1)//2] = 2

        # Broadcast h to X shape
        # X: (batch, channels, length)
        # h: (length)
        Z_freq = X * h

        # 3. IFFT to get the analytic signal
        z = torch.fft.ifft(Z_freq, dim=-1)

        # 4. Instantaneous Envelope
        envelope = torch.abs(z)

        # 5. Instantaneous Frequency
        # Approximation: phase difference between consecutive samples
        # angle(z[n] * conj(z[n-1]))
        z_conj = torch.conj(z)
        # Shift z to get z[n-1]
        # We can use roll or just slice. Slice is better to avoid wrap-around at the boundary
        # for a non-periodic signal, but mnist1d might be considered periodic or we can just pad.
        # Let's use padding to keep the same length.
        z_shifted = F.pad(z[:, :, :-1], (1, 0), mode='replicate')

        # freq = angle(z * conj(z_shifted))
        # Note: torch.angle returns values in (-pi, pi]
        frequency = torch.angle(z * torch.conj(z_shifted))

        # Concatenate original signal, envelope, and frequency
        # Output shape: (batch, 3 * channels, length)
        return torch.cat([x, envelope, frequency], dim=1)

class HilbertMLP(nn.Module):
    def __init__(self, input_dim=40, in_channels=1, hidden_dim=256, output_dim=10):
        super().__init__()
        self.hilbert = DifferentiableHilbertLayer()
        # Hilbert layer outputs 3 features per input channel: original, envelope, frequency
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * in_channels * 3, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x shape: (batch, 40)
        if x.dim() == 2:
            x = x.unsqueeze(1) # (batch, 1, 40)

        x = self.hilbert(x) # (batch, 3, 40)
        x = x.view(x.size(0), -1)
        return self.mlp(x)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, in_channels=1, hidden_dim=256, output_dim=10):
        super().__init__()
        # To match parameter count roughly, we can use a slightly larger hidden_dim if needed,
        # but let's first keep them similar as the Hilbert layer has no learnable parameters
        # except for what's in the MLP.
        # HilbertMLP has input_dim * 3 -> hidden_dim
        # BaselineMLP has input_dim -> hidden_dim
        # We should adjust hidden_dim or keep it same to see the effect of features.
        # Let's keep hidden_dim the same for now, or maybe make BaselineMLP wider
        # to match the number of parameters in the first layer.

        # HilbertMLP first layer: (40*3) * 256 + 256 = 120 * 256 + 256 = 30720 + 256 = 30976
        # BaselineMLP first layer: 40 * W + W = 41 * W
        # 41 * W = 30976 => W approx 755

        # Actually, let's just use the same hidden_dim and see if the features help.
        # If the features are good, they should help even with same hidden_dim.
        # But for fairness, we could match parameters.

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

if __name__ == "__main__":
    layer = DifferentiableHilbertLayer()
    x = torch.randn(2, 1, 40)
    y = layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Check if gradients flow
    x.requires_grad = True
    y = layer(x)
    y.sum().backward()
    print(f"Gradient: {x.grad is not None}")
