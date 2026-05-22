import torch
import torch.nn as nn
import torch.fft

class DifferentiableCepstrum(nn.Module):
    """
    Differentiable Real Cepstrum Layer.
    Computes the real cepstrum of a 1D signal: IDFT(log(|DFT(x)| + eps)).
    Uses a manual DFT matrix for maximum stability in environments with MKL issues.
    """
    def __init__(self, n=40, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.n = n

        # Precompute DFT matrix: W_jk = exp(-2pi i j k / n)
        j = torch.arange(n).reshape(n, 1)
        k = torch.arange(n).reshape(1, n)
        W = torch.exp(-2j * torch.pi * j * k / n)
        self.register_buffer('W', W)

        # IDFT matrix: (1/n) * exp(2pi i j k / n)
        W_inv = torch.exp(2j * torch.pi * j * k / n) / n
        self.register_buffer('W_inv', W_inv)

    def forward(self, x):
        # x shape: (batch, n)
        x = x.to(torch.float32)
        batch_size = x.shape[0]

        # 1. Compute the FFT via matrix multiplication
        # x is real, so we cast to complex
        x_complex = x.to(torch.complex64)
        X = torch.matmul(x_complex, self.W.T)

        # 2. Compute the log-magnitude
        log_mag = torch.log(torch.abs(X) + self.eps)

        # 3. Compute the Inverse FFT
        # log_mag is real
        cepstrum_complex = torch.matmul(log_mag.to(torch.complex64), self.W_inv.T)
        cepstrum = cepstrum_complex.real

        return cepstrum

class CepstrumNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.cepstrum_layer = DifferentiableCepstrum(n=input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        c = self.cepstrum_layer(x)
        return self.mlp(c)

class CepstrumAugmentedNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.cepstrum_layer = DifferentiableCepstrum(n=input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        c = self.cepstrum_layer(x)
        combined = torch.cat([x, c], dim=-1)
        return self.mlp(combined)

class BaselineNet(nn.Module):
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
        return self.mlp(x)
