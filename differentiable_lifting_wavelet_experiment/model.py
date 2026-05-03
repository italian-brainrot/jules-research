import torch
import torch.nn as nn

class LiftingLayer(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        # Predictor P: takes even samples, predicts odd samples
        self.P = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        # Updater U: takes residual (odd), updates even samples
        self.U = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)

        # Scaling factors for energy preservation (optional but common in wavelets)
        self.s = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # x: (B, 1, L)
        # 1. Split
        xe = x[:, :, 0::2]
        xo = x[:, :, 1::2]

        # 2. Predict
        d = xo - self.P(xe)

        # 3. Update
        a = xe + self.U(d)

        # 4. Scale
        a = a * self.s
        d = d / (self.s + 1e-6)

        return a, d

    def inverse(self, a, d):
        # 4. Descale
        a = a / self.s
        d = d * self.s

        # 3. Inverse Update
        xe = a - self.U(d)

        # 2. Inverse Predict
        xo = d + self.P(xe)

        # 1. Merge
        B, C, L_half = a.shape
        x = torch.zeros((B, C, L_half * 2), device=a.device, dtype=a.dtype)
        x[:, :, 0::2] = xe
        x[:, :, 1::2] = xo
        return x

class HaarWaveletLayer(nn.Module):
    """Fixed Haar Wavelet using lifting scheme."""
    def __init__(self):
        super().__init__()
        # Haar Predict: d = xo - xe (P=1)
        # Haar Update: a = xe + 0.5 * d (U=0.5)
        # Haar Scale: s = sqrt(2)
        # Note: Standard Haar is normalized by 1/sqrt(2) for filters,
        # but lifting can vary. Let's use the standard lifting form.
        pass

    def forward(self, x):
        xe = x[:, :, 0::2]
        xo = x[:, :, 1::2]
        d = xo - xe
        a = xe + 0.5 * d
        # Normalize to keep it orthonormal
        a = a * torch.sqrt(torch.tensor(2.0))
        d = d / torch.sqrt(torch.tensor(2.0))
        return a, d

class LLWNet(nn.Module):
    def __init__(self, input_dim=40, levels=2, kernel_size=3, hidden_dim=256, output_dim=10, use_learnable=True):
        super().__init__()
        self.levels = levels
        self.use_learnable = use_learnable
        if use_learnable:
            self.layers = nn.ModuleList([LiftingLayer(kernel_size) for _ in range(levels)])
        else:
            self.layers = nn.ModuleList([HaarWaveletLayer() for _ in range(levels)])

        # Calculate feature dimension after decomposition
        # Level 1: a (L/2), d1 (L/2)
        # Level 2: a (L/4), d2 (L/4), d1 (L/2)
        # Total = L
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (B, L)
        x = x.unsqueeze(1).to(torch.float32) # (B, 1, L)

        coeffs = []
        a = x
        for i in range(self.levels):
            a, d = self.layers[i](a)
            coeffs.append(d.flatten(1))
        coeffs.append(a.flatten(1))

        # Combine all coefficients
        feat = torch.cat(coeffs, dim=1)
        return self.mlp(feat)

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
        x = x.to(torch.float32)
        return self.mlp(x)
