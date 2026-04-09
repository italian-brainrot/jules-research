import torch
import torch.nn as nn
import torch.fft

class DifferentiableBispectrum(nn.Module):
    def __init__(self, n_input, use_magnitude=True, use_phase=False):
        super().__init__()
        self.n_input = n_input
        self.use_magnitude = use_magnitude
        self.use_phase = use_phase

        # Determine the non-redundant indices for the bispectrum of a real signal
        # 0 <= l <= k <= n_input // 2
        # k + l <= n_input // 2 (to stay within the first Nyquist zone)
        # Actually, for discrete signals, it can go further, but let's stick to the
        # standard non-redundant region for simplicity and to avoid aliasing issues.

        indices = []
        n_freq = n_input // 2 + 1
        for k in range(n_freq):
            for l in range(k + 1):
                if k + l < n_freq:
                    indices.append((k, l))

        self.register_buffer('indices_k', torch.tensor([i[0] for i in indices], dtype=torch.long))
        self.register_buffer('indices_l', torch.tensor([i[1] for i in indices], dtype=torch.long))
        self.register_buffer('indices_kl', torch.tensor([(i[0] + i[1]) for i in indices], dtype=torch.long))

        self.num_features = len(indices)
        if use_magnitude and use_phase:
            self.output_dim = self.num_features * 2
        else:
            self.output_dim = self.num_features

    def forward(self, x):
        # x: [batch, n_input] or [batch, 1, n_input]
        if x.dim() == 3:
            x = x.squeeze(1)

        X = torch.fft.rfft(x, n=self.n_input) # [batch, n_input // 2 + 1]

        X_k = X[:, self.indices_k]
        X_l = X[:, self.indices_l]
        X_kl = X[:, self.indices_kl]

        # B(k, l) = X(k) * X(l) * conj(X(k+l))
        bispectrum = X_k * X_l * torch.conj(X_kl)

        features = []
        if self.use_magnitude:
            features.append(torch.abs(bispectrum).to(x.dtype))
        if self.use_phase:
            features.append(torch.angle(bispectrum).to(x.dtype))

        return torch.cat(features, dim=-1)

class BispectrumMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10, use_magnitude=True, use_phase=False):
        super().__init__()
        self.bispectrum_layer = DifferentiableBispectrum(input_dim, use_magnitude, use_phase)
        bispectrum_dim = self.bispectrum_layer.output_dim

        self.mlp = nn.Sequential(
            nn.Linear(bispectrum_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # We can also concatenate with the original input if we want,
        # but let's see if bispectrum alone is sufficient.
        # Actually, let's concatenate for better performance.
        b_feat = self.bispectrum_layer(x)

        # Some models use bispectrum as an additional feature
        # x_flat = x.flatten(1)
        # feat = torch.cat([x_flat, b_feat], dim=1)
        # return self.mlp(feat)

        return self.mlp(b_feat)
