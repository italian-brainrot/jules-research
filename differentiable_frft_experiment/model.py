import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FrFTLayer(nn.Module):
    """
    Differentiable Fractional Fourier Transform layer based on the eigendecomposition
    of the Discrete Fourier Transform matrix.
    """
    def __init__(self, n, num_orders=4):
        super().__init__()
        self.n = n
        self.num_orders = num_orders
        # Learnable fractional orders, initialized uniformly between 0 and 1
        self.alphas = nn.Parameter(torch.linspace(0, 1, num_orders))

        # Precompute DFT matrix and its powers
        # F_mn = (1/sqrt(n)) * exp(-2pi * i * m * n / n)
        m_idx = torch.arange(n).reshape(-1, 1)
        n_idx = torch.arange(n).reshape(1, -1)
        F_mat = torch.exp(-2j * np.pi * m_idx * n_idx / n) / np.sqrt(n)

        F2 = F_mat @ F_mat
        F3 = F2 @ F_mat
        F0 = torch.eye(n, dtype=torch.complex64)

        # Projectors P_k onto the eigenspaces of F
        # P_k = 1/4 * sum_{j=0}^3 exp(i * pi * k * j / 2) * F^j
        # these project onto the eigenspace corresponding to eigenvalue exp(-i * pi * k / 2)
        P = torch.zeros((4, n, n), dtype=torch.complex64)
        for k in range(4):
            for j in range(4):
                val = torch.exp(torch.tensor(1j * np.pi * k * j / 2, dtype=torch.complex64))
                if j == 0: mat = F0
                elif j == 1: mat = F_mat
                elif j == 2: mat = F2
                else: mat = F3
                P[k] += (val * mat) / 4.0

        self.register_buffer('P', P)

    def forward(self, x):
        # x: (batch, n)
        batch_size = x.shape[0]
        x_complex = x.to(torch.complex64)

        # Decompose x into components in each eigenspace
        # x_k = P_k @ x
        # x_k shape: (4, batch, n)
        # Using matmul: (4, n, n) @ (batch, n, 1) -> (4, batch, n)
        x_k = torch.matmul(self.P.unsqueeze(1), x_complex.unsqueeze(0).unsqueeze(-1)).squeeze(-1)

        # alphas: (num_orders)
        # We want to compute: x_alpha = sum_{k=0}^3 exp(-i * pi * k * alpha / 2) * x_k
        k_indices = torch.arange(4, device=x.device).reshape(4, 1, 1)
        alphas = self.alphas.reshape(1, self.num_orders, 1)

        # phases: (4, num_orders, 1)
        phases = torch.exp(-1j * np.pi * k_indices.to(torch.complex64) * alphas.to(torch.complex64) / 2)

        # x_alpha: (4, num_orders, batch, n)
        # x_k: (4, batch, n)
        # broadcast and multiply
        x_alpha_comp = x_k.unsqueeze(1) * phases.unsqueeze(2)
        x_alpha = torch.sum(x_alpha_comp, dim=0) # (num_orders, batch, n)

        # Transpose to (batch, num_orders, n)
        x_alpha = x_alpha.transpose(0, 1)

        # Use magnitude as features
        features = torch.abs(x_alpha) # (batch, num_orders, n)
        return features.reshape(batch_size, -1)

class FrFTAugmentedMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10, num_orders=4):
        super().__init__()
        self.frft = FrFTLayer(input_dim, num_orders=num_orders)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + input_dim * num_orders, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        frft_features = self.frft(x)
        combined = torch.cat([x, frft_features], dim=1)
        return self.mlp(combined)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=320, output_dim=10):
        super().__init__()
        # hidden_dim=320 roughly matches FrFTAugmentedMLP(hidden_dim=256, num_orders=4)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    frft_mlp = FrFTAugmentedMLP()
    base_mlp = BaselineMLP()
    print(f"FrFTAugmentedMLP parameters: {count_parameters(frft_mlp)}")
    print(f"BaselineMLP parameters: {count_parameters(base_mlp)}")
