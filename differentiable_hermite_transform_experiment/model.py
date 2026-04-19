import torch
import torch.nn as nn
import math

class HermiteTransformLayer(nn.Module):
    def __init__(self, input_dim, n_coeffs, initial_scale=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.n_coeffs = n_coeffs
        self.log_scale = nn.Parameter(torch.tensor([math.log(initial_scale)]))

        # Precompute the grid
        # We assume the signal is centered at 0 and spans from -L/2 to L/2
        grid = torch.linspace(-1, 1, input_dim)
        self.register_buffer('grid', grid)

    def get_hermite_functions(self):
        scale = torch.exp(self.log_scale)
        x = self.grid / scale

        psi = []

        # H0(x) = 1
        # H1(x) = 2x
        # H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)

        h_prev = torch.ones_like(x)
        h_curr = 2 * x

        # Psi_n(x) = (2^n n! sqrt(pi))^-0.5 * exp(-x^2/2) * H_n(x)

        gaussian = torch.exp(-0.5 * x**2)
        norm0 = (math.pi**0.25)
        psi.append(gaussian * h_prev / norm0)

        if self.n_coeffs > 1:
            norm1 = (2.0 * math.sqrt(2.0) * (math.pi**0.25)) # Wait, norm_n = sqrt(2^n * n! * sqrt(pi))
            # Actually norm_n = sqrt(2^n * n!) * pi^0.25
            norm1 = math.sqrt(2.0) * (math.pi**0.25)
            psi.append(gaussian * h_curr / norm1)

            for n in range(1, self.n_coeffs - 1):
                h_next = 2 * x * h_curr - 2 * n * h_prev
                h_prev = h_curr
                h_curr = h_next

                norm_n_plus_1 = math.sqrt(2**(n+1) * math.factorial(n+1)) * (math.pi**0.25)
                psi.append(gaussian * h_curr / norm_n_plus_1)

        return torch.stack(psi, dim=0) # (n_coeffs, input_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        psi = self.get_hermite_functions() # (n_coeffs, input_dim)
        # Compute coefficients: sum_i x_i * psi_n(i)
        # We can use matrix multiplication
        coeffs = torch.matmul(x, psi.t()) # (batch_size, n_coeffs)
        return coeffs

class HermiteAugmentedMLP(nn.Module):
    def __init__(self, input_dim, n_coeffs, hidden_dim, output_dim):
        super().__init__()
        self.hermite = HermiteTransformLayer(input_dim, n_coeffs)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + n_coeffs, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # Ensure x is float32
        x = x.to(torch.float32)
        h_features = self.hermite(x)
        combined = torch.cat([x, h_features], dim=1)
        return self.mlp(combined)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
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
        x = x.to(torch.float32)
        return self.mlp(x)
