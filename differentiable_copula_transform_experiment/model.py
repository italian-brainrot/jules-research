import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableCopulaLayer(nn.Module):
    """
    A layer that performs a differentiable copula transformation on the input.
    It maps each feature to its empirical marginal CDF (uniform distribution)
    and optionally to a normal distribution.
    """
    def __init__(self, num_features, initial_alpha=10.0, learnable_alpha=True, output_type='gaussian'):
        super().__init__()
        self.num_features = num_features
        self.output_type = output_type # 'uniform' or 'gaussian'

        if learnable_alpha:
            self.alpha = nn.Parameter(torch.full((num_features,), float(initial_alpha)))
        else:
            self.register_buffer('alpha', torch.full((num_features,), float(initial_alpha)))

    def forward(self, x):
        # x shape: (batch_size, num_features)
        # We compute the soft empirical CDF using sigmoids.
        # For each element x_i, CDF(x_i) = mean(sigmoid(alpha * (x_i - x_j)))

        # x_i: (batch_size, 1, num_features)
        # x_j: (1, batch_size, num_features)
        diff = x.unsqueeze(1) - x.unsqueeze(0)

        # scaled_diff: (batch_size, batch_size, num_features)
        scaled_diff = diff * self.alpha.view(1, 1, -1)

        # soft_ranks: (batch_size, batch_size, num_features)
        soft_ranks = torch.sigmoid(scaled_diff)

        # empirical_cdf: (batch_size, num_features)
        # We average over the second dimension (x_j)
        # This gives values in [0, 1]
        u = soft_ranks.mean(dim=1)

        if self.output_type == 'uniform':
            return u
        elif self.output_type == 'gaussian':
            # Map uniform [0, 1] to Gaussian using inverse CDF (probit)
            # Standard Normal Inverse CDF: sqrt(2) * erfinv(2u - 1)
            # Clamp u to avoid +/- infinity
            u_clamped = torch.clamp(u, 1e-7, 1 - 1e-7)
            gaussian_margins = 1.41421356 * torch.erfinv(2 * u_clamped - 1)
            return gaussian_margins
        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")

class CopulaAugmentedMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10, output_type='gaussian'):
        super().__init__()
        self.copula = DifferentiableCopulaLayer(input_dim, output_type=output_type)
        # Concatenate original features and copula transformed features
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        u = self.copula(x)
        combined = torch.cat([x, u], dim=1)
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
        return self.mlp(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    x = torch.randn(32, 40)
    model = CopulaAugmentedMLP()
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {count_parameters(model)}")

    baseline = BaselineMLP()
    print(f"Baseline parameters: {count_parameters(baseline)}")
