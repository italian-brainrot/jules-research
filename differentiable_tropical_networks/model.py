import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TropicalLinear(nn.Module):
    """
    Tropical linear layer using a soft-max (logsumexp) over (x + W).
    Smoothly interpolates between max (large positive beta) and mean (small beta).
    y = (1/beta) * (logsumexp(beta * (x + W)) - log(I))
    where I is the number of input features.
    """
    def __init__(self, in_features, out_features, init_beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weights act as additive offsets in the tropical semiring (x + W)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        # Beta controls the smoothness
        self.beta = nn.Parameter(torch.full((out_features,), float(init_beta)))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # x shape: (batch_size, in_features)

        # Expand x and weight to (batch_size, out_features, in_features)
        x_expanded = x.unsqueeze(1)  # (batch_size, 1, in_features)
        w_expanded = self.weight.unsqueeze(0)  # (1, out_features, in_features)

        # Inner sum (x + W)
        combined = x_expanded + w_expanded  # (batch_size, out_features, in_features)

        # To handle beta near 0, we can use a Taylor expansion or just a small epsilon
        # However, subtraction of log(in_features) is key for the mean property.

        beta = self.beta.view(1, -1, 1) # (1, out_features, 1)

        # For small beta, we use Taylor expansion of logsumexp to avoid numerical instability
        # logsumexp(beta * z) approx log(I) + beta * mean(z) + 0.5 * beta^2 * var(z)
        # (logsumexp(beta * z) - log(I)) / beta approx mean(z)

        eps = 1e-4
        is_small = beta.abs() < eps

        # Standard path
        safe_beta = torch.where(is_small, torch.ones_like(beta), beta)
        scaled = combined * safe_beta
        lse = torch.logsumexp(scaled, dim=2)
        res_standard = (lse - math.log(self.in_features)) / safe_beta.squeeze(-1)

        # Small beta path (mean)
        res_small = combined.mean(dim=2)

        res = torch.where(is_small.squeeze(-1), res_small, res_standard)

        return res

class TropicalMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, init_beta=1.0):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for i in range(num_layers - 1):
            layers.append(TropicalLinear(curr_dim, hidden_dim, init_beta=init_beta))
            layers.append(nn.BatchNorm1d(hidden_dim))
            curr_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(curr_dim, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class TropicalAugmentedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, init_beta=1.0):
        super().__init__()
        self.tropical = TropicalLinear(input_dim, hidden_dim, init_beta=init_beta)

        # Combine raw features and tropical features
        self.mlp = BaselineMLP(input_dim + hidden_dim, hidden_dim, output_dim, num_layers=num_layers)

    def forward(self, x):
        t_feat = self.tropical(x)
        combined = torch.cat([x, t_feat], dim=1)
        return self.mlp(combined)
