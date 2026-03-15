import torch
import torch.nn as nn
import torch.nn.functional as F

class LogLinearInteractionLayer(nn.Module):
    """
    A layer that captures both additive and multiplicative interactions.
    Multiplicative interactions are modeled via the log-space identity:
    exp(sum(w_i * log|x_i|)) = product(|x_i|^w_i)
    """
    def __init__(self, in_features, out_features, epsilon=1e-5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.epsilon = epsilon

        # Additive path
        self.linear_add = nn.Linear(in_features, out_features)

        # Multiplicative path
        # 1. Project to a latent space in log-domain
        self.linear_log = nn.Linear(in_features, out_features)
        # 2. Project back from exp-domain
        self.linear_mult = nn.Linear(out_features, out_features)

    def forward(self, x):
        # Additive component
        out_add = self.linear_add(x)

        # Multiplicative component
        # We use log(|x| + eps) to handle negative inputs and avoid log(0)
        # Note: This loses the sign information of x, which out_add preserves.
        x_log = torch.log(torch.abs(x) + self.epsilon)
        x_mult_latent = self.linear_log(x_log)
        out_mult = self.linear_mult(torch.exp(x_mult_latent))

        return out_add + out_mult

class LLIN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, n_layers=2):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for _ in range(n_layers):
            layers.append(LogLinearInteractionLayer(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, n_layers=2):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
