import torch
import torch.nn as nn
import torch.nn.functional as F

class DOMPLayer(nn.Module):
    """
    Differentiable "Soft" Matching Pursuit Layer.
    It iteratively selects atoms from a dictionary that are most correlated with the residual.
    """
    def __init__(self, input_dim, dict_size, num_iterations=10, beta=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.num_iterations = num_iterations

        # Learnable dictionary
        self.dictionary = nn.Parameter(torch.randn(input_dim, dict_size) / (input_dim ** 0.5))

        # Softness parameter (temperature)
        self.log_beta = nn.Parameter(torch.log(torch.tensor(beta)))

    def forward(self, x):
        # x: (batch, input_dim)
        batch_size = x.shape[0]
        device = x.device

        # Normalize dictionary atoms
        D = F.normalize(self.dictionary, p=2, dim=0)

        residual = x
        coeffs = torch.zeros(batch_size, self.dict_size, device=device)
        beta = torch.exp(self.log_beta)

        for _ in range(self.num_iterations):
            # Correlation between residual and atoms
            # (batch, input_dim) @ (input_dim, dict_size) -> (batch, dict_size)
            corr = torch.matmul(residual, D)

            # Soft selection of atoms
            # We use absolute correlation because we can have negative coefficients
            abs_corr = torch.abs(corr)
            weights = F.softmax(beta * abs_corr, dim=1)

            # Update coefficients: add to the coefficient of the "selected" atoms
            # We add the correlation value weighted by the soft selection
            # This is a bit like a soft version of MP
            delta_coeffs = weights * corr
            coeffs = coeffs + delta_coeffs

            # Update residual
            # (batch, dict_size) @ (dict_size, input_dim) -> (batch, input_dim)
            reconstruction = torch.matmul(coeffs, D.t())
            residual = x - reconstruction

        return coeffs

class DOMPNet(nn.Module):
    def __init__(self, input_dim, dict_size, num_iterations, hidden_dim, output_dim):
        super().__init__()
        self.domp = DOMPLayer(input_dim, dict_size, num_iterations)
        self.classifier = nn.Sequential(
            nn.Linear(dict_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        z = self.domp(x)
        return self.classifier(z)

class DOMPAugmentedMLP(nn.Module):
    def __init__(self, input_dim, dict_size, num_iterations, hidden_dim, output_dim):
        super().__init__()
        self.domp = DOMPLayer(input_dim, dict_size, num_iterations)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + dict_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        z = self.domp(x)
        combined = torch.cat([x, z], dim=1)
        return self.mlp(combined)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
