import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_threshold(x, threshold):
    return torch.sign(x) * torch.relu(torch.abs(x) - threshold)

class ISTALayer(nn.Module):
    def __init__(self, input_dim, sparse_dim, num_iterations=10):
        super().__init__()
        self.input_dim = input_dim
        self.sparse_dim = sparse_dim
        self.num_iterations = num_iterations

        # Learnable dictionary
        self.W = nn.Parameter(torch.randn(input_dim, sparse_dim) * (1.0 / (input_dim ** 0.5)))

        # Learnable step size and regularization parameter (in log space for positivity)
        self.log_eta = nn.Parameter(torch.tensor(-2.0))  # Initial eta = exp(-2) approx 0.13
        self.log_lambda = nn.Parameter(torch.tensor(-4.0)) # Initial lambda = exp(-4) approx 0.018

    def forward(self, x):
        batch_size = x.shape[0]
        eta = torch.exp(self.log_eta)
        lmbda = torch.exp(self.log_lambda)

        # Initial sparse code z
        z = torch.zeros(batch_size, self.sparse_dim, device=x.device)

        # WT = W^T
        WT = self.W.t()

        for _ in range(self.num_iterations):
            # Gradient of 0.5 * ||x - Wz||^2 is W^T (Wz - x)
            grad = F.linear(F.linear(z, self.W) - x, WT)
            z = soft_threshold(z - eta * grad, eta * lmbda)

        return z

class ISTANet(nn.Module):
    def __init__(self, input_dim, sparse_dim, num_iterations, hidden_dim, output_dim):
        super().__init__()
        self.ista = ISTALayer(input_dim, sparse_dim, num_iterations)
        self.classifier = nn.Sequential(
            nn.Linear(sparse_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        z = self.ista(x)
        return self.classifier(z)

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
