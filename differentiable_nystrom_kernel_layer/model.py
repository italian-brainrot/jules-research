import torch
import torch.nn as nn
import torch.nn.functional as F

class NystromKernelLayer(nn.Module):
    def __init__(self, input_dim, num_landmarks, gamma=1.0, trainable_landmarks=True, trainable_gamma=True):
        super().__init__()
        self.input_dim = input_dim
        self.num_landmarks = num_landmarks

        # Initialize landmarks with random normal
        self.landmarks = nn.Parameter(torch.randn(num_landmarks, input_dim), requires_grad=trainable_landmarks)

        # Gamma parameter for RBF kernel: exp(-gamma * ||x-y||^2)
        self.log_gamma = nn.Parameter(torch.log(torch.tensor([gamma])), requires_grad=trainable_gamma)

    def _rbf_kernel(self, x, y):
        # x: (batch, dim) or (n, dim)
        # y: (m, dim)
        # Result: (batch, m) or (n, m)

        x_norm = (x**2).sum(dim=-1, keepdim=True)
        y_norm = (y**2).sum(dim=-1, keepdim=True)
        dist_sq = x_norm + y_norm.transpose(-2, -1) - 2 * torch.matmul(x, y.transpose(-2, -1))
        dist_sq = torch.clamp(dist_sq, min=0.0)

        return torch.exp(-torch.exp(self.log_gamma) * dist_sq)

    def forward(self, x):
        # x: (batch, input_dim)
        K_xC = self._rbf_kernel(x, self.landmarks) # (batch, num_landmarks)
        K_CC = self._rbf_kernel(self.landmarks, self.landmarks) # (num_landmarks, num_landmarks)

        # Add a small ridge for numerical stability
        K_CC = K_CC + torch.eye(self.num_landmarks, device=x.device) * 1e-4

        # Compute K_CC^{-1/2} using eigendecomposition
        # K_CC is symmetric positive definite
        try:
            L, V = torch.linalg.eigh(K_CC)
            # L contains eigenvalues, V contains eigenvectors
            # Clamp eigenvalues to be positive
            L = torch.clamp(L, min=1e-6)
            K_CC_inv_sqrt = V @ torch.diag(L**-0.5) @ V.transpose(-2, -1)
        except torch._C._LinAlgError:
            # Fallback if eigh fails
            K_CC_inv_sqrt = torch.eye(self.num_landmarks, device=x.device)

        # Resulting features: K_xC @ K_CC^{-1/2}
        phi_x = K_xC @ K_CC_inv_sqrt # (batch, num_landmarks)

        return phi_x

class NystromMLP(nn.Module):
    def __init__(self, input_dim, num_landmarks, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.nystrom = NystromKernelLayer(input_dim, num_landmarks)

        layers = []
        in_dim = num_landmarks
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        phi_x = self.nystrom(x)
        return self.mlp(phi_x)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
