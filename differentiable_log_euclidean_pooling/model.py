import torch
import torch.nn as nn
import torch.nn.functional as F

class LogEuclideanPooling(nn.Module):
    """
    Differentiable Log-Euclidean Pooling layer.
    Computes the covariance matrix of the input channels,
    applies matrix logarithm, and extracts the unique elements.
    """
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x shape: (B, C, L)
        batch_size, channels, length = x.shape

        # Compute mean and center x
        mean = x.mean(dim=2, keepdim=True)
        x_centered = x - mean

        # Compute covariance matrix: (B, C, C)
        # We use 1/(L-1) for unbiased estimate if L > 1
        denom = length - 1 if length > 1 else 1
        cov = torch.bmm(x_centered, x_centered.transpose(1, 2)) / denom

        # Regularization to ensure SPD
        # Adding a small relative epsilon based on the trace or max value
        # to handle various signal scales
        trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        rel_eps = self.eps * trace / channels

        identity = torch.eye(channels, device=x.device, dtype=x.dtype).unsqueeze(0)
        cov = cov + rel_eps * identity + 1e-8 * identity

        # Matrix Logarithm via Eigendecomposition
        # Use double precision for stability in eigendecomposition
        cov_64 = cov.to(torch.float64)
        try:
            L, U = torch.linalg.eigh(cov_64)
        except RuntimeError:
            # Fallback to a slightly more regularized version if it fails
            cov_64 = cov_64 + 1e-4 * torch.eye(channels, device=x.device, dtype=torch.float64).unsqueeze(0)
            L, U = torch.linalg.eigh(cov_64)

        # Ensure eigenvalues are positive before log
        L = torch.clamp(L, min=self.eps)
        log_L = torch.log(L)

        log_cov_64 = torch.bmm(torch.bmm(U, torch.diag_embed(log_L)), U.transpose(1, 2))
        log_cov = log_cov_64.to(x.dtype)

        # Extract unique elements (upper triangle)
        # Using a mask for tril/triu
        triu_indices = torch.triu_indices(channels, channels, offset=0)
        out = log_cov[:, triu_indices[0], triu_indices[1]]

        return out

class DLEPModel(nn.Module):
    def __init__(self, input_dim=40, num_channels=16, hidden_dim=64, num_classes=10):
        super().__init__()
        # Initial 1D convolution to create channels
        # Kernel size 3 to capture local dependencies
        self.conv = nn.Conv1d(1, num_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(num_channels)
        self.dlep = LogEuclideanPooling()

        # Number of unique elements in num_channels x num_channels symmetric matrix
        # is C*(C+1)/2
        dlep_out_dim = num_channels * (num_channels + 1) // 2

        self.classifier = nn.Sequential(
            nn.Linear(dlep_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x shape: (B, L)
        x = x.unsqueeze(1) # (B, 1, L)
        x = F.relu(self.bn(self.conv(x)))
        x = self.dlep(x)
        x = self.classifier(x)
        return x

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)
