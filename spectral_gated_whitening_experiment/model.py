import torch
import torch.nn as nn
import torch.nn.functional as F

class ZCAWhitening(nn.Module):
    def __init__(self, train_x, eps=1e-6):
        super().__init__()
        mean = train_x.mean(dim=0)
        x_centered = train_x - mean
        cov = (x_centered.T @ x_centered) / (train_x.shape[0] - 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)

        self.register_buffer("mean", mean)
        self.register_buffer("eigvals", eigvals)
        self.register_buffer("eigvecs", eigvecs)
        self.eps = eps

    def forward(self, x):
        # x: (B, D)
        x_centered = x - self.mean
        x_spec = x_centered @ self.eigvecs
        # ZCA whitening: scale by 1/sqrt(lambda)
        x_white_spec = x_spec / torch.sqrt(self.eigvals + self.eps)
        # Transform back to original space
        x_out = x_white_spec @ self.eigvecs.T
        return x_out

class SoftPCAWhitening(nn.Module):
    """Fixed Wiener-like gate: s = lambda / (lambda + eta)"""
    def __init__(self, train_x, eta=0.1, eps=1e-6):
        super().__init__()
        mean = train_x.mean(dim=0)
        x_centered = train_x - mean
        cov = (x_centered.T @ x_centered) / (train_x.shape[0] - 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)

        self.register_buffer("mean", mean)
        self.register_buffer("eigvals", eigvals)
        self.register_buffer("eigvecs", eigvecs)
        self.eta = eta
        self.eps = eps

    def forward(self, x):
        x_centered = x - self.mean
        x_spec = x_centered @ self.eigvecs
        # Soft-PCA gate: s = eigvals / (eigvals + eta)
        s = self.eigvals / (self.eigvals + self.eta)
        x_white_spec = x_spec * s / torch.sqrt(self.eigvals + self.eps)
        x_out = x_white_spec @ self.eigvecs.T
        return x_out

class SpectralGatedWhitening(nn.Module):
    """Learnable gate: s = MLP(log(lambda))"""
    def __init__(self, train_x, hidden_dim=16, eps=1e-6):
        super().__init__()
        mean = train_x.mean(dim=0)
        x_centered = train_x - mean
        cov = (x_centered.T @ x_centered) / (train_x.shape[0] - 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)

        self.register_buffer("mean", mean)
        self.register_buffer("eigvals", eigvals)
        self.register_buffer("eigvecs", eigvecs)
        self.eps = eps

        # Gate MLP: log(lambda) -> weight in [0, 1]
        self.gate = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        # Initialize last layer to give high logits so it starts near ZCA (s=1)
        nn.init.constant_(self.gate[2].bias, 2.0)

    def forward(self, x):
        x_centered = x - self.mean
        x_spec = x_centered @ self.eigvecs

        # Compute gate weights from eigenvalues
        log_eigvals = torch.log(self.eigvals + 1e-8).unsqueeze(1)
        s = self.gate(log_eigvals).squeeze(1)

        # Apply gated whitening
        x_white_spec = x_spec * s / torch.sqrt(self.eigvals + self.eps)

        # Transform back to original space
        x_out = x_white_spec @ self.eigvecs.T
        return x_out

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, preprocessing=None):
        super().__init__()
        self.preprocessing = preprocessing
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if self.preprocessing is not None:
            x = self.preprocessing(x)
        return self.net(x)
