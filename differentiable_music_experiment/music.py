import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DifferentiableMUSIC(nn.Module):
    def __init__(self, window_size, num_freqs=64, eps=1e-6):
        super().__init__()
        self.window_size = window_size
        self.num_freqs = num_freqs
        self.eps = eps

        # Learnable parameters for soft subspace selection
        # log_threshold: where we transition from signal to noise
        self.log_threshold = nn.Parameter(torch.tensor(0.0))
        # log_beta: steepness of the transition
        self.log_beta = nn.Parameter(torch.tensor(2.0))

        # Precompute steering vectors (Vandermonde matrix)
        # frequencies from 0 to pi (since signal is real)
        freqs = torch.linspace(0, np.pi, num_freqs)
        # a(w) = [1, e^jw, e^j2w, ..., e^j(M-1)w]^T
        # Shape: (window_size, num_freqs)
        indices = torch.arange(window_size).unsqueeze(1).float()
        angles = indices * freqs.unsqueeze(0)

        self.register_buffer("steering_vectors_real", torch.cos(angles))
        self.register_buffer("steering_vectors_imag", torch.sin(angles))

    def forward(self, x):
        """
        x: (batch_size, seq_len)
        """
        B, L = x.shape
        K = self.window_size
        N = L - K + 1

        # 1. Form Hankel matrix: (B, K, N)
        # Each column is a segment of length K
        H = x.unfold(1, K, 1).transpose(1, 2)

        # 2. Correlation matrix R = H @ H^T / N: (B, K, K)
        # Using SVD on H is often more stable: H = U S V^T
        # R = U S^2 U^T / N. The eigenvectors of R are U.
        # Eigenvalues of R are S^2 / N.

        U, S, Vh = torch.linalg.svd(H, full_matrices=False)
        lambdas = (S**2) / N # (B, K)

        # 3. Soft noise subspace selection
        # We want weights to be 1 for small eigenvalues (noise) and 0 for large ones (signal)
        threshold = torch.exp(self.log_threshold)
        beta = torch.exp(self.log_beta)

        # weights = sigmoid(beta * (threshold - lambdas))
        # If lambdas < threshold, weights -> 1
        # If lambdas > threshold, weights -> 0
        weights = torch.sigmoid(beta * (threshold - lambdas)) # (B, K)

        # 4. Compute Pseudospectrum
        # P(w) = 1 / sum_i (w_i * |a(w)^H u_i|^2)
        # a(w)^H u_i is complex.
        # u_i are real since H is real.
        # (Actually U from SVD of real H is real)

        # steering_vectors: (K, F)
        # U: (B, K, K)
        # U^T @ steering_vectors: (B, K, F)

        # Real part of a(w)^H u_i: real(a)^T u_i
        # Imaginary part of a(w)^H u_i: -imag(a)^T u_i

        proj_real = torch.matmul(U.transpose(1, 2), self.steering_vectors_real) # (B, K, F)
        proj_imag = torch.matmul(U.transpose(1, 2), self.steering_vectors_imag) # (B, K, F)

        squared_dist = proj_real**2 + proj_imag**2 # (B, K, F)

        # Weighted sum over eigenvectors (K dimension)
        denominator = torch.sum(weights.unsqueeze(-1) * squared_dist, dim=1) # (B, F)

        pseudospectrum = 1.0 / (denominator + self.eps)

        # Optional: return log pseudospectrum for better scaling
        return torch.log10(pseudospectrum)

class MUSICMLP(nn.Module):
    def __init__(self, input_size, window_size, hidden_dim, num_classes, num_freqs=64):
        super().__init__()
        self.music = DifferentiableMUSIC(window_size, num_freqs=num_freqs)
        self.fc = nn.Sequential(
            nn.Linear(num_freqs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        features = self.music(x)
        return self.fc(features)

class MUSICAugmentedMLP(nn.Module):
    def __init__(self, input_size, window_size, hidden_dim, num_classes, num_freqs=64):
        super().__init__()
        self.music = DifferentiableMUSIC(window_size, num_freqs=num_freqs)
        self.fc = nn.Sequential(
            nn.Linear(input_size + num_freqs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        music_features = self.music(x)
        combined = torch.cat([x, music_features], dim=1)
        return self.fc(combined)

class BaselineMLP(nn.Module):
    def __init__(self, input_size, hidden_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.fc(x)
