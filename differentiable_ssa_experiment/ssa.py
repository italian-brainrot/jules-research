import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableSSA(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        # Learnable weights for each singular component, initialized to be near 1 (sigmoid(10) approx 1)
        self.weights = nn.Parameter(torch.ones(self.window_size) * 5.0)

    def hankelize(self, x):
        # x: (B, L)
        B, L = x.shape
        K = self.window_size
        N = L - K + 1

        # Use unfolding to create the Hankel matrix efficiently
        # returns (B, N, K) - wait, unfold gives (B, num_patches, patch_size)
        H = x.unfold(1, K, 1) # (B, N, K)
        # We want (B, K, N) to match standard SSA notation
        return H.transpose(1, 2)

    def diagonal_averaging(self, H):
        # H: (B, K, N)
        B, K, N = H.shape
        L = K + N - 1

        x_reconstructed = torch.zeros((B, L), device=H.device, dtype=H.dtype)
        counts = torch.zeros(L, device=H.device, dtype=H.dtype)

        for i in range(K):
            x_reconstructed[:, i:i+N] += H[:, i, :]
            counts[i:i+N] += 1

        return x_reconstructed / counts

    def forward(self, x):
        # x: (B, L)
        B, L = x.shape
        H = self.hankelize(x) # (B, K, N)

        # SVD: H = U S V^T
        # U: (B, K, r), S: (B, r), Vh: (B, r, N) where r = min(K, N)
        U, S, Vh = torch.linalg.svd(H, full_matrices=False)

        # Apply weights to singular values
        # Sigmoid ensures weights are in (0, 1)
        w = torch.sigmoid(self.weights[:S.shape[1]])
        S_weighted = S * w

        # Reconstruct H_weighted = U * diag(S_weighted) * Vh
        H_weighted = U @ (S_weighted.unsqueeze(-1) * Vh)

        # Diagonal averaging
        x_out = self.diagonal_averaging(H_weighted)

        return x_out

class SSANet(nn.Module):
    def __init__(self, input_size, window_size, hidden_size, num_classes):
        super().__init__()
        self.ssa = DifferentiableSSA(window_size)
        # Use same hidden size as baseline to keep capacity comparable
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, L)
        x_ssa = self.ssa(x)
        # Only use x_ssa to see if it provides better features than raw x
        out = self.fc1(x_ssa)
        out = self.relu(out)
        out = self.fc2(out)
        return out
