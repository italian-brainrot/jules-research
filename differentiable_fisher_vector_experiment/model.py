import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FisherVectorLayer(nn.Module):
    def __init__(self, num_clusters, patch_size, stride):
        super().__init__()
        self.num_clusters = num_clusters
        self.patch_size = patch_size
        self.stride = stride
        self.dim = patch_size

        # GMM parameters
        self.logits = nn.Parameter(torch.zeros(num_clusters)) # weights = softmax(logits)
        self.means = nn.Parameter(torch.randn(num_clusters, self.dim) * 0.1)
        self.log_vars = nn.Parameter(torch.zeros(num_clusters, self.dim)) # variances = exp(log_vars)

    def forward(self, x):
        # x: (Batch, 1, Length)
        # Extract patches
        # patches: (Batch, dim, T)
        patches = x.unfold(2, self.patch_size, self.stride).transpose(1, 2).reshape(x.size(0), -1, self.patch_size).transpose(1, 2)
        batch_size, dim, T = patches.shape

        # Reshape patches to (Batch, T, dim)
        patches = patches.transpose(1, 2) # (Batch, T, dim)

        # Compute GMM components
        # weights: (K,)
        weights = F.softmax(self.logits, dim=0)
        # vars: (K, dim)
        variances = torch.exp(self.log_vars) + 1e-6
        # precision: (K, dim)
        precisions = 1.0 / variances

        # log_p: (Batch, T, K)
        # log p_k(x_t) = -0.5 * [dim * log(2pi) + sum(log(sigma_k^2)) + sum((x_t - mu_k)^2 / sigma_k^2)]
        diff = patches.unsqueeze(2) - self.means.unsqueeze(0).unsqueeze(0) # (Batch, T, K, dim)
        exponent = torch.sum(diff**2 * precisions.unsqueeze(0).unsqueeze(0), dim=-1) # (Batch, T, K)
        log_det = torch.sum(self.log_vars, dim=-1) # (K,)

        log_p = -0.5 * (self.dim * math.log(2 * math.pi) + log_det.unsqueeze(0).unsqueeze(0) + exponent)

        # Gamma (responsibilities): (Batch, T, K)
        log_weighted_p = log_p + torch.log(weights + 1e-6).unsqueeze(0).unsqueeze(0)
        gamma = F.softmax(log_weighted_p, dim=-1)

        # Fisher Vector components
        # u_k: (Batch, K, dim)
        # u_k = 1/(T*sqrt(w_k)) * sum_t gamma_t(k) * (x_t - mu_k) / sigma_k
        sqrt_w = torch.sqrt(weights + 1e-6).view(1, 1, self.num_clusters, 1)
        sigma = torch.sqrt(variances).view(1, 1, self.num_clusters, self.dim)

        u = (gamma.unsqueeze(-1) * (diff / sigma)) # (Batch, T, K, dim)
        u = u.sum(dim=1) / (T * sqrt_w.squeeze(1)) # (Batch, K, dim)

        # v_k: (Batch, K, dim)
        # v_k = 1/(T*sqrt(2*w_k)) * sum_t gamma_t(k) * [(x_t - mu_k)^2 / sigma_k^2 - 1]
        v = (gamma.unsqueeze(-1) * (diff**2 / variances.unsqueeze(0).unsqueeze(0) - 1.0))
        v = v.sum(dim=1) / (T * math.sqrt(2) * sqrt_w.squeeze(1)) # (Batch, K, dim)

        # Concatenate u and v
        fv = torch.cat([u.reshape(batch_size, -1), v.reshape(batch_size, -1)], dim=1) # (Batch, 2 * K * dim)

        # Power normalization
        fv = torch.sign(fv) * torch.sqrt(torch.abs(fv) + 1e-6)

        # L2 normalization
        fv = F.normalize(fv, p=2, dim=1)

        return fv

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
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

class DFVNet(nn.Module):
    def __init__(self, input_dim, num_clusters, patch_size, stride, hidden_dim, output_dim):
        super().__init__()
        self.dfv = FisherVectorLayer(num_clusters, patch_size, stride)
        fv_dim = 2 * num_clusters * patch_size
        self.mlp = nn.Sequential(
            nn.Linear(fv_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        fv = self.dfv(x)
        return self.mlp(fv)

class DFVAugmentedMLP(nn.Module):
    def __init__(self, input_dim, num_clusters, patch_size, stride, hidden_dim, output_dim):
        super().__init__()
        self.dfv = FisherVectorLayer(num_clusters, patch_size, stride)
        fv_dim = 2 * num_clusters * patch_size
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + fv_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        raw_x = x
        if x.dim() == 2:
            x = x.unsqueeze(1)
        fv = self.dfv(x)
        combined = torch.cat([raw_x, fv], dim=1)
        return self.mlp(combined)
