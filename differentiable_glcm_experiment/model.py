import torch
import torch.nn as nn
import torch.nn.functional as F

class DGLCMLayer(nn.Module):
    def __init__(self, num_bins=8, offsets=[1, 2, 3, 5], sigma=0.1):
        super(DGLCMLayer, self).__init__()
        self.num_bins = num_bins
        self.offsets = offsets
        # Learnable bin centers, initialized linearly between 0 and 1
        self.centers = nn.Parameter(torch.linspace(0, 1, num_bins))
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(sigma))))

        # Precompute grids for Haralick features
        i = torch.arange(self.num_bins, dtype=torch.float32)
        j = torch.arange(self.num_bins, dtype=torch.float32)
        I, J = torch.meshgrid(i, j, indexing='ij')
        self.register_buffer('I', I)
        self.register_buffer('J', J)
        self.register_buffer('i_indices', i)

    def forward(self, x):
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.shape

        # Normalize x to [0, 1] per sample
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        x = (x - x_min) / (x_max - x_min + 1e-8)

        # Soft binning: (batch_size, seq_len, num_bins)
        # (batch_size, seq_len, 1) - (num_bins)
        diff = x.unsqueeze(-1) - self.centers
        sigma = torch.exp(self.log_sigma)
        weights = torch.exp(-0.5 * (diff / sigma)**2)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        features = []
        for d in self.offsets:
            if d >= seq_len:
                continue

            # P_ij = sum_t weights[t, i] * weights[t+d, j]
            # (batch_size, seq_len - d, num_bins)
            w1 = weights[:, :-d, :]
            w2 = weights[:, d:, :]

            # GLCM: (batch_size, num_bins, num_bins)
            P = torch.bmm(w1.transpose(1, 2), w2)
            P = P / (P.sum(dim=(1, 2), keepdim=True) + 1e-8)

            # Haralick features
            # Contrast: sum (i-j)^2 * P_ij
            contrast = (P * (self.I - self.J)**2).sum(dim=(1, 2))

            # Energy (ASM): sum P_ij^2
            energy = (P**2).sum(dim=(1, 2))

            # Homogeneity: sum P_ij / (1 + (i-j)^2)
            homogeneity = (P / (1 + (self.I - self.J)**2)).sum(dim=(1, 2))

            # Entropy: -sum P_ij log(P_ij + eps)
            entropy = -(P * torch.log(P + 1e-8)).sum(dim=(1, 2))

            # Correlation
            # mu_i = sum_i i * sum_j P_ij
            # mu_j = sum_j j * sum_i P_ij
            pi = P.sum(dim=2) # (batch_size, num_bins)
            pj = P.sum(dim=1) # (batch_size, num_bins)

            mu_i = (self.i_indices * pi).sum(dim=1, keepdim=True)
            mu_j = (self.i_indices * pj).sum(dim=1, keepdim=True)

            var_i = (pi * (self.i_indices - mu_i)**2).sum(dim=1, keepdim=True)
            var_j = (pj * (self.i_indices - mu_j)**2).sum(dim=1, keepdim=True)

            # Correlation: sum (i - mu_i)(j - mu_j) P_ij / (sigma_i * sigma_j)
            corr = (P * (self.I - mu_i.unsqueeze(-1)) * (self.J - mu_j.unsqueeze(-1))).sum(dim=(1, 2)) / (torch.sqrt(var_i.squeeze(-1) * var_j.squeeze(-1)) + 1e-8)

            features.extend([contrast, energy, homogeneity, entropy, corr])

        # Stack features: (batch_size, num_offsets * 5)
        return torch.stack(features, dim=1)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super(BaselineMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class GLCMMLP(nn.Module):
    def __init__(self, input_dim=40, num_bins=8, offsets=[1, 2, 3, 5], hidden_dim=256, output_dim=10):
        super(GLCMMLP, self).__init__()
        self.glcm = DGLCMLayer(num_bins=num_bins, offsets=offsets)
        glcm_out_dim = len(offsets) * 5
        self.net = nn.Sequential(
            nn.Linear(glcm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        glcm_feats = self.glcm(x)
        return self.net(glcm_feats)

class GLCMAugmentedMLP(nn.Module):
    def __init__(self, input_dim=40, num_bins=8, offsets=[1, 2, 3, 5], hidden_dim=256, output_dim=10):
        super(GLCMAugmentedMLP, self).__init__()
        self.glcm = DGLCMLayer(num_bins=num_bins, offsets=offsets)
        glcm_out_dim = len(offsets) * 5
        self.net = nn.Sequential(
            nn.Linear(input_dim + glcm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        glcm_feats = self.glcm(x)
        combined = torch.cat([x, glcm_feats], dim=1)
        return self.net(combined)
