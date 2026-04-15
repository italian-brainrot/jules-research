import torch
import torch.nn as nn
import torch.nn.functional as F

class DRQALayer(nn.Module):
    """
    Differentiable Recurrence Quantification Analysis (DRQA) layer.
    Computes soft RQA features from 1D signals.
    """
    def __init__(self, eps=0.1, gamma=10.0):
        super().__init__()
        self.log_eps = nn.Parameter(torch.log(torch.tensor(eps)))
        self.log_gamma = nn.Parameter(torch.log(torch.tensor(gamma)))

    def forward(self, x):
        # x shape: (batch, seq_len)
        batch_size, seq_len = x.shape
        x_unsqueezed = x.unsqueeze(-1) # (batch, seq_len, 1)

        # Compute pairwise distances
        # dists[b, i, j] = |x[b, i] - x[b, j]|
        dists = torch.abs(x_unsqueezed - x_unsqueezed.transpose(1, 2))

        eps = torch.exp(self.log_eps)
        gamma = torch.exp(self.log_gamma)

        # Soft recurrence matrix
        # R_ij = sigmoid(gamma * (eps - dist_ij))
        R = torch.sigmoid(gamma * (eps - dists))

        # 1. Recurrence Rate (RR)
        # Average of R excluding diagonal (though for RR diagonal is often included)
        rr = R.mean(dim=(1, 2))

        # 2. Determinism (DET)
        # Measures proportion of recurrence points that form diagonal lines
        # We can approximate this by looking at R[i, j] * R[i+1, j+1]
        if seq_len > 1:
            diag_sim = R[:, :-1, :-1] * R[:, 1:, 1:]
            det = diag_sim.mean(dim=(1, 2)) / (rr + 1e-6)
        else:
            det = torch.zeros_like(rr)

        # 3. Laminarity (LAM)
        # Measures proportion of recurrence points that form vertical lines
        if seq_len > 1:
            vert_sim = R[:, :-1, :] * R[:, 1:, :]
            lam = vert_sim.mean(dim=(1, 2)) / (rr + 1e-6)
        else:
            lam = torch.zeros_like(rr)

        # Stack features
        features = torch.stack([rr, det, lam], dim=1)
        return features

class RQAAugmentedMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super().__init__()
        self.drqa = DRQALayer()
        # DRQA returns 3 features
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x is (batch, 40)
        rqa_features = self.drqa(x)
        combined = torch.cat([x, rqa_features], dim=1)
        return self.mlp(combined)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super().__init__()
        # Adjust hidden_dim slightly to match parameters if needed,
        # but BaselineMLP is already quite similar.
        # RQAAugmentedMLP has (43*hidden_dim + hidden_dim*hidden_dim + hidden_dim*10) + biases + 2 params
        # BaselineMLP has (40*hidden_dim + hidden_dim*hidden_dim + hidden_dim*10) + biases
        # Difference is roughly 3 * hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    rqa_mlp = RQAAugmentedMLP()
    base_mlp = BaselineMLP()
    print(f"RQAAugmentedMLP parameters: {count_parameters(rqa_mlp)}")
    print(f"BaselineMLP parameters: {count_parameters(base_mlp)}")
