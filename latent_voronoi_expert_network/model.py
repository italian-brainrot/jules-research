import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LVEN(nn.Module):
    """Latent Voronoi Expert Network"""
    def __init__(self, input_dim, output_dim, latent_dim=32, num_experts=64, encoder_hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, latent_dim)
        )
        self.prototypes = nn.Parameter(torch.randn(num_experts, latent_dim) * 0.1)
        self.experts_weight = nn.Parameter(torch.empty(num_experts, input_dim, output_dim))
        self.experts_bias = nn.Parameter(torch.empty(num_experts, output_dim))
        self.log_temp = nn.Parameter(torch.zeros(1))

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming-like initialization for experts
        # We treat each expert as a linear layer (input_dim -> output_dim)
        stdv = 1. / math.sqrt(self.experts_weight.size(1))
        self.experts_weight.data.uniform_(-stdv, stdv)
        self.experts_bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: (B, I)
        latent = self.encoder(x) # (B, L)

        # Voronoi gating in latent space
        # dists_sq: (B, K)
        # Using expanded form for better performance: |a-b|^2 = |a|^2 + |b|^2 - 2ab
        latent_sq = torch.sum(latent**2, dim=1, keepdim=True) # (B, 1)
        proto_sq = torch.sum(self.prototypes**2, dim=1, keepdim=True).t() # (1, K)
        dists_sq = latent_sq + proto_sq - 2 * torch.matmul(latent, self.prototypes.t())

        # dists_sq might have small negative values due to precision
        dists_sq = F.relu(dists_sq)

        gating = F.softmax(-dists_sq * torch.exp(-self.log_temp), dim=1) # (B, K)

        # Expert outputs: (B, K, O)
        # x: (B, I), weights: (K, I, O)
        expert_outputs = torch.einsum('bi,kio->bko', x, self.experts_weight) + self.experts_bias # (B, K, O)

        # Final output as weighted sum: (B, O)
        out = torch.einsum('bk,bko->bo', gating, expert_outputs)
        return out

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=170):
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    lven = LVEN(40, 10)
    mlp = BaselineMLP(40, 10)
    print(f"LVEN parameters: {count_parameters(lven)}")
    print(f"BaselineMLP parameters: {count_parameters(mlp)}")
