import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, n_prototypes_per_class=10):
        super().__init__()
        self.n_classes = output_dim
        self.n_prototypes_per_class = n_prototypes_per_class
        # (n_classes, n_prototypes_per_class, input_dim)
        self.prototypes = nn.Parameter(torch.randn(output_dim, n_prototypes_per_class, input_dim))
        # Distance metric: each prototype has its own diagonal distance metric
        self.log_dist_scales = nn.Parameter(torch.zeros(output_dim, n_prototypes_per_class, input_dim))

    def forward(self, x):
        # x: (B, I)
        # prototypes: (C, K, I)
        # diff: (B, C, K, I)
        diff = x.unsqueeze(1).unsqueeze(2) - self.prototypes.unsqueeze(0)
        # scaled_dist: (B, C, K)
        dist_sq = torch.sum((diff**2) * torch.exp(self.log_dist_scales), dim=3)
        # For each class, use a softmin over its prototypes to get class-level distance
        # softmin(d) = -log(sum(exp(-d)))
        # Here we just use min for simplicity, or negative logsumexp for smoothness
        class_dist = -torch.logsumexp(-dist_sq, dim=2) # (B, C)

        # Logits are negative distances
        return -class_dist

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, n_layers=2):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
