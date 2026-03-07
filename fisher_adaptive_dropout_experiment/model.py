import torch
import torch.nn as nn
import torch.nn.functional as F

class FisherDropout(nn.Module):
    """
    Fisher-Adaptive Dropout layer.
    Adjusts per-neuron dropout probability based on Fisher Information.
    """
    def __init__(self, num_features, p_base=0.5, gamma=0.0, alpha=0.9):
        super().__init__()
        self.num_features = num_features
        self.p_base = p_base
        self.gamma = gamma
        self.alpha = alpha  # Momentum for running Fisher Information

        # Register running Fisher Information estimate
        self.register_buffer('fisher_info', torch.ones(num_features))

    def forward(self, x):
        if not self.training or self.p_base == 0:
            return x

        # Update running Fisher estimate during the forward pass using squared activations
        # as a proxy for 'activity' or 'importance'.
        with torch.no_grad():
            batch_fisher = (x**2).mean(dim=0)
            self.fisher_info.mul_(self.alpha).add_(batch_fisher, alpha=1 - self.alpha)

            # Compute dropout probabilities
            # Normalize fisher info to have mean 1
            norm_fisher = self.fisher_info / (self.fisher_info.mean() + 1e-8)

            if self.gamma == 0:
                p = torch.full_like(norm_fisher, self.p_base)
            else:
                # p_j = p_base * (norm_fisher_j)^gamma
                p = self.p_base * torch.pow(norm_fisher, self.gamma)
                p = torch.clamp(p, 0.0, 0.95) # Avoid dropping everything

        # Apply dropout
        mask = (torch.rand_like(x) > p).float()
        # Rescale for expectation preservation: x / (1-p)
        return x * mask / (1 - p + 1e-8)

class AdaptiveMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10, p_base=0.5, gamma=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.drop1 = FisherDropout(hidden_dim, p_base=p_base, gamma=gamma)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = FisherDropout(hidden_dim, p_base=p_base, gamma=gamma)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x
