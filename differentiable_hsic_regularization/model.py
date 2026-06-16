import torch
import torch.nn as nn
from differentiable_hsic_regularization.hsic import hsic, hsic_normalized

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
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

class HSICRegularizedMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_hidden=False):
        h1 = self.act1(self.layer1(x))
        h2 = self.act2(self.layer2(h1))
        out = self.layer3(h2)
        if return_hidden:
            return out, h2
        return out

def get_hsic_loss(hidden, inputs, targets, hsic_type='standard', weight_in=0.1, weight_out=0.1):
    """
    Compute HSIC-based bottleneck loss.
    We want to minimize dependency between hidden and inputs (bottleneck)
    and maximize dependency between hidden and targets (sufficiency).
    """
    # Convert targets to one-hot or just use labels if they are categorical?
    # For HSIC, categorical labels can be used if kernel is appropriate.
    # Alternatively, one-hot encode them.

    # One-hot encode targets for HSIC calculation
    num_classes = targets.max().item() + 1
    y_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()

    if hsic_type == 'standard':
        h_in = hsic(hidden, inputs)
        h_out = hsic(hidden, y_onehot)
    else:
        h_in = hsic_normalized(hidden, inputs)
        h_out = hsic_normalized(hidden, y_onehot)

    # We want to minimize h_in and maximize h_out
    # Loss = weight_in * h_in - weight_out * h_out
    return weight_in * h_in - weight_out * h_out
