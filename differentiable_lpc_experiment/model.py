import torch
import torch.nn as nn
from burg_method import LPCLayer

class LPCClassifier(nn.Module):
    def __init__(self, input_dim, order, hidden_dim, num_classes, method='burg'):
        super().__init__()
        self.lpc = LPCLayer(order, method=method)
        self.fc = nn.Sequential(
            nn.Linear(order, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: (batch, input_dim)
        lpc_coeffs = self.lpc(x) # (batch, order)
        return self.fc(lpc_coeffs)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.fc(x)
