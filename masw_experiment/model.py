import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=40, hidden_sizes=[256, 256], num_classes=10):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
