import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=40, hidden_size=256, num_classes=10):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.layers(x)
