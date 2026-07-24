import torch
import torch.nn as nn

class LDAClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_features=False):
        features = self.net(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits
