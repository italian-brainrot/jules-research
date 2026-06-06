import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_features=False):
        feat = self.features(x)
        logits = self.classifier(feat)
        if return_features:
            return logits, feat
        return logits

def get_model(input_dim=40, hidden_dim=256, output_dim=10):
    return MLP(input_dim, hidden_dim, output_dim)
