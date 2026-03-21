import torch
import torch.nn as nn

class QuadraticLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.u = nn.Parameter(torch.Tensor(out_features, in_features))
        self.v = nn.Parameter(torch.Tensor(out_features, in_features))

        # Initialize u and v with smaller weights to prevent the quadratic term
        # from dominating the initial gradients
        nn.init.xavier_uniform_(self.u, gain=0.1)
        nn.init.xavier_uniform_(self.v, gain=0.1)

    def forward(self, x):
        # x shape: (batch, in_features)
        # u, v shape: (out_features, in_features)
        lin = self.linear(x)
        # We want to compute (Ux) * (Vx) element-wise
        # torch.functional.linear(x, self.u) has shape (batch, out_features)
        quad = torch.nn.functional.linear(x, self.u) * torch.nn.functional.linear(x, self.v)
        return lin + quad

class LRQIN(nn.Module):
    """Low-Rank Quadratic Interaction Network"""
    def __init__(self, in_dim=40, hidden_dim=68, out_dim=10):
        super().__init__()
        self.layers = nn.Sequential(
            QuadraticLayer(in_dim, hidden_dim),
            nn.ReLU(),
            QuadraticLayer(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)

class BaselineMLP(nn.Module):
    """Standard Multi-Layer Perceptron"""
    def __init__(self, in_dim=40, hidden_dim=128, out_dim=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    lrqin = LRQIN(hidden_dim=68)
    mlp = BaselineMLP(hidden_dim=128)
    print(f"LRQIN parameters: {count_parameters(lrqin)}")
    print(f"MLP parameters: {count_parameters(mlp)}")
