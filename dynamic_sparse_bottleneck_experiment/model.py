import torch
import torch.nn as nn
import torch.nn.functional as F

class DSBLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.tau_predictor = nn.Linear(dim, dim)
        # Initialize to small values so tau is initially small
        nn.init.zeros_(self.tau_predictor.weight)
        nn.init.constant_(self.tau_predictor.bias, -2.0) # softplus(-2) approx 0.127

    def forward(self, x):
        tau = F.softplus(self.tau_predictor(x))
        # Soft thresholding: sign(x) * max(0, |x| - tau)
        return torch.sign(x) * torch.maximum(torch.zeros_like(x), torch.abs(x) - tau)

class FSBLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.tau = nn.Parameter(torch.full((dim,), -2.0)) # Store in log-space/pre-softplus

    def forward(self, x):
        tau = F.softplus(self.tau)
        return torch.sign(x) * torch.maximum(torch.zeros_like(x), torch.abs(x) - tau)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10, num_layers=3):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.GELU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DSBMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        curr_dim = input_dim
        for _ in range(num_layers):
            self.layers.append(nn.Linear(curr_dim, hidden_dim))
            self.layers.append(DSBLayer(hidden_dim))
            curr_dim = hidden_dim
        self.classifier = nn.Linear(curr_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x)

class FSBMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        curr_dim = input_dim
        for _ in range(num_layers):
            self.layers.append(nn.Linear(curr_dim, hidden_dim))
            self.layers.append(FSBLayer(hidden_dim))
            curr_dim = hidden_dim
        self.classifier = nn.Linear(curr_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x)
