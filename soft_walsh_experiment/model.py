import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftWalshLayer(nn.Module):
    def __init__(self, in_features, num_terms, init_scale=0.05):
        super().__init__()
        # w is in [-1, 1]. Initialized small so terms are near 1.
        self.w = nn.Parameter(torch.randn(num_terms, in_features) * init_scale)

    def forward(self, x):
        # x: (B, in_features) in [-1, 1]
        # w: (M, in_features)
        w = torch.tanh(self.w)

        # terms: (M, B, in_features)
        # T_mk = 1 - |w_mk| + w_mk * x_k
        # broadcasting w: (M, 1, in_features), x: (1, B, in_features)
        terms = 1.0 - torch.abs(w).unsqueeze(1) + w.unsqueeze(1) * x.unsqueeze(0)

        # product over in_features: (M, B)
        # Using torch.prod. 40 elements.
        y = torch.prod(terms, dim=2)
        return y.T # (B, M)

class SoftWalshNetwork(nn.Module):
    def __init__(self, in_features, num_terms, out_features, deep=False, init_scale=0.05):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.swl = SoftWalshLayer(in_features, num_terms, init_scale=init_scale)
        self.deep = deep
        if deep:
            self.swl2 = SoftWalshLayer(num_terms, num_terms, init_scale=init_scale)
            self.linear = nn.Linear(num_terms, out_features)
        else:
            self.linear = nn.Linear(num_terms, out_features)

    def forward(self, x):
        x = self.norm(x)
        x = torch.tanh(x)
        h = self.swl(x)
        if self.deep:
            h = torch.tanh(h)
            h = self.swl2(h)
        return self.linear(h)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, num_layers=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features, hidden_size))
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
