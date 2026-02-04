import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ORelu(nn.Module):
    def __init__(self, num_parameters=1):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((num_parameters,), 0.01))
        self.beta = nn.Parameter(torch.full((num_parameters,), 1.0))

    def forward(self, x):
        if self.alpha.shape[0] == 1:
            return F.relu(x) + self.alpha * torch.sin(self.beta * x)
        else:
            if x.dim() == 2:
                return F.relu(x) + self.alpha * torch.sin(self.beta * x)
            elif x.dim() == 3:
                return F.relu(x) + self.alpha.view(1, -1, 1) * torch.sin(self.beta.view(1, -1, 1) * x)
            else:
                return F.relu(x) + self.alpha * torch.sin(self.beta * x)

class Snake(nn.Module):
    def __init__(self, num_parameters=1):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((num_parameters,), 1.0))

    def forward(self, x):
        if self.alpha.shape[0] == 1:
            return x + (1.0 / (self.alpha + 1e-9)) * (torch.sin(self.alpha * x) ** 2)
        else:
            if x.dim() == 2:
                return x + (1.0 / (self.alpha + 1e-9)) * (torch.sin(self.alpha * x) ** 2)
            elif x.dim() == 3:
                return x + (1.0 / (self.alpha.view(1, -1, 1) + 1e-9)) * (torch.sin(self.alpha.view(1, -1, 1) * x) ** 2)
            else:
                return x + (1.0 / (self.alpha + 1e-9)) * (torch.sin(self.alpha * x) ** 2)

class LFSA(nn.Module):
    """Learnable Fourier Series Activation"""
    def __init__(self, num_parameters=1, K=4):
        super().__init__()
        self.num_parameters = num_parameters
        self.K = K
        # Linear trend: f(x) = w*x + b + sum(a_k * sin(omega_k * x + phi_k))
        self.w = nn.Parameter(torch.full((num_parameters,), 1.0))
        self.b = nn.Parameter(torch.full((num_parameters,), 0.0))

        # Fourier components
        self.a = nn.Parameter(torch.randn(num_parameters, K) * 0.01)
        self.omega = nn.Parameter(torch.randn(num_parameters, K) * 1.0)
        self.phi = nn.Parameter(torch.randn(num_parameters, K) * math.pi)

    def forward(self, x):
        # x shape: (batch, hidden)
        if self.num_parameters == 1:
            res = self.w * x + self.b
            for k in range(self.K):
                res = res + self.a[0, k] * torch.sin(self.omega[0, k] * x + self.phi[0, k])
            return res
        else:
            # Broadcast parameters
            if x.dim() == 2:
                # x: (B, H), w: (H,), a: (H, K)
                res = self.w * x + self.b
                # We can optimize this with einsum or just a loop if K is small
                for k in range(self.K):
                    res = res + self.a[:, k] * torch.sin(self.omega[:, k] * x + self.phi[:, k])
                return res
            elif x.dim() == 3:
                # x: (B, H, L)
                res = self.w.view(1, -1, 1) * x + self.b.view(1, -1, 1)
                for k in range(self.K):
                    res = res + self.a[:, k].view(1, -1, 1) * torch.sin(self.omega[:, k].view(1, -1, 1) * x + self.phi[:, k].view(1, -1, 1))
                return res
            else:
                return self.w * x + self.b # Fallback

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_type='relu', num_params=None):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            act_params = h_dim if num_params == 'per_neuron' else 1

            if activation_type == 'relu':
                layers.append(nn.ReLU())
            elif activation_type == 'gelu':
                layers.append(nn.GELU())
            elif activation_type == 'orelu':
                layers.append(ORelu(act_params))
            elif activation_type == 'snake':
                layers.append(Snake(act_params))
            elif activation_type == 'lfsa':
                layers.append(LFSA(act_params, K=4))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
