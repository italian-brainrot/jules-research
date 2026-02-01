import torch
import torch.nn as nn
import torch.nn.functional as F

class ORelu(nn.Module):
    def __init__(self, num_parameters=1):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((num_parameters,), 0.01))
        self.beta = nn.Parameter(torch.full((num_parameters,), 1.0))

    def forward(self, x):
        if self.alpha.shape[0] == 1:
            return F.relu(x) + self.alpha * torch.sin(self.beta * x)
        else:
            # Assume x has shape (batch, channels) or similar
            # If x is (batch, channels, length), we need to reshape parameters
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

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_type='relu'):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            if activation_type == 'relu':
                layers.append(nn.ReLU())
            elif activation_type == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.01))
            elif activation_type == 'elu':
                layers.append(nn.ELU())
            elif activation_type == 'orelu':
                layers.append(ORelu(h_dim))
            elif activation_type == 'snake':
                layers.append(Snake(h_dim))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
