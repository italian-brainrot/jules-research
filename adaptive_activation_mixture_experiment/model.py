import torch
import torch.nn as nn
import torch.nn.functional as F

class MixtureActivation(nn.Module):
    def __init__(self, init_weights=None, learn_omega=True):
        super().__init__()
        # Initializing weights so they are equal after softmax
        if init_weights is None:
            # ReLU, Tanh, Sin, Identity
            self.weights = nn.Parameter(torch.zeros(4))
        else:
            self.weights = nn.Parameter(torch.tensor(init_weights))

        self.omega = nn.Parameter(torch.tensor(1.0))
        self.omega.requires_grad = learn_omega

    def forward(self, x):
        w = torch.softmax(self.weights, dim=0)
        return w[0] * F.relu(x) + \
               w[1] * torch.tanh(x) + \
               w[2] * torch.sin(self.omega * x) + \
               w[3] * x

class AdaptiveMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, use_mixture=True):
        super().__init__()
        layers = []
        in_dim = input_size
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_mixture:
                layers.append(MixtureActivation())
            else:
                layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def get_mixture_weights(self):
        weights = []
        for module in self.modules():
            if isinstance(module, MixtureActivation):
                weights.append(torch.softmax(module.weights, dim=0).detach().cpu().numpy())
        return weights

    def get_omegas(self):
        omegas = []
        for module in self.modules():
            if isinstance(module, MixtureActivation):
                omegas.append(module.omega.item())
        return omegas
