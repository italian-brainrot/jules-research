import torch
import torch.nn as nn
from torch.func import functional_call, vmap, jacrev

class Encoder(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, latent_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=2, hidden_dim=128, output_dim=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class Autoencoder(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, latent_dim=2):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

def get_jacobian(model, x):
    # model: the encoder
    # x: [B, D]
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def model_func(p, b, x_single):
        # x_single is [D]
        return functional_call(model, (p, b), x_single)

    # jacrev(model_func, argnums=2) returns [latent_dim, input_dim]
    return vmap(lambda x_val: jacrev(model_func, argnums=2)(params, buffers, x_val))(x)

def cae_penalty(model, x):
    J = get_jacobian(model, x) # [B, d, D]
    return (J**2).sum(dim=(1, 2)).mean()

def lip_penalty(model, x):
    J = get_jacobian(model, x) # [B, d, D]
    # J J^T
    JJT = torch.bmm(J, J.transpose(1, 2)) # [B, d, d]
    I = torch.eye(JJT.size(1), device=JJT.device).unsqueeze(0) # [1, d, d]
    diff = JJT - I
    return (diff**2).sum(dim=(1, 2)).mean()

class LatentClassifier(nn.Module):
    def __init__(self, latent_dim, hidden_dim=64, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)
