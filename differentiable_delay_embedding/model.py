import torch
import torch.nn as nn
import torch.nn.functional as F

class DLDELayer(nn.Module):
    """
    Differentiable Learnable Delay Embedding Layer.
    Reconstructs the phase space of a 1D signal with a learnable delay tau.
    """
    def __init__(self, m=5, initial_tau=1.0, L=40):
        super().__init__()
        self.m = m
        self.tau = nn.Parameter(torch.tensor(initial_tau))
        self.L = L

    def forward(self, x):
        # x: (B, L)
        B, L = x.shape
        device = x.device

        # j is the time index, d is the embedding dimension index
        j = torch.arange(L, device=device, dtype=torch.float32) # (L,)
        d = torch.arange(self.m, device=device, dtype=torch.float32) # (m,)

        # indices[d, j] = j - d * tau
        # We use a broad casting to get (m, L) indices
        indices = j.unsqueeze(0) - d.unsqueeze(1) * self.tau # (m, L)

        # Flatten for interpolation
        flat_indices = indices.view(-1) # (m * L)

        # Linear interpolation
        low = flat_indices.floor().long()
        high = low + 1
        alpha = (flat_indices - low.float()).unsqueeze(0) # (1, m * L)

        # Clamp to [0, L-1] to handle boundaries
        low_clamped = torch.clamp(low, 0, L - 1)
        high_clamped = torch.clamp(high, 0, L - 1)

        # x is (B, L). We want to gather at (B, m*L)
        batch_low = low_clamped.unsqueeze(0).expand(B, -1)
        batch_high = high_clamped.unsqueeze(0).expand(B, -1)

        val_low = torch.gather(x, 1, batch_low)
        val_high = torch.gather(x, 1, batch_high)

        # Linear interpolation: val_low + alpha * (val_high - val_low)
        out = val_low + alpha * (val_high - val_low)
        out = out.view(B, self.m, L)
        return out

class DLDEModel(nn.Module):
    def __init__(self, input_dim=40, m=5, hidden_dim=256, output_dim=10):
        super().__init__()
        self.dlde = DLDELayer(m=m, L=input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(m * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (B, L)
        emb = self.dlde(x) # (B, m, L)
        emb_flat = emb.view(x.size(0), -1)
        return self.mlp(emb_flat)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=320, output_dim=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    dlde_model = DLDEModel()
    baseline_model = BaselineMLP()
    print(f"DLDEModel parameters: {count_parameters(dlde_model)}")
    print(f"BaselineMLP parameters: {count_parameters(baseline_model)}")
