import torch
import torch.nn as nn
import torch.nn.functional as F

class VolterraLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(in_features, out_features)

        # Number of second-order terms: n(n+1)/2
        self.num_quad = in_features * (in_features + 1) // 2
        self.quad_weight = nn.Parameter(torch.Tensor(out_features, self.num_quad))

        # Indices for the upper triangular part of the outer product
        # Using register_buffer to keep it on the same device and saved in state_dict
        indices = torch.triu_indices(in_features, in_features)
        self.register_buffer('triu_indices_0', indices[0])
        self.register_buffer('triu_indices_1', indices[1])

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        # Use a smaller gain for quadratic terms to avoid initial instability
        nn.init.xavier_uniform_(self.quad_weight, gain=0.1)

    def forward(self, x):
        # x: (B, n)
        # Linear part
        out = self.linear(x) # (B, m)

        # Quadratic part: sum_{i <= k} K_{jik} x_i x_k
        # Efficiently compute x_i * x_k for all i, k
        # outer: (B, n, n)
        outer = torch.bmm(x.unsqueeze(2), x.unsqueeze(1))

        # Extract unique quadratic terms (upper triangular part of the outer product)
        # quad_terms: (B, num_quad)
        quad_terms = outer[:, self.triu_indices_0, self.triu_indices_1]

        # Apply quadratic weights
        quad_out = F.linear(quad_terms, self.quad_weight) # (B, m)

        return out + quad_out

class VolterraMLP(nn.Module):
    def __init__(self, in_dim=40, hidden_dim=40, out_dim=10, n_layers=2):
        super().__init__()
        layers = []
        curr_dim = in_dim
        for _ in range(n_layers):
            layers.append(VolterraLayer(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class BaselineMLP(nn.Module):
    def __init__(self, in_dim=40, hidden_dim=256, out_dim=10, n_layers=2):
        super().__init__()
        layers = []
        curr_dim = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    v_mlp = VolterraMLP(hidden_dim=40)
    b_mlp = BaselineMLP(hidden_dim=256)
    print(f"VolterraMLP parameters: {count_parameters(v_mlp)}")
    print(f"BaselineMLP parameters: {count_parameters(b_mlp)}")
