import torch
import torch.nn as nn
import torch.nn.functional as F

def gelu_derivative(x):
    return 0.5 * (1.0 + torch.erf(x / 1.41421356237)) + (x / 2.50662827463) * torch.exp(-0.5 * x**2)

class INGOMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, penalty_type='none'):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
        self.output = nn.Linear(hidden_dims[-1], output_dim)
        self.penalty_type = penalty_type

    def forward(self, x):
        h = x
        penalties = []
        for i, layer in enumerate(self.layers):
            z = layer(h)
            h = F.gelu(z)

            if self.penalty_type == 'weight_ortho':
                penalties.append(self.compute_weight_ortho(layer.weight))
            elif self.penalty_type == 'ingo':
                mask = gelu_derivative(z)
                penalties.append(self.compute_ingo(layer.weight, mask))

        logits = self.output(h)
        if self.penalty_type != 'none':
            return logits, sum(penalties)
        return logits

    def compute_weight_ortho(self, W):
        norm = W.norm(dim=1, keepdim=True) + 1e-8
        W_normalized = W / norm
        S = torch.mm(W_normalized, W_normalized.t())
        S2 = S * S
        penalty = (torch.sum(S2) - W.shape[0]) / (W.shape[0] * (W.shape[0] - 1) + 1e-8)
        return penalty

    def compute_ingo(self, W, mask):
        norm = W.norm(dim=1, keepdim=True) + 1e-8
        W_normalized = W / norm
        total_grad = torch.mm(mask, W_normalized)
        # Mean squared norm over batch, normalized by hidden dim to be comparable to weight_ortho
        penalty = torch.mean(torch.sum(total_grad**2, dim=1)) / W.shape[0]
        return penalty
