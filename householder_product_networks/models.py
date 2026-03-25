import torch
import torch.nn as nn
import torch.nn.functional as F

class HouseholderLinear(nn.Module):
    """
    Orthogonal linear layer using a product of Householder reflectors.
    A = (I - 2v_1 v_1^T) (I - 2v_2 v_2^T) ... (I - 2v_k v_k^T)
    x' = A x

    This layer only supports input_features == output_features.
    """
    def __init__(self, features, num_reflectors, bias=True):
        super().__init__()
        self.features = features
        self.num_reflectors = num_reflectors

        # v must be unit norm for Householder reflector H = I - 2vv^T
        # We store them as nn.Parameter and normalize in forward.
        self.v = nn.Parameter(torch.randn(num_reflectors, features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # x: (batch, features)
        # To make it efficient, we iterate over reflectors.
        # Hv = (I - 2vv^T)x = x - 2v(v^Tx)
        res = x
        for i in range(self.num_reflectors):
            v = self.v[i]
            v = v / (torch.norm(v) + 1e-8)
            # Efficiently compute x - 2v(v^Tx)
            # v^Tx is dot product of each x with v
            dot = torch.matmul(res, v) # (batch,)
            # Reshape dot to (batch, 1) to multiply with v (1, features)
            res = res - 2.0 * torch.outer(dot, v)

        if self.bias is not None:
            res = res + self.bias
        return res

class HouseholderMLP(nn.Module):
    """
    MLP that uses HouseholderLinear layers.

    Architecture:
    Input -> Linear(input, hidden) -> [Householder(hidden) -> ReLU] x num_layers -> Linear(hidden, output)
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, num_reflectors=None):
        super().__init__()
        if num_reflectors is None:
            num_reflectors = hidden_size

        self.input_proj = nn.Linear(input_size, hidden_size)

        self.householder_layers = nn.ModuleList([
            HouseholderLinear(hidden_size, num_reflectors)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.householder_layers:
            x = layer(x)
            x = F.relu(x)
        x = self.output_proj(x)
        return x

class BaselineMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def get_model(name, input_size, hidden_size, output_size, **kwargs):
    if name == "householder":
        return HouseholderMLP(input_size, hidden_size, output_size, **kwargs)
    elif name == "baseline":
        return BaselineMLP(input_size, hidden_size, output_size, **kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
