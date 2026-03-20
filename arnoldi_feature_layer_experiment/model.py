import torch
import torch.nn as nn

def arnoldi_process(A, x, k, eps=1e-8):
    """
    Performs k steps of the Arnoldi process for matrix A and starting vector x.
    Returns the upper Hessenberg matrix H.

    A: (d, d) matrix
    x: (B, d) starting vectors
    k: Number of Arnoldi steps
    """
    batch_size, d = x.shape
    # Normalize starting vector x
    q = x / (torch.norm(x, dim=1, keepdim=True) + eps)

    qs = [q]
    # H will be (B, k+1, k)
    H = torch.zeros(batch_size, k + 1, k, device=x.device, dtype=x.dtype)

    for j in range(k):
        # v = A * q_j (for each batch)
        v = torch.matmul(qs[j].unsqueeze(1), A).squeeze(1) # (B, d)

        for i in range(j + 1):
            # h_ij = q_i^* * v
            h_ij = torch.sum(qs[i] * v, dim=1) # (B,)
            H[:, i, j] = h_ij
            # v = v - h_ij * q_i
            v = v - h_ij.unsqueeze(1) * qs[i]

        # h_{j+1, j} = ||v||
        h_next_j = torch.norm(v, dim=1)
        H[:, j + 1, j] = h_next_j

        # q_{j+1} = v / h_{j+1, j}
        qs.append(v / (h_next_j.unsqueeze(1) + eps))

    return H

class ArnoldiFeatureLayer(nn.Module):
    """
    A layer that applies the Arnoldi process to extract structured features from input data.
    """
    def __init__(self, input_dim, k, num_heads=1):
        super().__init__()
        self.input_dim = input_dim
        self.k = k
        self.num_heads = num_heads
        # Learnable matrices A for each head
        self.A = nn.Parameter(torch.randn(num_heads, input_dim, input_dim) * 0.1)
        # Residual connection weight
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x shape: (B, input_dim)
        B, d = x.shape
        all_H = []
        for i in range(self.num_heads):
            # Extract features via Arnoldi process for each head
            H = arnoldi_process(self.A[i], x, self.k) # (B, k+1, k)
            all_H.append(H.reshape(B, -1))

        # Concatenate features from all heads
        out = torch.cat(all_H, dim=1)
        return out

class ArnoldiMLP(nn.Module):
    def __init__(self, input_dim, k, num_heads, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        self.arnoldi = ArnoldiFeatureLayer(input_dim, k, num_heads)
        arnoldi_out_dim = num_heads * (k + 1) * k

        # Initial projection to hidden_dim while keeping input
        self.proj = nn.Linear(input_dim + arnoldi_out_dim, hidden_dim)

        layers = []
        curr_dim = hidden_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        features = self.arnoldi(x)
        x = torch.cat([x, features], dim=1)
        x = torch.relu(self.proj(x))
        return self.net(x)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
