import torch
import torch.nn as nn
import torch.nn.functional as F

class TridiagonalResolventLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        # Initialize such that M is close to Identity
        self.d = nn.Parameter(torch.zeros(n_features))
        self.l = nn.Parameter(torch.zeros(n_features - 1))
        self.u = nn.Parameter(torch.zeros(n_features - 1))

    def forward(self, x):
        # x: (batch_size, n_features)
        # We solve My = x
        diag = F.softplus(self.d) + 2.0
        off_l = torch.tanh(self.l)
        off_u = torch.tanh(self.u)

        # Build M
        # Note: diag is (N,), off_l is (N-1,), off_u is (N-1,)
        # torch.diag creates a 2D tensor
        M = torch.diag(diag) + torch.diag(off_l, -1) + torch.diag(off_u, 1)

        # Solve My = x.T for all samples
        # x: (B, N) -> x.T: (N, B)
        # y.T: (N, B) -> y: (B, N)
        # torch.linalg.solve supports batching if M was (B, N, N),
        # but here M is (N, N) and x.T is (N, B), so it works as (N, N) @ (N, B) = (N, B)
        y = torch.linalg.solve(M, x.transpose(-2, -1)).transpose(-2, -1)
        return y

class TRLModel(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=100, output_dim=10):
        super().__init__()
        self.trl1 = TridiagonalResolventLayer(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.trl2 = TridiagonalResolventLayer(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.trl1(x)
        x = F.relu(self.fc1(x))
        x = self.trl2(x)
        x = self.fc2(x)
        return x

class MLPModel(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=100, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
