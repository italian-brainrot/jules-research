import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NLMFLinear(nn.Module):
    """
    Non-Linear Matrix Factorization Linear Layer.
    W = phi(U V^T) where phi is an element-wise MLP.
    """
    def __init__(self, in_features, out_features, rank=1, hidden_dim=16, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Initialize U, V so that Var(U V^T) is roughly 2/in_features
        # Var(W_ij) = rank * Var(U_ik) * Var(V_jk)
        # We want rank * Var(U_ik) * Var(V_jk) = 2 / in_features
        # Let Var(U_ik) = Var(V_jk) = (2 / (in_features * rank^2))^0.25 ? No.
        # Let Var(U_ik) = Var(V_jk) = sqrt(2 / (in_features * rank))
        std = (2.0 / (in_features * rank))**0.25
        self.U = nn.Parameter(torch.randn(out_features, rank) * std)
        self.V = nn.Parameter(torch.randn(in_features, rank) * std)

        # Element-wise MLP phi
        # We use Conv2d with 1x1 kernel as an efficient way to apply an MLP element-wise
        self.phi = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, 1, 1)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # Generate weight matrix W = phi(U V^T)
        W_pre = torch.matmul(self.U, self.V.t()) # (out_features, in_features)
        # Apply phi element-wise
        W = self.phi(W_pre.unsqueeze(0).unsqueeze(0)) # (1, 1, out_features, in_features)
        W = W.squeeze(0).squeeze(0)

        return F.linear(x, W, self.bias)

class LowRankLinear(nn.Module):
    """
    Standard Low-Rank Linear Layer.
    W = U V^T
    """
    def __init__(self, in_features, out_features, rank=1, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        std = (2.0 / (in_features * rank))**0.25
        self.U = nn.Parameter(torch.randn(out_features, rank) * std)
        self.V = nn.Parameter(torch.randn(in_features, rank) * std)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        W = torch.matmul(self.U, self.V.t())
        return F.linear(x, W, self.bias)

class KroneckerLinear(nn.Module):
    """
    Kronecker Product Linear Layer.
    W = A \otimes B
    """
    def __init__(self, m1, n1, m2, n2, bias=True):
        super().__init__()
        self.m1, self.n1, self.m2, self.n2 = m1, n1, m2, n2
        # in_features = n1 * n2, out_features = m1 * m2

        std = (2.0 / (n1 * n2))**0.25
        self.A = nn.Parameter(torch.randn(m1, n1) * std)
        self.B = nn.Parameter(torch.randn(m2, n2) * std)

        if bias:
            self.bias = nn.Parameter(torch.zeros(m1 * m2))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        batch_size = x.shape[0]
        # Reshape x to (batch, n1, n2)
        X = x.view(batch_size, self.n1, self.n2)
        # Y = A X B^T
        Y = torch.matmul(self.A, X) # (batch, m1, n2)
        Y = torch.matmul(Y, self.B.t()) # (batch, m1, m2)
        y = Y.reshape(batch_size, -1)
        if self.bias is not None:
            y = y + self.bias
        return y

class DenseLinear(nn.Module):
    """
    Wrapper for standard nn.Linear.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x)
