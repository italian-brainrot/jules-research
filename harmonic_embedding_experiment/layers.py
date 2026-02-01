import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HarmonicParameterizedEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, K):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.K = K

        # Per-token parameters
        self.amplitudes = nn.Parameter(torch.randn(num_embeddings, K) / math.sqrt(K))
        self.frequencies = nn.Parameter(torch.randn(num_embeddings, K) * math.pi)
        self.phases = nn.Parameter(torch.rand(num_embeddings, K) * 2 * math.pi)

        # Grid of dimension indices [0, 1]
        self.register_buffer('grid', torch.linspace(0, 1, embedding_dim))

    def forward(self, x):
        # x: [batch, seq_len]
        a = self.amplitudes[x]    # [batch, seq_len, K]
        w = self.frequencies[x]   # [batch, seq_len, K]
        phi = self.phases[x]     # [batch, seq_len, K]

        # grid: [D]
        # We compute sum_{k} a_k * sin(w_k * grid_j + phi_k)
        # Using broadcasting:
        # a, w, phi: [batch, seq_len, K, 1]
        # grid: [1, 1, 1, D]

        a = a.unsqueeze(-1)
        w = w.unsqueeze(-1)
        phi = phi.unsqueeze(-1)
        grid = self.grid.view(1, 1, 1, -1)

        # [batch, seq_len, K, D]
        inner = w * grid + phi
        out = a * torch.sin(inner)

        # Sum over oscillators: [batch, seq_len, D]
        return out.sum(dim=-2)

class FactorizedEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, K):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, K)
        self.proj = nn.Linear(K, embedding_dim, bias=False)

    def forward(self, x):
        return self.proj(self.emb(x))

class StandardEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        return self.emb(x)
