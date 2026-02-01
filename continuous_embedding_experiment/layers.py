import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FourierFeatures(nn.Module):
    def __init__(self, input_dim, output_dim, sigma=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.D = output_dim // input_dim
        if self.D % 2 != 0:
             self.D += 1

        self.register_buffer('B', torch.randn(input_dim, self.D // 2) * sigma)

    def forward(self, x):
        # x: [batch, input_dim]
        x_proj = x.unsqueeze(-1) * self.B.unsqueeze(0)
        out = torch.cat([torch.sin(2 * math.pi * x_proj), torch.cos(2 * math.pi * x_proj)], dim=-1)
        return out.view(x.size(0), -1)

class LinearInterpolationEmbedding(nn.Module):
    def __init__(self, input_dim, num_embeddings, embedding_dim, value_range=(-6.0, 6.0)):
        super().__init__()
        self.input_dim = input_dim
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.value_range = value_range
        self.embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, x):
        v_min, v_max = self.value_range
        x_scaled = (x - v_min) / (v_max - v_min) * (self.num_embeddings - 1)
        x_scaled = torch.clamp(x_scaled, 0, self.num_embeddings - 1)

        idx_low = x_scaled.long()
        idx_high = torch.min(idx_low + 1, torch.tensor(self.num_embeddings - 1, device=x.device))

        weight_high = x_scaled - idx_low.float()
        weight_low = 1.0 - weight_high

        # self.embeddings: [num_embeddings, embedding_dim]
        # idx_low: [batch, input_dim]
        emb_low = self.embeddings[idx_low]   # [batch, input_dim, embedding_dim]
        emb_high = self.embeddings[idx_high] # [batch, input_dim, embedding_dim]

        out = weight_low.unsqueeze(-1) * emb_low + weight_high.unsqueeze(-1) * emb_high
        return out.view(x.size(0), -1)

class DACE(nn.Module):
    """Differentiable Anchor-based Continuous Embedding"""
    def __init__(self, input_dim, num_embeddings, embedding_dim, value_range=(-6.0, 6.0), gamma=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        v_min, v_max = value_range
        anchors = torch.linspace(v_min, v_max, num_embeddings)
        self.anchors = nn.Parameter(anchors)

        self.embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))

    def forward(self, x):
        # x: [batch, input_dim]
        # anchors: [num_embeddings]
        dist_sq = (x.unsqueeze(-1) - self.anchors.view(1, 1, -1))**2
        weights = F.softmax(-torch.abs(self.gamma) * dist_sq, dim=-1) # [batch, input_dim, num_embeddings]

        # weights: [batch, input_dim, num_embeddings]
        # embeddings: [num_embeddings, embedding_dim]
        out = torch.matmul(weights, self.embeddings) # [batch, input_dim, embedding_dim]
        return out.view(x.size(0), -1)

    def get_smoothness_loss(self):
        sorted_anchors, indices = torch.sort(self.anchors)
        sorted_embeddings = self.embeddings[indices]
        diffs = sorted_embeddings[1:] - sorted_embeddings[:-1]
        return torch.mean(diffs**2)
