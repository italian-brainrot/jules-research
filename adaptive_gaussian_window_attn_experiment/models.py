import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaptiveGaussianWindowAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.log_lambda = nn.Parameter(torch.full((n_heads,), -2.0)) # Start with a somewhat broad window
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        B, L, _ = q.shape
        Q = self.q_proj(q).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(k).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(v).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_head)

        grid = torch.arange(L, device=q.device).float()
        dist_sq = (grid.unsqueeze(0) - grid.unsqueeze(1))**2

        lambdas = torch.exp(self.log_lambda).view(1, self.n_heads, 1, 1)
        bias = - lambdas * dist_sq.view(1, 1, L, L)

        scores = scores + bias

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out), attn

class DynamicAdaptiveGaussianWindowAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.lambda_net = nn.Linear(d_model, n_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        B, L, _ = q.shape
        Q = self.q_proj(q).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(k).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(v).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_head)

        grid = torch.arange(L, device=q.device).float()
        dist_sq = (grid.unsqueeze(0) - grid.unsqueeze(1))**2

        # lambdas depend on query
        lambdas = F.softplus(self.lambda_net(q)) # (B, L, n_heads)
        lambdas = lambdas.transpose(1, 2).unsqueeze(-1) # (B, n_heads, L, 1)

        bias = - lambdas * dist_sq.view(1, 1, L, L) # (B, n_heads, L, L)

        scores = scores + bias

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out), attn

class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, attn_type):
        super().__init__()
        if attn_type == 'standard':
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        elif attn_type == 'agwa':
            self.attn = AdaptiveGaussianWindowAttention(d_model, n_heads, dropout=dropout)
        elif attn_type == 'dagwa':
            self.attn = DynamicAdaptiveGaussianWindowAttention(d_model, n_heads, dropout=dropout)
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)

        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, n_tokens=40, d_model=64, n_heads=4, n_layers=2, n_classes=10, dropout=0.1, attn_type='standard'):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_tokens, d_model))

        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, dropout, attn_type)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        # x: (B, 40, 1)
        x = self.embedding(x)
        x = x + self.pos_encoding

        for layer in self.layers:
            x = layer(x)

        x = x.mean(dim=1)
        x = self.norm(x)
        return self.fc(x)
