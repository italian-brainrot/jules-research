import torch
import torch.nn as nn
import torch.nn.functional as F

class MarkovStationaryPooling(nn.Module):
    def __init__(self, d_model, num_iters=20, d_keys=None, include_entropy=False):
        super().__init__()
        self.d_model = d_model
        self.d_keys = d_keys or d_model
        self.num_iters = num_iters
        self.include_entropy = include_entropy

        self.q_proj = nn.Linear(d_model, self.d_keys)
        self.k_proj = nn.Linear(d_model, self.d_keys)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape

        q = self.q_proj(x) # [B, L, Dk]
        k = self.k_proj(x) # [B, L, Dk]
        v = self.v_proj(x) # [B, L, D]

        # Compute transition matrix M
        # A_ij = q_i^T k_j
        attn_logits = torch.bmm(q, k.transpose(-1, -2)) / (self.d_keys ** 0.5)
        M = F.softmax(attn_logits, dim=-1) # [B, L, L] row-stochastic

        # Power iteration for stationary distribution pi
        # pi_t+1 = pi_t M
        pi = torch.ones(B, 1, L, device=x.device) / L
        for _ in range(self.num_iters):
            pi = torch.bmm(pi, M)

        pi = pi.squeeze(1) # [B, L]

        # Pooled output
        out = torch.bmm(pi.unsqueeze(1), v).squeeze(1) # [B, D]

        if self.include_entropy:
            # Entropy of each state: H_i = -sum_j M_ij log M_ij
            # Avoid log(0)
            H = -torch.sum(M * torch.log(M + 1e-9), dim=-1) # [B, L]
            # Entropy rate: h = sum_i pi_i H_i
            h = torch.sum(pi * H, dim=-1, keepdim=True) # [B, 1]
            # Combine entropy rate with output
            # We can either concatenate or add a projected value.
            # Let's concatenate and use a linear layer to keep dimensions consistent if needed,
            # but for simplicity let's just use it as an extra feature.
            # In our model we will just handle the extra dimension.
            return out, h

        return out

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        q = self.query.expand(B, -1, -1) # [B, 1, D]
        k = self.k_proj(x) # [B, L, D]
        v = self.v_proj(x) # [B, L, D]

        attn_logits = torch.bmm(q, k.transpose(-1, -2)) / (D ** 0.5)
        attn_weights = F.softmax(attn_logits, dim=-1) # [B, 1, L]

        out = torch.bmm(attn_weights, v).squeeze(1) # [B, D]
        return out
