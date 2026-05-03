import torch
import torch.nn as nn
import torch.nn.functional as F

class DLCALayer(nn.Module):
    def __init__(self, num_levels=10, beta=10.0, learn_levels=True, learn_beta=True):
        super().__init__()
        self.num_levels = num_levels

        # Initialize levels uniformly between -1 and 1 (typical range for normalized signals)
        levels = torch.linspace(-1, 1, num_levels)
        if learn_levels:
            self.levels = nn.Parameter(levels)
        else:
            self.register_buffer('levels', levels)

        if learn_beta:
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer('beta', torch.tensor(beta))

    def forward(self, x):
        # x: (batch, seq_len)
        batch_size, seq_len = x.shape

        # x_t: (batch, seq_len - 1)
        # x_next: (batch, seq_len - 1)
        x_t = x[:, :-1]
        x_next = x[:, 1:]

        # levels: (num_levels,)
        # Reshape for broadcasting
        # x_t, x_next: (batch, 1, seq_len - 1)
        # levels: (1, num_levels, 1)
        x_t = x_t.unsqueeze(1)
        x_next = x_next.unsqueeze(1)
        levels = self.levels.view(1, -1, 1)

        # Upward crossings: x_t < level and x_next > level
        # cross_up = sigma(beta * (x_next - level)) * sigma(beta * (level - x_t))
        cross_up = torch.sigmoid(self.beta * (x_next - levels)) * torch.sigmoid(self.beta * (levels - x_t))

        # Downward crossings: x_t > level and x_next < level
        # cross_down = sigma(beta * (level - x_next)) * sigma(beta * (x_t - level))
        cross_down = torch.sigmoid(self.beta * (levels - x_next)) * torch.sigmoid(self.beta * (x_t - levels))

        # Sum over time to get total crossings per level
        # (batch, num_levels)
        total_up = cross_up.sum(dim=-1)
        total_down = cross_down.sum(dim=-1)

        # Features: [total_up, total_down] or [total_up + total_down]
        # Let's return both concatenated
        features = torch.cat([total_up, total_down], dim=1)

        return features
