import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, dim, variant='baseline'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.variant = variant
        # Use buffers for values we want to track but not as parameters
        self.register_buffer('last_cos_sim_sq', torch.tensor(0.0))
        self.register_buffer('last_avg_cos_sim', torch.tensor(0.0))

    def forward(self, x):
        v = self.net(x)

        # Compute cosine similarity for tracking and potential penalty
        dot = (x * v).sum(dim=-1)
        norm_x = torch.norm(x, dim=-1)
        norm_v = torch.norm(v, dim=-1)
        cos_sim = dot / (norm_x * norm_v + 1e-8)

        # We store the average values for tracking
        self.last_cos_sim_sq = (cos_sim**2).mean().detach()
        self.last_avg_cos_sim = cos_sim.mean().detach()

        if self.variant == 'baseline' or self.variant == 'penalty':
            # For 'penalty', the loss will be computed outside using the current cos_sim
            # So we need to return or store the non-detached version if needed.
            # Actually, to make it easier for autograd, we'll return it or store it in a way
            # that's accessible.
            self.current_cos_sim_sq = (cos_sim**2).mean()
            return x + v
        elif self.variant == 'forced':
            dot_k = dot.unsqueeze(-1)
            norm_x_sq = (x * x).sum(dim=-1, keepdim=True)
            v_orth = v - (dot_k / (norm_x_sq + 1e-8)) * x
            self.current_cos_sim_sq = torch.tensor(0.0, device=x.device)
            return x + v_orth
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

class ResMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks, variant='baseline'):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, variant=variant) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.variant = variant

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        return x

    def get_orth_loss(self):
        if self.variant != 'penalty':
            return torch.tensor(0.0, device=self.output_layer.weight.device)
        return sum(block.current_cos_sim_sq for block in self.blocks)

    def get_avg_cos_sim(self):
        return sum(block.last_avg_cos_sim.item() for block in self.blocks) / len(self.blocks)

    def get_avg_cos_sim_sq(self):
        return sum(block.last_cos_sim_sq.item() for block in self.blocks) / len(self.blocks)
