import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableDelaySignature(nn.Module):
    """
    Differentiable Delay Signature (DDS) Layer.
    Lifts a 1D signal into a D-dimensional space using a learnable delay tau,
    and computes the path signature of the resulting trajectory.
    """
    def __init__(self, d=3, k=2, initial_tau=1.0):
        super().__init__()
        self.d = d # Embedding dimension
        self.k = k # Signature depth
        self.log_tau = nn.Parameter(torch.log(torch.tensor(initial_tau)))

    def _get_delayed_signal(self, x, tau):
        # x: (B, L)
        B, L = x.shape
        device = x.device

        # We want to create D versions of the signal: x(t), x(t-tau), x(t-2tau), ...
        # t will go from (D-1)*tau to L-1
        max_delay = (self.d - 1) * tau
        num_points = 40

        # Actually, we can use a fixed number of samples from the continuous-ish path
        t = torch.linspace(max_delay, L - 1, steps=num_points, device=device) # (num_points,)

        delays = torch.arange(self.d, device=device).float() * tau # (d,)

        # Grid of sample points: (d, num_points)
        sample_indices = t.unsqueeze(0) - delays.unsqueeze(1) # (d, num_points)

        # Flatten for interpolation
        sample_indices_flat = sample_indices.flatten() # (d * num_points)

        # Linear interpolation
        idx_floor = torch.floor(sample_indices_flat).long()
        idx_ceil = idx_floor + 1

        # Clamp to avoid out of bounds
        idx_floor = torch.clamp(idx_floor, 0, L - 1)
        idx_ceil = torch.clamp(idx_ceil, 0, L - 1)

        frac = sample_indices_flat - idx_floor.float()

        # x: (B, L) -> gather along dim 1
        val_floor = torch.gather(x, 1, idx_floor.unsqueeze(0).expand(B, -1))
        val_ceil = torch.gather(x, 1, idx_ceil.unsqueeze(0).expand(B, -1))

        samples = val_floor * (1 - frac) + val_ceil * frac

        # Reshape to (B, d, num_points)
        samples = samples.view(B, self.d, num_points)
        return samples

    def _compute_signature(self, path):
        # path: (B, D, N)
        B, D, N = path.shape
        device = path.device

        # Compute increments
        # dX: (B, D, N-1)
        dX = path[:, :, 1:] - path[:, :, :-1]
        num_segments = N - 1

        # Signature depth 1: (B, D)
        # Sum of increments
        S1 = dX.sum(dim=-1)

        if self.k == 1:
            return S1

        if self.k == 2:
            # Signature depth 2: (B, D, D)
            # S2_{i,j} = sum_{t < u} dX_t^i * dX_u^j + 1/2 sum_t dX_t^i * dX_t^j
            # We can use a simpler way:
            # Let X_t be the path at time t.
            # S2_{i,j} = integral_{0 < t < u < T} dX_t^i dX_u^j
            # For piecewise linear segments:
            # S2 = sum_{n} (S1_{prev} \otimes dX_n + 1/2 dX_n \otimes dX_n)

            S2 = torch.zeros(B, D, D, device=device)
            S1_accum = torch.zeros(B, D, device=device)

            for n in range(num_segments):
                dx_n = dX[:, :, n] # (B, D)

                # Update S2: S2 = S2 + S1_accum \otimes dx_n + 0.5 * dx_n \otimes dx_n
                term1 = torch.bmm(S1_accum.unsqueeze(2), dx_n.unsqueeze(1))
                term2 = 0.5 * torch.bmm(dx_n.unsqueeze(2), dx_n.unsqueeze(1))

                S2 = S2 + term1 + term2
                S1_accum = S1_accum + dx_n

            return torch.cat([S1, S2.view(B, -1)], dim=1)

        raise NotImplementedError("Signature depth > 2 not implemented yet")

    def forward(self, x):
        # x: (B, L)
        tau = torch.exp(self.log_tau)
        path = self._get_delayed_signal(x, tau)
        sig = self._compute_signature(path)
        return sig

class DDSRNet(nn.Module):
    def __init__(self, input_dim=40, d=3, k=2, hidden_dim=256, output_dim=10):
        super().__init__()
        self.dds = DifferentiableDelaySignature(d=d, k=k)
        # d=3, k=2 -> 3 + 9 = 12 features
        sig_dim = d + d*d if k==2 else d

        self.mlp = nn.Sequential(
            nn.Linear(sig_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        sig = self.dds(x)
        return self.mlp(sig)

class DDSAugmentedMLP(nn.Module):
    def __init__(self, input_dim=40, d=3, k=2, hidden_dim=256, output_dim=10):
        super().__init__()
        self.dds = DifferentiableDelaySignature(d=d, k=k)
        sig_dim = d + d*d if k==2 else d

        self.mlp = nn.Sequential(
            nn.Linear(input_dim + sig_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        sig = self.dds(x)
        combined = torch.cat([x, sig], dim=1)
        return self.mlp(combined)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)
