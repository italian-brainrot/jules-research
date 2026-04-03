import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableTV1D(nn.Module):
    """
    Differentiable 1D Total Variation denoising layer using unrolled dual PGD.
    Solves: min_z  0.5 ||z - x||^2 + lambda * ||Dz||_1
    via the dual: min_y 0.5 ||x - D^T y||^2  s.t. ||y||_inf <= lambda
    where D is the finite difference operator.
    """
    def __init__(self, n_iters=10, initial_lambda=0.1, learnable=True):
        super().__init__()
        self.n_iters = n_iters
        if learnable:
            self.log_lambda = nn.Parameter(torch.tensor(initial_lambda).log())
        else:
            self.register_buffer('log_lambda', torch.tensor(initial_lambda).log())

    def forward(self, x):
        # x shape: (batch, seq_len)
        batch_size, seq_len = x.shape
        device = x.device
        lam = torch.exp(self.log_lambda)

        # Dual variable y has shape (batch, seq_len - 1)
        y = torch.zeros((batch_size, seq_len - 1), device=device)

        # Step size for PGD. For 1D TV, the operator norm of D^T D is at most 4.
        # So tau < 2/4 = 0.5 is safe.
        tau = 0.25

        for _ in range(self.n_iters):
            # Compute gradient of 0.5 ||x - D^T y||^2 w.r.t y
            # D^T y is:
            # (D^T y)_0 = -y_0
            # (D^T y)_i = y_{i-1} - y_i
            # (D^T y)_{N-1} = y_{N-2}

            # Efficiently compute D^T y
            DTy = torch.zeros_like(x)
            DTy[:, 0] = -y[:, 0]
            if seq_len > 2:
                DTy[:, 1:-1] = y[:, :-1] - y[:, 1:]
            DTy[:, -1] = y[:, -1]

            res = x - DTy

            # D res is: (res_1 - res_0, res_2 - res_1, ...)
            Dres = res[:, 1:] - res[:, :-1]

            # Gradient step: y = y + tau * Dres (since we want to minimize wrt y, and grad is -D(x - D^T y))
            # Wait, grad_y = -D(x - D^T y).
            # Step: y_{k+1} = proj(y_k - tau * grad_y) = proj(y_k + tau * D(x - D^T y))

            y = torch.clamp(y + tau * Dres, -lam, lam)

        # Final solution z = x - D^T y
        DTy = torch.zeros_like(x)
        DTy[:, 0] = -y[:, 0]
        if seq_len > 2:
            DTy[:, 1:-1] = y[:, :-1] - y[:, 1:]
        DTy[:, -1] = y[:, -1]

        return x - DTy

class TVDenoisingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_iters=10, learnable_tv=True):
        super().__init__()
        self.tv = DifferentiableTV1D(n_iters=n_iters, learnable=learnable_tv)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (batch, input_dim)
        x_denoised = self.tv(x)
        # We can also concatenate if we want to keep raw info
        # combined = torch.cat([x, x_denoised], dim=1)
        # return self.mlp(combined)
        return self.mlp(x_denoised)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
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
