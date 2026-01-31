import torch
import torch.nn as nn
import numpy as np

class BisineNetwork(nn.Module):
    def __init__(self, input_dim, num_classes, num_units=1):
        super(BisineNetwork, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_units = num_units
        self.params_per_unit = 2 * input_dim + 3
        self.params_per_class = num_units * self.params_per_unit

        # Flattened parameters for easy Hessian computation
        # Shape: (num_classes, params_per_class)
        self.params = nn.Parameter(torch.randn(num_classes, self.params_per_class) * 0.1)

    def forward(self, x):
        # x: (batch_size, input_dim)
        batch_size = x.shape[0]
        z = torch.zeros(batch_size, self.num_classes, device=x.device)

        for c in range(self.num_classes):
            p_c = self.params[c]
            z[:, c] = self._forward_class(x, p_c)

        return z

    def _forward_class(self, x, p_c):
        # p_c: (params_per_class,)
        z_c = torch.zeros(x.shape[0], device=x.device)
        for k in range(self.num_units):
            start = k * self.params_per_unit
            a = p_c[start]
            w1 = p_c[start + 1 : start + 1 + self.input_dim]
            b1 = p_c[start + 1 + self.input_dim]
            w2 = p_c[start + 2 + self.input_dim : start + 2 + 2 * self.input_dim]
            b2 = p_c[start + 2 + 2 * self.input_dim]

            u1 = torch.matmul(x, w1) + b1
            u2 = torch.matmul(x, w2) + b2
            z_c += a * torch.sin(u1) * torch.sin(u2)
        return z_c

    def get_flat_params(self):
        return self.params.view(-1)

    def set_flat_params(self, flat_params):
        self.params.data = flat_params.view(self.num_classes, self.params_per_class)

    def compute_grad_and_hessian(self, x):
        """
        Computes gradient and Hessian for each class output separately.
        Returns:
            G: (batch_size, num_classes, params_per_class)
            H: (batch_size, num_classes, params_per_class, params_per_class)
        """
        N = x.shape[0]
        D = self.input_dim
        C = self.num_classes
        K = self.num_units
        P_c = self.params_per_class
        P_u = self.params_per_unit

        G = torch.zeros(N, C, P_c, device=x.device)
        H = torch.zeros(N, C, P_c, P_c, device=x.device)

        xx = torch.bmm(x.unsqueeze(2), x.unsqueeze(1)) # (N, D, D)

        for c in range(C):
            p_c = self.params[c]
            for k in range(K):
                start = k * P_u
                a = p_c[start]
                w1 = p_c[start + 1 : start + 1 + D]
                b1 = p_c[start + 1 + D]
                w2 = p_c[start + 2 + D : start + 2 + 2 * D]
                b2 = p_c[start + 2 + 2 * D]

                u1 = torch.matmul(x, w1) + b1
                u2 = torch.matmul(x, w2) + b2

                s1, s2 = torch.sin(u1), torch.sin(u2)
                c1, c2 = torch.cos(u1), torch.cos(u2)

                # Gradient
                G[..., c, start] = s1 * s2
                G[..., c, start + 1 : start + 1 + D] = (a * c1 * s2).unsqueeze(1) * x
                G[..., c, start + 1 + D] = a * c1 * s2
                G[..., c, start + 2 + D : start + 2 + 2 * D] = (a * s1 * c2).unsqueeze(1) * x
                G[..., c, start + 2 + 2 * D] = a * s1 * c2

                # Hessian
                idx_a = start
                idx_w1 = slice(start + 1, start + 1 + D)
                idx_b1 = start + 1 + D
                idx_w2 = slice(start + 2 + D, start + 2 + 2 * D)
                idx_b2 = start + 2 + 2 * D

                H[..., c, idx_a, idx_w1] = (c1 * s2).unsqueeze(1) * x
                H[..., c, idx_a, idx_b1] = c1 * s2
                H[..., c, idx_a, idx_w2] = (s1 * c2).unsqueeze(1) * x
                H[..., c, idx_a, idx_b2] = s1 * c2

                H[..., c, idx_w1, idx_a] = H[..., c, idx_a, idx_w1]
                H[..., c, idx_b1, idx_a] = H[..., c, idx_a, idx_b1]
                H[..., c, idx_w2, idx_a] = H[..., c, idx_a, idx_w2]
                H[..., c, idx_b2, idx_a] = H[..., c, idx_a, idx_b2]

                val_w1w1 = -a * s1 * s2
                H[..., c, idx_w1, idx_w1] = val_w1w1.view(N, 1, 1) * xx
                H[..., c, idx_w1, idx_b1] = val_w1w1.unsqueeze(1) * x
                H[..., c, idx_b1, idx_w1] = H[..., c, idx_w1, idx_b1]
                H[..., c, idx_b1, idx_b1] = val_w1w1

                val_w1w2 = a * c1 * c2
                H[..., c, idx_w1, idx_w2] = val_w1w2.view(N, 1, 1) * xx
                H[..., c, idx_w2, idx_w1] = H[..., c, idx_w1, idx_w2]
                H[..., c, idx_w1, idx_b2] = val_w1w2.unsqueeze(1) * x
                H[..., c, idx_b2, idx_w1] = H[..., c, idx_w1, idx_b2]
                H[..., c, idx_b1, idx_w2] = val_w1w2.unsqueeze(1) * x
                H[..., c, idx_w2, idx_b1] = H[..., c, idx_b1, idx_w2]
                H[..., c, idx_b1, idx_b2] = val_w1w2
                H[..., c, idx_b2, idx_b1] = val_w1w2

                val_w2w2 = -a * s1 * s2
                H[..., c, idx_w2, idx_w2] = val_w2w2.view(N, 1, 1) * xx
                H[..., c, idx_w2, idx_b2] = val_w2w2.unsqueeze(1) * x
                H[..., c, idx_b2, idx_w2] = H[..., c, idx_w2, idx_b2]
                H[..., c, idx_b2, idx_b2] = val_w2w2

        return G, H
