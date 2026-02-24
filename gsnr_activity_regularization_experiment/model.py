import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class GWARMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.h1 = None
        self.h2 = None
        self.logits = None

    def forward(self, x):
        self.h1 = F.relu(self.fc1(x))
        self.h2 = F.relu(self.fc2(self.h1))
        self.logits = self.fc3(self.h2)
        return self.logits

    def compute_gwar_loss(self, targets, lambda_gwar, epsilon=1e-8):
        # Calculate GSNR (detached)
        with torch.no_grad():
            probs = F.softmax(self.logits, dim=1)
            batch_size = targets.size(0)

            y_onehot = torch.zeros_like(probs)
            y_onehot.scatter_(1, targets.view(-1, 1), 1)

            g_logits = probs - y_onehot
            g_h2 = g_logits @ self.fc3.weight
            g_a2 = g_h2 * (self.h2 > 0).float()

            g_h1 = g_a2 @ self.fc2.weight
            g_a1 = g_h1 * (self.h1 > 0).float()

            def get_gsnr(g_h):
                B = g_h.size(0)
                sum_g = g_h.sum(dim=0)
                sum_g2 = (g_h**2).sum(dim=0)
                gsnr = (sum_g**2) / (B * sum_g2 + epsilon)
                return gsnr

            gsnr2 = get_gsnr(g_a2)
            gsnr1 = get_gsnr(g_a1)

        # Differentiable part: penalize activations weighted by (1 - GSNR)
        p2 = ((1 - gsnr2) * (self.h2**2).mean(dim=0)).sum()
        p1 = ((1 - gsnr1) * (self.h1**2).mean(dim=0)).sum()

        return lambda_gwar * (p1 + p2)
