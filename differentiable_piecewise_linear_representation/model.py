import torch
import torch.nn as nn
import torch.nn.functional as F

class DPLRLayer(nn.Module):
    def __init__(self, input_dim, num_segments, temperature=10.0):
        super(DPLRLayer, self).__init__()
        self.input_dim = input_dim
        self.num_segments = num_segments
        self.temperature = temperature

        # Initialize breakpoints evenly spaced
        initial_breakpoints = torch.linspace(0, input_dim, num_segments + 1)[1:-1]
        self.breakpoints = nn.Parameter(initial_breakpoints)

        # Time indices
        self.register_buffer('t', torch.arange(input_dim, dtype=torch.float32))

    def get_weights(self):
        # Ensure breakpoints are ordered and within bounds
        sorted_breakpoints = torch.sort(self.breakpoints)[0]
        sorted_breakpoints = torch.clamp(sorted_breakpoints, 0.1, self.input_dim - 1.1)

        # Concatenate boundaries
        all_breaks = torch.cat([torch.tensor([0.0], device=self.breakpoints.device),
                                sorted_breakpoints,
                                torch.tensor([float(self.input_dim)], device=self.breakpoints.device)])

        weights = []
        for i in range(self.num_segments):
            start = all_breaks[i]
            end = all_breaks[i+1]
            # Soft boxcar: sigmoid(t - start) - sigmoid(t - end)
            w = torch.sigmoid(self.temperature * (self.t - start)) - torch.sigmoid(self.temperature * (self.t - end))
            weights.append(w)

        return torch.stack(weights) # (num_segments, input_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        batch_size = x.size(0)
        weights = self.get_weights() # (num_segments, input_dim)

        features = []

        for i in range(self.num_segments):
            w = weights[i] # (input_dim,)

            # Weighted Least Squares for segment i: y = a*t + c
            # A = [t, 1], W = diag(w)
            # theta = (A^T W A)^-1 A^T W y

            sum_w = torch.sum(w) + 1e-6
            sum_wt = torch.sum(w * self.t)
            sum_wt2 = torch.sum(w * self.t**2)

            # Matrix M = A^T W A = [[sum_wt2, sum_wt], [sum_wt, sum_w]]
            # Determinant
            det = sum_wt2 * sum_w - sum_wt**2 + 1e-6

            # Inverse M^-1 = (1/det) * [[sum_w, -sum_wt], [-sum_wt, sum_wt2]]

            # A^T W y = [[sum_wtx], [sum_wx]]
            sum_wx = torch.sum(w * x, dim=1) # (batch_size,)
            sum_wtx = torch.sum(w * self.t * x, dim=1) # (batch_size,)

            # a = (1/det) * (sum_w * sum_wtx - sum_wt * sum_wx)
            a = (sum_w * sum_wtx - sum_wt * sum_wx) / det
            # c = (1/det) * (-sum_wt * sum_wtx + sum_wt2 * sum_wx)
            c = (-sum_wt * sum_wtx + sum_wt2 * sum_wx) / det

            features.append(a)
            features.append(c)

            # Optional: Mean Squared Error of the fit
            # fit = a*t + c
            fit = a.unsqueeze(1) * self.t.unsqueeze(0) + c.unsqueeze(1) # (batch_size, input_dim)
            mse = torch.sum(w * (x - fit)**2, dim=1) / sum_w
            features.append(mse)

        return torch.stack(features, dim=1) # (batch_size, num_segments * 3)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BaselineMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DPLRMLP(nn.Module):
    def __init__(self, input_dim, num_segments, hidden_dim, output_dim):
        super(DPLRMLP, self).__init__()
        self.dplr = DPLRLayer(input_dim, num_segments)
        self.net = nn.Sequential(
            nn.Linear(num_segments * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        feat = self.dplr(x)
        return self.net(feat)

class DPLRAugmentedMLP(nn.Module):
    def __init__(self, input_dim, num_segments, hidden_dim, output_dim):
        super(DPLRAugmentedMLP, self).__init__()
        self.dplr = DPLRLayer(input_dim, num_segments)
        self.net = nn.Sequential(
            nn.Linear(input_dim + num_segments * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        feat = self.dplr(x)
        combined = torch.cat([x, feat], dim=1)
        return self.net(combined)
