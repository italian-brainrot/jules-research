import torch
import torch.nn as nn
import torch.nn.functional as F

class DLOFLoss(nn.Module):
    """
    Differentiable Local Outlier Factor Loss.
    Computes LOF for a batch of embeddings and returns a penalty.
    Ideally, for in-distribution data, LOF should be close to 1.
    """
    def __init__(self, k=5, eps=1e-6):
        super().__init__()
        self.k = k
        self.eps = eps

    def forward(self, x):
        # x: (B, D)
        B, D = x.shape
        if B <= self.k:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        # Compute pairwise distances
        dist_matrix = torch.cdist(x, x, p=2) # (B, B)

        # Find k-nearest neighbors (excluding self)
        # topk returns values and indices. We want the k+1 smallest distances,
        # and then exclude the smallest (which is 0 distance to self).
        values, indices = torch.topk(dist_matrix, k=self.k + 1, largest=False, sorted=True)

        # distances to k-nearest neighbors: (B, k)
        knn_distances = values[:, 1:]
        knn_indices = indices[:, 1:]

        # k-distance(B): (B,)
        k_distances = knn_distances[:, -1]

        # reach_dist_k(A, B) = max(k-distance(B), dist(A, B))
        # We need reach_dist for each A and its neighbors B.
        # k_distances[knn_indices]: (B, k) - k-distance of each neighbor
        reach_dist = torch.max(k_distances[knn_indices], knn_distances) # (B, k)

        # lrd_k(A) = 1 / (mean(reach_dist_k(A, B)))
        lrd = 1.0 / (torch.mean(reach_dist, dim=1) + self.eps) # (B,)

        # LOF_k(A) = mean(lrd_k(B) / lrd_k(A))
        # lrd[knn_indices]: (B, k) - lrd of each neighbor
        lof = torch.mean(lrd[knn_indices], dim=1) / (lrd + self.eps) # (B,)

        # Penalty: how much LOF deviates from 1.0
        # We want to minimize LOF for in-distribution batch
        loss = torch.mean((lof - 1.0)**2)
        return loss

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class LOFRegularizedMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10, k=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.lof_loss_fn = DLOFLoss(k=k)
        self.last_lof_loss = torch.tensor(0.0)

    def forward(self, x, return_lof=False):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))

        if self.training:
            self.last_lof_loss = self.lof_loss_fn(h2)

        out = self.fc3(h2)
        if return_lof:
            return out, self.last_lof_loss
        return out
