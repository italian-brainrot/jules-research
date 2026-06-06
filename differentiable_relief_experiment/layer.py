import torch
import torch.nn as nn
import torch.nn.functional as F

class DReliefLoss(nn.Module):
    """
    Differentiable Relief-inspired Loss.
    Encourages local neighborhood consistency in the feature space.
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, targets):
        """
        features: (B, D)
        targets: (B,)
        """
        B, D = features.shape
        if B < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # Compute pairwise squared Euclidean distances
        # dists[i, j] = ||f_i - f_j||^2
        dist_sq = torch.cdist(features, features, p=2)**2

        # Masks for hits (same class) and misses (different class)
        # Exclude self-distance by setting diagonal to a large value
        mask_same = (targets.unsqueeze(0) == targets.unsqueeze(1))
        mask_diff = ~mask_same

        # Set diagonal to False for hits to avoid picking self
        mask_same.fill_diagonal_(False)

        # Soft hits
        # For each i, compute softmax over same-class neighbors
        # Weighting: exp(-dist^2 / tau)
        hits_weights = F.softmax(-dist_sq / self.temperature + (~mask_same).float() * -1e9, dim=1)
        # Expected hit distance for each i
        dist_hit = torch.sum(hits_weights * dist_sq, dim=1)

        # Soft misses
        # ReliefF typically finds nearest misses for each other class
        unique_classes = torch.unique(targets)
        num_classes = len(unique_classes)

        miss_dists = []
        for c in unique_classes:
            # Mask for samples of class c
            mask_c = (targets == c)

            # We only care about i where target[i] != c
            # But for implementation ease, we can compute it for all and then mask

            # Softmax over samples of class c
            miss_c_weights = F.softmax(-dist_sq / self.temperature + (~mask_c).unsqueeze(0).float() * -1e9, dim=1)
            dist_miss_c = torch.sum(miss_c_weights * dist_sq, dim=1)
            miss_dists.append(dist_miss_c)

        miss_dists = torch.stack(miss_dists, dim=1) # (B, num_classes)

        # For each sample i, we want the average distance to misses of other classes
        # The class of i is targets[i]. We want average over c != targets[i].
        # We can use another mask
        target_indices = torch.searchsorted(unique_classes, targets)
        mask_other_classes = torch.ones((B, num_classes), device=features.device)
        mask_other_classes.scatter_(1, target_indices.unsqueeze(1), 0)

        # Average distance to soft-misses of other classes
        dist_miss = torch.sum(miss_dists * mask_other_classes, dim=1) / (num_classes - 1 + 1e-8)

        # Relief objective is usually to maximize (MissDist - HitDist)
        # So loss is (HitDist - MissDist)
        loss = torch.mean(dist_hit - dist_miss)

        return loss
