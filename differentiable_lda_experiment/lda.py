import torch
import torch.nn as nn

class DLDALoss(nn.Module):
    """
    Differentiable Linear Discriminant Analysis Loss.
    Encourages features to have high between-class variance and low within-class variance.
    """
    def __init__(self, reg=1e-5):
        super().__init__()
        self.reg = reg

    def forward(self, features, targets):
        """
        features: (B, D)
        targets: (B,)
        """
        B, D = features.shape
        unique_classes = torch.unique(targets)
        num_classes = len(unique_classes)

        if num_classes < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        mean_total = features.mean(dim=0, keepdim=True)

        sw = torch.zeros((D, D), device=features.device)
        sb = torch.zeros((D, D), device=features.device)

        for c in unique_classes:
            mask = (targets == c)
            class_features = features[mask]
            n_c = class_features.shape[0]
            if n_c < 1:
                continue

            mean_c = class_features.mean(dim=0, keepdim=True)

            # Within-class scatter
            diff_w = class_features - mean_c
            sw += diff_w.t() @ diff_w

            # Between-class scatter
            diff_b = mean_c - mean_total
            sb += n_c * (diff_b.t() @ diff_b)

        # Normalize by batch size
        sw /= B
        sb /= B

        # Regularize Sw for stability
        sw = sw + self.reg * torch.eye(D, device=features.device)

        # We want to maximize Tr(Sw^-1 Sb)
        # To make it differentiable and stable, we can solve the generalized eigenvalue problem
        # Or just compute Tr(Sw^-1 Sb).
        # Using torch.linalg.solve is usually better than explicit inverse.

        # Tr(Sw^-1 Sb) = sum of generalized eigenvalues
        # We use a trick: Tr(A^-1 B) = Tr(B A^-1)
        # sw_inv_sb = torch.linalg.solve(sw, sb)
        # loss = -torch.trace(sw_inv_sb)

        # Alternatively, use Cholesky for better stability if D is large,
        # but here D is likely small (hidden layer size).
        try:
            L = torch.linalg.cholesky(sw)
            # Sw = L L^T
            # Sw^-1 Sb = (L L^T)^-1 Sb = L^-T L^-1 Sb
            # Tr(L^-T L^-1 Sb) = Tr(L^-1 Sb L^-T)
            # Let M = L^-1 Sb L^-T. M is symmetric if Sb is symmetric.
            tmp = torch.linalg.solve_triangular(L, sb, upper=False)
            M = torch.linalg.solve_triangular(L, tmp.t(), upper=False).t()
            # The eigenvalues of M are the generalized eigenvalues of (Sb, Sw)
            # Since we only need the trace, we can just sum the diagonal of M
            loss = -torch.trace(M)
        except RuntimeError:
            # Fallback if Cholesky fails
            sw_inv_sb = torch.linalg.solve(sw, sb)
            loss = -torch.trace(sw_inv_sb)

        return loss
