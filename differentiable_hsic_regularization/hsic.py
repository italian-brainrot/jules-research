import torch

def rbf_kernel(x, y, sigma=None):
    """
    Compute RBF kernel between x and y.
    x: (n1, d)
    y: (n2, d)
    """
    dist = torch.cdist(x, y, p=2)**2
    if sigma is None:
        # Median heuristic
        sigma = torch.median(dist[dist > 0]).sqrt() if torch.any(dist > 0) else 1.0

    gamma = 1.0 / (2 * sigma**2 + 1e-8)
    return torch.exp(-gamma * dist)

def hsic(x, y, sigma_x=None, sigma_y=None):
    """
    Compute Hilbert-Schmidt Independence Criterion (biased estimator).
    x: (n, d1)
    y: (n, d2)
    """
    n = x.size(0)
    if n < 2:
        return torch.tensor(0.0, device=x.device, requires_grad=True)

    K = rbf_kernel(x, x, sigma_x)
    L = rbf_kernel(y, y, sigma_y)

    H = torch.eye(n, device=x.device) - (1.0 / n) * torch.ones((n, n), device=x.device)

    # HSIC = 1/(n-1)^2 * Tr(K H L H)
    # Centering K: Kc = H K H
    Kc = H @ K @ H
    Lc = H @ L @ H

    hsic_val = torch.trace(Kc @ Lc) / ((n - 1) ** 2)
    return hsic_val

def hsic_normalized(x, y, sigma_x=None, sigma_y=None):
    """
    Normalized HSIC (HSIC / (std(K)*std(L)))
    Equivalent to Centered Kernel Alignment (CKA) if using linear kernels,
    but here we use RBF.
    """
    n = x.size(0)
    if n < 2:
        return torch.tensor(0.0, device=x.device, requires_grad=True)

    K = rbf_kernel(x, x, sigma_x)
    L = rbf_kernel(y, y, sigma_y)

    H = torch.eye(n, device=x.device) - (1.0 / n) * torch.ones((n, n), device=x.device)

    Kc = H @ K @ H
    Lc = H @ L @ H

    hsic_val = torch.trace(Kc @ Lc)
    norm_k = torch.norm(Kc, p='fro')
    norm_l = torch.norm(Lc, p='fro')

    return hsic_val / (norm_k * norm_l + 1e-8)
