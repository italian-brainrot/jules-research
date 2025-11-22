import torch

def newton_schulz_sqrt(A, num_iters=5):
    """
    Computes the square root of a positive definite matrix A
    using the Newton-Schulz iteration.
    """
    A_norm = torch.norm(A, p='fro')
    Y = A / A_norm
    I = torch.eye(A.shape[0], device=A.device)
    Z = torch.eye(A.shape[0], device=A.device)

    for _ in range(num_iters):
        T = (3.0 * I - Z @ Y) / 2.0
        Y = Y @ T
        Z = T @ Z

    return Z * torch.sqrt(A_norm)

def orthogonalize(G, steps=5, eps=1e-7):
    """
    Orthogonalizes a matrix G using the Newton-Schulz iteration.
    Handles N-D tensors by reshaping them.
    """
    original_shape = G.shape

    G_reshaped = G
    if G.ndim > 2:
        G_reshaped = G.view(original_shape[0], -1)

    GG = G_reshaped @ G_reshaped.T
    GG_sqrt = newton_schulz_sqrt(GG, num_iters=steps)
    GG_sqrt_inv = torch.inverse(GG_sqrt + eps * torch.eye(G_reshaped.shape[0], device=G.device))
    ortho_G = GG_sqrt_inv @ G_reshaped

    return ortho_G.view(original_shape)
