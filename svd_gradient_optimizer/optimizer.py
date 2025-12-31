import torch

class SVDGradientOptimizer:
    """
    A wrapper optimizer that simplifies gradients using Singular Value Decomposition (SVD).
    """
    def __init__(self, base_optimizer, k):
        """
        Args:
            base_optimizer (torch.optim.Optimizer): The base optimizer (e.g., Adam).
            k (int): The number of singular values to use for gradient reconstruction.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"Rank k must be a positive integer, but got {k}")
        self.base_optimizer = base_optimizer
        self.k = k
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = self.base_optimizer.defaults

    def zero_grad(self, set_to_none: bool = False):
        """Clears the gradients of all optimized a torch.Tensor s."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that re-evaluates the model
                                          and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.dim() < 2:
                    # SVD is not applicable to vectors or scalars, so we skip them.
                    continue

                # Perform SVD on the gradient
                U, S, Vh = torch.linalg.svd(grad, full_matrices=False)

                # Determine the rank for reconstruction (cannot be > number of singular values)
                rank = min(self.k, S.size(0))

                # Reconstruct the gradient using the top-k singular values
                S_k = torch.zeros_like(S)
                S_k[:rank] = S[:rank]
                grad_k = U @ torch.diag(S_k) @ Vh

                # Replace the original gradient with the low-rank approximation
                p.grad = grad_k

        self.base_optimizer.step()
        return loss

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
