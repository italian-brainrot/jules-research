import torch
from torch.optim.optimizer import Optimizer

class CurveFitOptimizer(Optimizer):
    def __init__(self, params, base_optimizer, poly_degree=3, **kwargs):
        if not 0 <= poly_degree:
            raise ValueError("Invalid polynomial degree: {}".format(poly_degree))

        defaults = dict(poly_degree=poly_degree, **kwargs)
        super(CurveFitOptimizer, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('CurveFitOptimizer does not support sparse gradients')

                original_shape = grad.shape
                grad_flat = grad.flatten()

                sorted_grad, sort_indices = torch.sort(grad_flat)

                # Fit a polynomial to the sorted gradient
                x = torch.linspace(-1, 1, len(sorted_grad), device=grad.device, dtype=grad.dtype)

                # Vandermonde matrix
                V = torch.vander(x, N=group['poly_degree'] + 1)

                # Least squares fit
                try:
                    coeffs = torch.linalg.lstsq(V, sorted_grad).solution
                except torch.linalg.LinAlgError as e:
                    print(f"Warning: Least squares failed: {e}. Using original gradient.")
                    continue

                # Reconstruct the gradient from the polynomial
                reconstructed_sorted_grad = V @ coeffs

                # Un-sort the reconstructed gradient
                reconstructed_flat_grad = torch.empty_like(grad_flat)
                reconstructed_flat_grad[sort_indices] = reconstructed_sorted_grad

                # Reshape and replace the original gradient
                p.grad.data = reconstructed_flat_grad.reshape(original_shape)

        self.base_optimizer.step()
        return loss
