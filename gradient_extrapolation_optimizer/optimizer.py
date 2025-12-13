import torch
from torch.optim.optimizer import Optimizer
from collections import deque

class GPE(Optimizer):
    """
    Gradient Polynomial Extrapolation (GPE) Optimizer.

    This optimizer wraps a base optimizer and uses polynomial extrapolation on the
    gradient history to potentially accelerate convergence.
    """
    def __init__(self, params, base_optimizer, history_size=10, degree=2, alpha=0.5):
        """
        Initializes the GPE optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups.
            base_optimizer: The base optimizer (e.g., Adam, SGD).
            history_size (int): The number of past gradients to use for extrapolation.
            degree (int): The degree of the polynomial to fit.
            alpha (float): The interpolation factor between the current gradient
                           and the extrapolated gradient. 0.0 means only the current
                           gradient is used, 1.0 means only the extrapolation is used.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if not history_size >= degree + 1:
            raise ValueError("History size must be at least degree + 1")

        defaults = dict(history_size=history_size, degree=degree, alpha=alpha)
        super(GPE, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups

        for group in self.param_groups:
            group.setdefault('history_size', history_size)
            group.setdefault('degree', degree)
            group.setdefault('alpha', alpha)
            for p in group['params']:
                self.state[p]['grad_history'] = deque(maxlen=group['history_size'])

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that re-evaluates the model
                                          and returns the loss.
        """
        for group in self.param_groups:
            degree = group['degree']
            alpha = group['alpha']
            history_size = group['history_size']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad_history = state['grad_history']

                grad_history.append(p.grad.data.clone())

                if len(grad_history) == history_size:
                    # 1. Prepare data for polynomial fitting
                    y = torch.stack(list(grad_history))
                    y_flat = y.view(history_size, -1)

                    x = torch.arange(history_size, device=y.device, dtype=y.dtype)

                    # 2. Fit polynomial coefficients using least squares
                    V = torch.vander(x, N=degree + 1)
                    try:
                        coeffs = torch.linalg.lstsq(V, y_flat).solution
                    except torch.linalg.LinAlgError:
                        continue

                    # 3. Extrapolate to the next time step
                    x_next = torch.tensor(history_size, device=y.device, dtype=y.dtype)
                    x_next_powers = x_next.pow(torch.arange(degree, -1, -1, device=y.device, dtype=y.dtype))

                    extrapolated_flat = x_next_powers @ coeffs
                    extrapolated_grad = extrapolated_flat.view_as(p.grad.data)

                    # 4. Apply the update as a weighted average to the gradient
                    p.grad.data.add_(extrapolated_grad - p.grad.data, alpha=alpha)

        return self.base_optimizer.step(closure)
