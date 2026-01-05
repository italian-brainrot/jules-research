import torch
from torch.optim.optimizer import Optimizer

class RelativeGradientPredictiveCoding(Optimizer):
    def __init__(self, params, base_optimizer, history_size=5):
        if not history_size >= 2:
            raise ValueError(f"Invalid history_size: {history_size}, must be at least 2 for linear fit.")

        self.base_optimizer = base_optimizer
        self.history_size = history_size

        # Initialize the wrapper optimizer to create its own state dictionary.
        # We pass an empty defaults dict because we will rely on the base_optimizer's defaults.
        defaults = {}
        super().__init__(params, defaults)

        # Ensure the wrapper and base optimizer share the same param_groups.
        # This makes sure that changes to learning rate, etc., are reflected in both.
        self.param_groups = self.base_optimizer.param_groups

    def __setstate__(self, state):
        super(RelativeGradientPredictiveCoding, self).__setstate__(state)

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

                grad = p.grad
                # Use the wrapper's own state, not the base_optimizer's.
                state = self.state[p]

                # Lazy initialization of history buffer
                if 'rgpc_history' not in state:
                    state['rgpc_history'] = []

                history = state['rgpc_history']

                # Store current (param_value, grad) pair
                history.append((p.clone(), grad.clone()))
                if len(history) > self.history_size:
                    history.pop(0)

                if len(history) == self.history_size:
                    param_vals = torch.stack([h[0] for h in history])
                    grad_vals = torch.stack([h[1] for h in history])

                    p_flat = param_vals.view(self.history_size, -1).t()
                    g_flat = grad_vals.view(self.history_size, -1).t()

                    A = torch.stack([p_flat, torch.ones_like(p_flat)], dim=-1)

                    try:
                        # Using a regularized least squares for stability
                        A_t_A = A.transpose(-2, -1) @ A
                        A_t_A.diagonal(dim1=-2, dim2=-1).add_(1e-6)
                        A_t_g = A.transpose(-2, -1) @ g_flat.unsqueeze(-1)
                        solution = torch.linalg.solve(A_t_A, A_t_g)

                        a = solution[:, 0, 0].view_as(p)
                        b = solution[:, 1, 0].view_as(p)

                        predicted_grad = a * p + b

                        p.grad = (grad + predicted_grad) / 2.0

                    except torch.linalg.LinAlgError:
                        # If solver fails, proceed with the original gradient
                        pass

        # Now, call the base_optimizer's step. It will use its own state
        # which has been untouched by the wrapper.
        self.base_optimizer.step()
        return loss

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
