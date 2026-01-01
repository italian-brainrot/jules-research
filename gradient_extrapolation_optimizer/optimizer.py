import torch
from torch.optim.optimizer import Optimizer
import collections

class GPE(Optimizer):
    def __init__(self, base_optimizer, history_size=10, degree=2):
        if not 0 <= degree <= history_size - 1:
            raise ValueError("Degree must be between 0 and history_size - 1.")

        self.base_optimizer = base_optimizer
        self.history_size = history_size
        self.degree = degree
        self.param_groups = self.base_optimizer.param_groups
        self.state = collections.defaultdict(lambda: collections.deque(maxlen=self.history_size))
        self.defaults = base_optimizer.defaults

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Store a clone of the gradient
                state.append(grad.clone())

                if len(state) == self.history_size:
                    # History is full, perform extrapolation
                    history = torch.stack(list(state))

                    # Timestamps for fitting
                    t = torch.arange(self.history_size, device=history.device, dtype=history.dtype)

                    # Fit polynomial and extrapolate
                    # We need to do this for each element of the gradient tensor
                    # Reshape to (history_size, num_elements)
                    history_flat = history.view(self.history_size, -1)

                    coeffs = torch.zeros(history_flat.shape[1], self.degree + 1, device=history.device, dtype=history.dtype)

                    # Use torch.linalg.lstsq for efficient batched polyfit
                    # Vandermonde matrix for timestamps
                    vander_t = torch.vander(t, N=self.degree + 1)

                    # Solve for coefficients for all gradient elements at once
                    # coeffs shape: (degree+1, num_elements)
                    coeffs = torch.linalg.lstsq(vander_t, history_flat).solution

                    # Extrapolate to the next time step (t = history_size)
                    next_t_powers = torch.pow(torch.tensor(self.history_size, device=history.device, dtype=history.dtype), torch.arange(self.degree, -1, -1, device=history.device, dtype=history.dtype))

                    # Calculate extrapolated gradient
                    # (num_elements, degree+1) @ (degree+1,) -> (num_elements,)
                    extrapolated_grad_flat = torch.matmul(coeffs.T, next_t_powers)

                    # Reshape back to original gradient shape
                    extrapolated_grad = extrapolated_grad_flat.view_as(grad)

                    # Replace the current gradient with the extrapolated one
                    p.grad.data = extrapolated_grad

        self.base_optimizer.step()
        return loss

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group):
        self.base_optimizer.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.base_optimizer.state_dict()
