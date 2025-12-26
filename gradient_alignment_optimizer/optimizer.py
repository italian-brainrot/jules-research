import torch
from torch.optim.optimizer import Optimizer

from collections import defaultdict

class GradientAlignmentOptimizer(Optimizer):
    def __init__(self, base_optimizer, beta=0.9):
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta: {beta}")
        self.base_optimizer = base_optimizer
        self.beta = beta

        # Expose the base optimizer's attributes
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = self.base_optimizer.defaults

        # Don't alias the state! Create our own for our specific values.
        self.state = defaultdict(dict)

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

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
                state = self.state[p] # Use our own state dict

                # Lazy initialization
                if 'grad_avg' not in state:
                    state['grad_avg'] = torch.zeros_like(p.data)

                grad_avg = state['grad_avg']

                # Update moving average of gradient
                grad_avg.mul_(self.beta).add_(grad, alpha=1 - self.beta)

                # Compute cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    grad.flatten(), grad_avg.flatten(), dim=0, eps=1e-8
                )

                # Clamp to avoid negative values
                cos_sim = torch.clamp(cos_sim, 0.0, 1.0)

                # Modulate the gradient of the parameter directly
                p.grad.data.mul_(cos_sim.item())

        self.base_optimizer.step()

        return loss
