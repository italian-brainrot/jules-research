import torch

def soft_thresholding(x, alpha):
    """Soft-thresholding operator for L1 regularization."""
    return torch.sign(x) * torch.relu(torch.abs(x) - alpha)

class ProximalOptimizer:
    """
    A wrapper for an optimizer that applies a proximal operator after the update step.
    This uses composition to avoid inheritance issues with torch.optim.Optimizer.
    """
    def __init__(self, params, base_optimizer, prox_fn, **kwargs):
        """
        Args:
            params (iterable): An iterable of parameters to optimize or dicts defining parameter groups.
            base_optimizer (torch.optim.Optimizer): The base optimizer class (e.g., torch.optim.Adam).
            prox_fn (callable): A function that takes a tensor of parameters and a strength parameter
                                and returns the result of applying the proximal operator.
            **kwargs: Arguments to be passed to the base optimizer.
        """
        self.base_optimizer = base_optimizer(params, **kwargs)
        self.prox_fn = prox_fn
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        loss = self.base_optimizer.step(closure)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # The proximal operator is applied to the parameters.
                # The strength of the operator is controlled by the learning rate.
                p.data = self.prox_fn(p.data, alpha=group['lr'])

        return loss

    # --- Delegate methods to the base optimizer ---
    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group):
        self.base_optimizer.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.base_optimizer.state_dict()
