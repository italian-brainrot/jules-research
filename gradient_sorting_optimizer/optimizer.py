import torch
from torch.optim.optimizer import Optimizer

class SortedGradientOptimizer(Optimizer):
    """
    A wrapper optimizer that sorts the gradients of each parameter before applying a base optimizer.
    """
    def __init__(self, base_optimizer):
        if not isinstance(base_optimizer, Optimizer):
            raise ValueError("base_optimizer must be an instance of torch.optim.Optimizer")
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
        self.defaults = base_optimizer.defaults

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
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

                grad = p.grad.data
                original_shape = grad.shape

                # Flatten, sort, and reshape the gradient
                sorted_grad = torch.sort(grad.flatten())[0]
                p.grad.data = sorted_grad.reshape(original_shape)

        self.base_optimizer.step()
        return loss
