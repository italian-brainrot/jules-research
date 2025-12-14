import torch
from torch.optim.optimizer import Optimizer

class GradientAgreementOptimizer(Optimizer):
    def __init__(self, params, base_optimizer, **kwargs):
        if not isinstance(base_optimizer, Optimizer):
            raise TypeError(f"base_optimizer must be an instance of torch.optim.Optimizer, but got {type(base_optimizer)}")

        self.base_optimizer = base_optimizer
        # The parameters are already registered in the base_optimizer,
        # so we just need to point to its param_groups.
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = {} # No defaults needed for the wrapper itself

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
                if grad.is_sparse:
                    raise RuntimeError('GradientAgreementOptimizer does not support sparse gradients.')

                # For a 1D tensor, std is 0. Avoid division by 1.
                if grad.numel() > 1:
                    std_dev = grad.std()
                    agreement_score = 1.0 / (1.0 + std_dev)
                    p.grad.mul_(agreement_score)

        return self.base_optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
