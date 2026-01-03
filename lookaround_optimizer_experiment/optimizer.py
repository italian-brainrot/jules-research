import torch
from torch.optim.optimizer import Optimizer

class Lookaround(Optimizer):
    def __init__(self, base_optimizer, lookaround_alpha=0.5):
        if not 0.0 <= lookaround_alpha <= 1.0:
            raise ValueError(f"Invalid lookaround_alpha: {lookaround_alpha}")

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.lookaround_alpha = lookaround_alpha
        self.state = self.base_optimizer.state
        self.defaults = self.base_optimizer.defaults

    def step(self, closure):
        # 1. Compute original gradients (with grad enabled)
        self.base_optimizer.zero_grad()
        loss = closure()

        original_params_and_grads = []
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        original_params_and_grads.append((p.clone().detach(), p.grad.clone().detach()))

            # 2. Perform a lookaround step
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.data.add_(p.grad, alpha=-group['lr'])

        # 3. Compute gradients at the lookaround position (with grad enabled)
        self.base_optimizer.zero_grad()
        closure()

        # 4. Restore original parameters and combine gradients
        with torch.no_grad():
            i = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        original_param, original_grad = original_params_and_grads[i]
                        p.data.copy_(original_param)
                        p.grad.data.mul_(1 - self.lookaround_alpha).add_(original_grad, alpha=self.lookaround_alpha)
                        i += 1

        # 5. Take a step with the base optimizer using the interpolated gradients
        self.base_optimizer.step()
        return loss

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def __getattr__(self, name):
        return getattr(self.base_optimizer, name)
