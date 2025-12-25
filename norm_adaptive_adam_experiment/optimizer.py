import torch
from torch.optim.optimizer import Optimizer

class NormAdaptiveAdam(Optimizer):
    def __init__(self, params, base_optimizer, clip_factor=1.0):
        if not 0.0 <= clip_factor:
            raise ValueError(f"Invalid clip_factor: {clip_factor}")

        self.base_optimizer = base_optimizer
        self.clip_factor = clip_factor
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = self.base_optimizer.defaults

    def __setstate__(self, state):
        super().__setstate__(state)

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

                weight_norm = torch.norm(p.data).item()
                clip_value = self.clip_factor * weight_norm

                if clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(p, max_norm=clip_value)

        self.base_optimizer.step()
        return loss
