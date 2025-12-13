import torch

class CyclicalGradientInversion(torch.optim.Optimizer):
    def __init__(self, base_optimizer, inversion_frequency):
        if inversion_frequency <= 1:
            raise ValueError("inversion_frequency must be greater than 1")

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
        self.defaults = self.base_optimizer.defaults
        self.inversion_frequency = inversion_frequency
        self.step_count = 0

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_count += 1

        if self.step_count % self.inversion_frequency == 0:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.mul_(-1)

        self.base_optimizer.step()

        # Invert back the gradients so that they are not stored inverted
        if self.step_count % self.inversion_frequency == 0:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.mul_(-1)

        return loss
