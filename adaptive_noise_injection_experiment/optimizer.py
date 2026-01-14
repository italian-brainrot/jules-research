import torch

class AdaptiveNoiseAdam:
    def __init__(self, base_optimizer, noise_level=0.1):
        if noise_level < 0.0:
            raise ValueError(f"Invalid noise_level: {noise_level}")
        self.base_optimizer = base_optimizer
        self.noise_level = noise_level
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state

    def step(self, closure=None):
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaptiveNoiseAdam does not support sparse gradients')

                grad_norm = torch.norm(grad)
                noise = torch.randn_like(grad) * self.noise_level * grad_norm
                p.grad.data.add_(noise)

        return self.base_optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
