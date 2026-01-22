import torch
from torch.optim.optimizer import Optimizer

class SoftWeightProjection(Optimizer):
    def __init__(self, optimizer, projection_strength=0.1):
        if not 0.0 <= projection_strength <= 1.0:
            raise ValueError(f"Invalid projection_strength: {projection_strength}")
        self.optimizer = optimizer
        self.projection_strength = projection_strength
        self.state = {}

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def defaults(self):
        return self.optimizer.defaults

    def zero_grad(self):
        self.optimizer.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state.setdefault(p, {})

                # Store old parameters
                if 'old_param' not in param_state:
                    param_state['old_param'] = torch.zeros_like(p.data)
                param_state['old_param'].copy_(p.data)

        # Let the inner optimizer perform its step
        self.optimizer.step()

        # Apply the soft projection
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                old_p = param_state['old_param']

                # Blend the new weights with the old weights
                # p_new = (1 - alpha) * p_updated + alpha * p_old
                p.data.mul_(1 - self.projection_strength).add_(old_p, alpha=self.projection_strength)

        return loss

if __name__ == '__main__':
    # Example usage:
    model = torch.nn.Linear(10, 1)
    base_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer = SoftWeightProjection(base_optimizer, projection_strength=0.5)

    # Dummy training loop
    for _ in range(5):
        optimizer.zero_grad()
        output = model(torch.randn(5, 10))
        loss = output.mean()
        loss.backward()
        print(f"Weight before step: {model.weight.data.clone()}")
        optimizer.step()
        print(f"Weight after step: {model.weight.data.clone()}")
