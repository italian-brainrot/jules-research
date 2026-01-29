import torch
from torch.optim.optimizer import Optimizer

class WNGD(Optimizer):
    def __init__(self, params, lr=1e-3, eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(lr=lr, eps=eps)
        super(WNGD, self).__init__(params, defaults)

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
                    raise RuntimeError("WNGD does not support sparse gradients")

                weight_norm = torch.norm(p.data)

                # Normalize the gradient by the weight norm
                normalized_grad = grad / (weight_norm + group['eps'])

                # Perform the update
                p.data.add_(normalized_grad, alpha=-group['lr'])

        return loss
