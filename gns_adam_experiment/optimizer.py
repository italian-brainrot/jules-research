import torch
from torch.optim import Optimizer

class GNSAdam(Optimizer):
    def __init__(self, params, base_optimizer, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr)
        super(GNSAdam, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        self.param_groups = self.base_optimizer.param_groups
        # Store the initial learning rate for each param group
        for group in self.param_groups:
            group['initial_lr'] = group['lr']

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        total_grad_norm = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        if total_grad_norm > 0:
            for group in self.param_groups:
                # Scale the INITIAL learning rate inversely to the gradient norm
                group['lr'] = group['initial_lr'] / total_grad_norm
        else:
            # If grad norm is zero, reset to initial LR. Adam will handle the zero gradient anyway.
            for group in self.param_groups:
                group['lr'] = group['initial_lr']

        self.base_optimizer.step()
        return loss
