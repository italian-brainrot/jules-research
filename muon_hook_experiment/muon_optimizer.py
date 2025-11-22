import torch
from torch.optim.optimizer import Optimizer
from .utils import orthogonalize

class MuonOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.9, weight_decay=0, nesterov=True, ns_steps=5):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay, nesterov=nesterov, ns_steps=ns_steps)
        super(MuonOptimizer, self).__init__(params, defaults)

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
                    raise RuntimeError('MuonOptimizer does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                beta = group['beta']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Update momentum
                exp_avg.mul_(beta).add_(grad)

                if group['nesterov']:
                    momentum_update = exp_avg.mul(beta).add(grad)
                else:
                    momentum_update = exp_avg

                if p.grad.ndim < 2:
                    # Standard momentum update for 1D parameters
                    p.add_(momentum_update, alpha=-group['lr'])
                else:
                    # Orthogonalize momentum for 2D+ parameters
                    ortho_momentum = orthogonalize(momentum_update, steps=group['ns_steps'])
                    p.add_(ortho_momentum, alpha=-group['lr'])

        return loss
