import torch
from torch.optim import Optimizer
import math

class FIWDAdamW(Optimizer):
    """
    Fisher-Inverse Weight Decay (FIWD) AdamW.
    Scales weight decay by (1 + v_hat / tau) ** -gamma,
    where v_hat is the bias-corrected empirical Fisher (moving average of squared gradients).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, gamma=1.0, tau=1e-4):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if not 0.0 < tau:
            raise ValueError(f"Invalid tau value: {tau}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, gamma=gamma, tau=tau)
        super(FIWDAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('FIWDAdamW does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                # Update moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Compute Fisher scaling for weight decay
                v_hat = exp_avg_sq / bias_correction2
                scaling = (1 + v_hat / group['tau']).pow(-group['gamma'])

                # Apply weight decay
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'] * scaling)

                # Adam update
                denom = (v_hat.sqrt()).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
