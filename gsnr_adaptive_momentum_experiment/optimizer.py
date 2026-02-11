import torch
from torch.optim import Optimizer
import math

class GAM(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, gamma=1.0):
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

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, gamma=gamma)
        super(GAM, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, gsnrs, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            gamma = group['gamma']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('GAM does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['m_bias_corr'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                m_bias_corr = state['m_bias_corr']
                state['step'] += 1

                # GSNR for this parameter
                gsnr = gsnrs[p]

                # Adaptive beta1
                # beta1_eff = 1 - (1 - beta1) * gsnr^gamma
                beta1_eff = 1.0 - (1.0 - beta1) * gsnr.pow(gamma)
                one_minus_beta1_eff = 1.0 - beta1_eff

                # Update momentum and its bias correction
                exp_avg.mul_(beta1_eff).addcmul_(grad, one_minus_beta1_eff)
                m_bias_corr.mul_(beta1_eff).add_(one_minus_beta1_eff)

                # Update second moment
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias corrections
                m_hat = exp_avg / (m_bias_corr + 1e-12)
                v_hat = exp_avg_sq / (1.0 - beta2 ** state['step'])

                # Apply weight decay (AdamW style)
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                # Step
                denom = v_hat.sqrt().add_(eps)
                p.addcdiv_(m_hat, denom, value=-lr)

        return loss
