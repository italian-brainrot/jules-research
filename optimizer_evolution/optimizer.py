import torch
from torch.optim.optimizer import Optimizer
from gp import func_map

class EvolvedOptimizer(Optimizer):
    def __init__(self, params, update_rule_str, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        super(EvolvedOptimizer, self).__init__(params, defaults)
        self.update_rule_str = update_rule_str
        self.func_map = func_map

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('EvolvedOptimizer does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['v'] = torch.zeros_like(p.data)

                m, v = state['m'], state['v']
                beta1, beta2 = group['beta1'], group['beta2']

                state['step'] += 1

                # Update biased first moment estimate
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                scope = {
                    **self.func_map,
                    'g': grad,
                    'm': m,
                    'v': v,
                    'lr': group['lr'],
                    'beta1': beta1,
                    'beta2': beta2,
                    'epsilon': group['epsilon'],
                    'one': 1.0,
                    'zero': 0.0
                }

                try:
                    scaling_factor = eval(self.update_rule_str, scope)
                    # Adam update rule
                    denom = v.sqrt().add(group['epsilon'])
                    update = (m / denom).mul_(-group['lr'] * scaling_factor)
                    p.data.add_(update)
                except Exception as e:
                    print(f"Error evaluating update rule: {e}")

        return loss
