import torch
from torch.optim.optimizer import Optimizer

class DynamicBlendedOptimizer(Optimizer):
    """
    Implements a dynamic blend of Adam and SGD with momentum.

    The blending factor `alpha` is determined by a gating network, which takes
    the flattened gradients of the model parameters as input. The final update is
    a convex combination of the updates from Adam and SGD:
    `update = alpha * adam_update + (1 - alpha) * sgd_update`

    The internal states for both Adam and SGD are maintained and updated at each step.
    """
    def __init__(self, params, gating_net, lr_adam=1e-3, lr_sgd=1e-2, momentum=0.9, betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 <= lr_adam:
            raise ValueError(f"Invalid Adam learning rate: {lr_adam}")
        if not 0.0 <= lr_sgd:
            raise ValueError(f"Invalid SGD learning rate: {lr_sgd}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")

        self.gating_net = gating_net
        defaults = dict(lr_adam=lr_adam, lr_sgd=lr_sgd, momentum=momentum, betas=betas, eps=eps)
        super(DynamicBlendedOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DynamicBlendedOptimizer, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Create a flat tensor of all gradients for the gating network
        all_grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    all_grads.append(p.grad.view(-1))

        if not all_grads:
            return loss

        flat_grads = torch.cat(all_grads)

        # Get the blending factor from the gating network.
        # We detach the grads to ensure the gating network's computation
        # doesn't become part of the main model's computation graph.
        # The meta-learning step will handle the graph for the gate itself.
        alpha = self.gating_net(flat_grads.detach())

        for group in self.param_groups:
            params_with_grad = []
            grads = []

            # Adam state
            exp_avgs = []
            exp_avg_sqs = []

            # SGD state
            momentum_buffer_list = []

            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)

                    state = self.state[p]
                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Adam state
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # SGD state
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    momentum_buffer_list.append(state['momentum_buffer'])

                    state['step'] += 1
                    state_steps.append(state['step'])

            # Perform blended update
            self._blended_step(params_with_grad,
                               grads,
                               exp_avgs,
                               exp_avg_sqs,
                               momentum_buffer_list,
                               state_steps,
                               alpha=alpha,
                               group=group)

        return loss

    def _blended_step(self, params, grads, exp_avgs, exp_avg_sqs, momentum_buffer_list, state_steps, alpha, group):
        beta1, beta2 = group['betas']
        lr_adam = group['lr_adam']
        lr_sgd = group['lr_sgd']
        momentum = group['momentum']
        eps = group['eps']

        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]
            momentum_buffer = momentum_buffer_list[i]

            # --- Calculate and apply Adam Update ---
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)
            adam_update = (exp_avg / bias_correction1) / denom

            # --- Calculate and apply SGD Update ---
            momentum_buffer.mul_(momentum).add_(grad)
            sgd_update = momentum_buffer

            # --- Blend and Apply Update ---
            # Note: The learning rates are applied here
            blended_update = alpha * (-lr_adam * adam_update) + (1.0 - alpha) * (-lr_sgd * sgd_update)
            param.add_(blended_update)
