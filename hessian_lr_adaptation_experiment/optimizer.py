import torch

class HessianLRAdaptation:
    """
    A wrapper around a base optimizer that adapts the learning rate for each
    parameter based on an approximation of the Hessian's diagonal.
    This is not a torch.optim.Optimizer subclass, but a wrapper that
    provides the same interface.
    """
    def __init__(self, base_optimizer, beta=0.9, eps=1e-8):
        if not isinstance(base_optimizer, torch.optim.Optimizer):
            raise TypeError(f"{type(base_optimizer).__name__} is not an Optimizer")

        self.base_optimizer = base_optimizer
        self.beta = beta
        self.eps = eps

        # This wrapper maintains its own state, separate from the base optimizer's state.
        self.state = {}

        # The parameters we operate on are those managed by the base optimizer.
        self.param_groups = self.base_optimizer.param_groups

    def zero_grad(self, set_to_none: bool = False):
        """Delegates the zero_grad call to the base optimizer."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """Delegates the state_dict call to the base optimizer."""
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Delegates the load_state_dict call to the base optimizer."""
        self.base_optimizer.load_state_dict(state_dict)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step. The gradients are first scaled
        based on the Hessian approximation, and then the base optimizer's
        step function is called.
        """
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
                    raise RuntimeError('HessianLRAdaptation does not support sparse gradients')

                # Use the wrapper's state for this parameter.
                if p not in self.state:
                    self.state[p] = {}

                param_state = self.state[p]

                # Initialize the wrapper's state for the parameter.
                if 'step' not in param_state:
                    param_state['step'] = 0
                    param_state['exp_avg_sq_hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    param_state['prev_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg_sq_hessian = param_state['exp_avg_sq_hessian']
                prev_grad = param_state['prev_grad']

                param_state['step'] += 1

                # On the first step, we only record the gradient.
                if param_state['step'] > 1:
                    grad_diff = grad - prev_grad
                    exp_avg_sq_hessian.mul_(self.beta).addcmul_(grad_diff, grad_diff, value=1 - self.beta)

                param_state['prev_grad'] = grad.clone()

                # Starting from the second step, we scale the gradient.
                if param_state['step'] > 1:
                    hessian_diag_approx = exp_avg_sq_hessian.sqrt().add_(self.eps)
                    p.grad.div_(hessian_diag_approx)

        # The base optimizer performs its step with the modified gradients.
        # Its own state (e.g., for momentum) is managed independently.
        self.base_optimizer.step()

        return loss
