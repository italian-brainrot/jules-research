import torch
from torch.optim.optimizer import Optimizer

class MetaCurvatureLR(Optimizer):
    def __init__(self, optimizer, update_freq=10, alpha=1.0, n_hutchinson_samples=1):
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if not 0 < update_freq:
            raise ValueError(f"Invalid update_freq value: {update_freq}")
        if not 0 < n_hutchinson_samples:
            raise ValueError(f"Invalid n_hutchinson_samples value: {n_hutchinson_samples}")

        self.optimizer = optimizer
        self.update_freq = update_freq
        self.alpha = alpha
        self.n_hutchinson_samples = n_hutchinson_samples
        self._step = 0
        self.initial_lrs = [group['lr'] for group in self.optimizer.param_groups]

        super(MetaCurvatureLR, self).__init__(self.optimizer.param_groups, {})

    def step(self, closure):
        self._step += 1

        if self._step % self.update_freq == 0:
            with torch.enable_grad():
                trace_h = self._estimate_hessian_trace(closure)

            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.initial_lrs[i] / (1.0 + self.alpha * trace_h)

        self.optimizer.step()

    def _estimate_hessian_trace(self, closure):
        loss = closure()
        params = [p for group in self.optimizer.param_groups for p in group['params'] if p.requires_grad]
        grads = torch.autograd.grad(loss, params, create_graph=True)

        trace = 0.0
        for _ in range(self.n_hutchinson_samples):
            v = [torch.randn_like(p) for p in params]
            hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
            trace += sum([(v_i * hv_i).sum() for v_i, hv_i in zip(v, hv)])

        return trace / self.n_hutchinson_samples

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    # All other methods should be proxied to the base optimizer
    def __getattr__(self, name):
        return getattr(self.optimizer, name)
