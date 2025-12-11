import torch
from torch.optim.optimizer import Optimizer

class ConjugateGradient(Optimizer):
    """
    Implements the Conjugate Gradient algorithm.
    """
    def __init__(self, params, lr=1.0):
        defaults = dict(lr=lr)
        super(ConjugateGradient, self).__init__(params, defaults)

    def _get_flat_grad(self):
        views = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    view = p.data.new(p.data.numel()).zero_()
                elif p.grad.data.is_sparse:
                    view = p.grad.data.to_dense().view(-1)
                else:
                    view = p.grad.data.view(-1)
                views.append(view)
        return torch.cat(views, 0)

    def _set_flat_params(self, flat_params):
        offset = 0
        for group in self.param_groups:
            for p in group['params']:
                numel = p.data.numel()
                # view as to avoid deprecated pointwise semantics
                p.data.copy_(flat_params[offset:offset + numel].view_as(p.data))
                offset += numel
        assert offset == len(flat_params)

    def _get_flat_params(self):
        views = []
        for group in self.param_groups:
            for p in group['params']:
                view = p.data.view(-1)
                views.append(view)
        return torch.cat(views, 0)

    def step(self, closure):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates the model and returns the loss.
        """
        loss = closure()
        g = self._get_flat_grad()

        # Define the Hessian-vector product function
        def hvp(v):
            grad_params = torch.autograd.grad(loss, self.param_groups[0]['params'], create_graph=True, retain_graph=True)
            flat_grad = torch.cat([grad.reshape(-1) for grad in grad_params])
            grad_v_prod = torch.dot(flat_grad, v)
            hvp_flat = torch.autograd.grad(grad_v_prod, self.param_groups[0]['params'], retain_graph=True)
            return torch.cat([h.reshape(-1) for h in hvp_flat if h is not None])

        # Solve for the search direction using CG
        s = self._conjugate_gradient(hvp, -g)

        # Line search (for now, use a fixed learning rate)
        current_params = self._get_flat_params()
        new_params = current_params + self.defaults['lr'] * s
        self._set_flat_params(new_params)

        return loss

    def _conjugate_gradient(self, hvp_fn, b, cg_iters=10, residual_tol=1e-10):
        """
        Solves for the search direction using the Conjugate Gradient method.
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs_old = torch.dot(r, r)

        for i in range(cg_iters):
            Ap = hvp_fn(p)
            alpha = rs_old / torch.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.dot(r, r)
            if torch.sqrt(rs_new) < residual_tol:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

class EigenConjugateGradient(Optimizer):
    """
    Implements the Eigen-Conjugate Gradient algorithm.
    """
    def __init__(self, params, lr=1.0, num_eigenthings=5, eigen_update_freq=10):
        defaults = dict(lr=lr, num_eigenthings=num_eigenthings, eigen_update_freq=eigen_update_freq)
        super(EigenConjugateGradient, self).__init__(params, defaults)
        self.state['step'] = 0

    def _get_flat_grad(self):
        views = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    view = p.data.new(p.data.numel()).zero_()
                elif p.grad.data.is_sparse:
                    view = p.grad.data.to_dense().view(-1)
                else:
                    view = p.grad.data.view(-1)
                views.append(view)
        return torch.cat(views, 0)

    def _set_flat_params(self, flat_params):
        offset = 0
        for group in self.param_groups:
            for p in group['params']:
                numel = p.data.numel()
                # view as to avoid deprecated pointwise semantics
                p.data.copy_(flat_params[offset:offset + numel].view_as(p.data))
                offset += numel
        assert offset == len(flat_params)

    def _get_flat_params(self):
        views = []
        for group in self.param_groups:
            for p in group['params']:
                view = p.data.view(-1)
                views.append(view)
        return torch.cat(views, 0)

    def step(self, closure, model, dataloader, loss_fn):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates the model and returns the loss.
        """
        loss = closure()
        g = self._get_flat_grad()

        # Define the Hessian-vector product function
        def hvp(v):
            grad_params = torch.autograd.grad(loss, self.param_groups[0]['params'], create_graph=True, retain_graph=True)
            flat_grad = torch.cat([grad.reshape(-1) for grad in grad_params])
            grad_v_prod = torch.dot(flat_grad, v)
            hvp_flat = torch.autograd.grad(grad_v_prod, self.param_groups[0]['params'], retain_graph=True)
            return torch.cat([h.reshape(-1) for h in hvp_flat if h is not None])

        # Solve for the search direction using CG
        s = self._conjugate_gradient(hvp, -g)

        # Periodically update eigenvectors and project search direction
        if self.state['step'] % self.defaults['eigen_update_freq'] == 0:
            from .hessian_eigenthings import compute_hessian_eigenthings
            eigenvals, eigenvecs = compute_hessian_eigenthings(model, dataloader, loss_fn, self.defaults['num_eigenthings'])

            # Project s onto the subspace of the top eigenvectors
            proj_s = torch.zeros_like(s)
            for i in range(len(eigenvecs)):
                eigenvec = torch.from_numpy(eigenvecs[i]).to(s.device)
                proj_s += torch.dot(s, eigenvec) * eigenvec
            s = proj_s

        # Line search (for now, use a fixed learning rate)
        current_params = self._get_flat_params()
        new_params = current_params + self.defaults['lr'] * s
        self._set_flat_params(new_params)

        self.state['step'] += 1
        return loss

    def _conjugate_gradient(self, hvp_fn, b, cg_iters=10, residual_tol=1e-10):
        """
        Solves for the search direction using the Conjugate Gradient method.
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs_old = torch.dot(r, r)

        for i in range(cg_iters):
            Ap = hvp_fn(p)
            alpha = rs_old / torch.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.dot(r, r)
            if torch.sqrt(rs_new) < residual_tol:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x
