import torch
from torch.optim.optimizer import Optimizer

class ENGD(Optimizer):
    def __init__(self, params, lr=1e-3, power_iterations=10, tolerance=1e-6):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, power_iterations=power_iterations, tolerance=tolerance)
        super(ENGD, self).__init__(params, defaults)

    def _get_params_and_grads(self):
        """Flattens parameters and their gradients for HVP."""
        params_with_grad = []
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad.view(-1))
        flat_grads = torch.cat(grads)
        return params_with_grad, flat_grads

    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. This is required for HVP calculation.
        """
        # We need to re-evaluate the loss to build the graph for HVP
        self.zero_grad()
        loss = closure()
        loss.backward(create_graph=True)

        params_with_grad, flat_grads = self._get_params_and_grads()

        # Power iteration to find the top eigenvalue of the Hessian
        v = torch.randn_like(flat_grads)
        v = v / torch.norm(v)

        group = self.param_groups[0]
        power_iterations = group['power_iterations']
        tolerance = group['tolerance']
        lr = group['lr']

        eigenvalue = torch.tensor(0.0, device=flat_grads.device)
        for _ in range(power_iterations):
            v.requires_grad_(False) # Make sure v is a leaf node

            grad_v_product = torch.dot(flat_grads, v)

            # The Hessian-vector product
            Hv = torch.autograd.grad(grad_v_product, params_with_grad, retain_graph=True)
            Hv_flat = torch.cat([h.contiguous().view(-1) for h in Hv])

            eigenvalue_new = torch.dot(Hv_flat, v)

            # Check for convergence
            if torch.abs(eigenvalue_new - eigenvalue) < tolerance:
                eigenvalue = eigenvalue_new
                break

            eigenvalue = eigenvalue_new
            v = Hv_flat / (torch.norm(Hv_flat) + 1e-8)

        # Fallback for zero eigenvalue
        if torch.abs(eigenvalue) < 1e-6:
            eigenvalue = torch.tensor(1.0, device=eigenvalue.device)

        # Update parameters
        with torch.no_grad():
            step_size = lr / eigenvalue.abs()
            for p in params_with_grad:
                p.data.add_(p.grad, alpha=-step_size)

        return loss
