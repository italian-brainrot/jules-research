import torch
from torch.optim import Optimizer

class OrthoAdam(Optimizer):
    """
    OrthoAdam optimizer.
    A wrapper around a base optimizer (e.g., Adam) that adds a penalty to encourage
    weight matrices to be orthogonal.
    """
    def __init__(self, params, base_optimizer=torch.optim.Adam, ortho_strength=1e-4, **kwargs):
        if ortho_strength < 0.0:
            raise ValueError("Invalid ortho_strength: {}".format(ortho_strength))

        defaults = dict(ortho_strength=ortho_strength)
        super(OrthoAdam, self).__init__(params, defaults)

        # We need to create the base optimizer with the parameters passed to this optimizer.
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

        # Sync the param_groups between this wrapper and the base optimizer.
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that re-evaluates the model
                and returns the loss.
        """
        # First, compute the loss and update the parameters using the base optimizer.
        loss = self.base_optimizer.step(closure)

        # After the base optimizer step, apply the orthogonality-inducing update.
        for group in self.param_groups:
            ortho_strength = group['ortho_strength']
            if ortho_strength == 0:
                continue

            for p in group['params']:
                if p.dim() == 2:  # Only apply to 2D weight matrices
                    # The gradient of the orthogonality penalty ||W^T W - I||_F^2
                    # is proportional to (W W^T W - W).
                    # We take a step in the negative gradient direction.
                    grad_ortho = p @ p.T @ p - p
                    p.add_(grad_ortho, alpha=-ortho_strength)

        return loss

    def zero_grad(self, set_to_none: bool = False):
        """
        Clears the gradients of all optimized `torch.Tensor`s.
        Arguments:
            set_to_none (bool): instead of setting to zero, set the grads to None.
        """
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
