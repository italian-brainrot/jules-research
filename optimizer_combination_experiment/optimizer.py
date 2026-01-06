import torch
from torch.optim import Optimizer

class CombinedOptimizer(Optimizer):
    """
    Wraps two optimizers and combines their updates.
    """
    def __init__(self, optimizer1, optimizer2, alpha_schedule):
        """
        Initializes the CombinedOptimizer.

        Args:
            optimizer1 (Optimizer): The first optimizer.
            optimizer2 (Optimizer): The second optimizer.
            alpha_schedule (callable): A function that takes the current step `t`
                                       and returns the mixing coefficient alpha.
                                       The combined update is (1-alpha)*update1 + alpha*update2.
        """
        # We need to ensure that the base Optimizer is initialized correctly.
        # The param_groups are managed by the individual optimizers.
        # We can just use the param_groups from the first optimizer.
        super(CombinedOptimizer, self).__init__(optimizer1.param_groups, {})

        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.alpha_schedule = alpha_schedule
        self.t = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        alpha = self.alpha_schedule(self.t)

        original_params = []
        for group in self.param_groups:
            for p in group['params']:
                original_params.append(p.clone())

        # Calculate update from the first optimizer
        self.optimizer1.step()
        updates1 = []
        param_idx = 0
        for group in self.param_groups:
            for p in group['params']:
                updates1.append(p.clone() - original_params[param_idx])
                p.data.copy_(original_params[param_idx])  # Revert parameters
                param_idx += 1

        # Calculate update from the second optimizer
        self.optimizer2.step()
        updates2 = []
        param_idx = 0
        for group in self.param_groups:
            for p in group['params']:
                updates2.append(p.clone() - original_params[param_idx])
                param_idx += 1

        # Apply the combined update
        param_idx = 0
        for group in self.param_groups:
            for p in group['params']:
                combined_update = (1 - alpha) * updates1[param_idx] + alpha * updates2[param_idx]
                p.data.copy_(original_params[param_idx] + combined_update)
                param_idx += 1

        self.t += 1
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer1.zero_grad(set_to_none=set_to_none)
        self.optimizer2.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        # We need to save the state of both optimizers
        state_dict1 = self.optimizer1.state_dict()
        state_dict2 = self.optimizer2.state_dict()
        return {'optimizer1': state_dict1, 'optimizer2': state_dict2, 't': self.t}

    def load_state_dict(self, state_dict):
        self.optimizer1.load_state_dict(state_dict['optimizer1'])
        self.optimizer2.load_state_dict(state_dict['optimizer2'])
        self.t = state_dict['t']
