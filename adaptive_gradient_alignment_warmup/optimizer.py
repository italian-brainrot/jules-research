import torch

class AGAWOptimizer:
    def __init__(self, base_optimizer, target_lr, initial_lr=1e-7, gamma=2.0, warmup_steps_nominal=100):
        self.base_optimizer = base_optimizer
        self.target_lr = target_lr
        self.current_lr = initial_lr
        self.gamma = gamma
        self.warmup_steps_nominal = warmup_steps_nominal
        self.prev_grad = None

        # Set initial LR in base optimizer
        for group in self.base_optimizer.param_groups:
            group['lr'] = self.current_lr

    def step(self, closure=None):
        # We need to capture the gradients BEFORE the base_optimizer.step()
        # because some optimizers might modify gradients (though Adam usually doesn't in-place until update).
        # Actually, it's safer to do it here.

        if self.current_lr < self.target_lr:
            # Compute current gradient vector
            grads = []
            for group in self.base_optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        grads.append(p.grad.detach().view(-1))

            if grads:
                current_grad = torch.cat(grads)
                if self.prev_grad is not None and self.prev_grad.shape == current_grad.shape:
                    # Compute cosine similarity
                    dot = torch.dot(current_grad, self.prev_grad)
                    norm1 = torch.norm(current_grad)
                    norm2 = torch.norm(self.prev_grad)

                    if norm1 > 1e-8 and norm2 > 1e-8:
                        cos_sim = dot / (norm1 * norm2)
                        # Adaptive increase
                        # If cos_sim = 1, we reach target_lr in warmup_steps_nominal steps.
                        increase = ((self.target_lr - 1e-7) / self.warmup_steps_nominal) * (torch.clamp(cos_sim, min=0.0) ** self.gamma)
                        self.current_lr = min(self.target_lr, self.current_lr + increase.item())

                        for group in self.base_optimizer.param_groups:
                            group['lr'] = self.current_lr

                self.prev_grad = current_grad.clone()

        return self.base_optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
