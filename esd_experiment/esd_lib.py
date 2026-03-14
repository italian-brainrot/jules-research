import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_correlations(features):
    """
    features: (batch_size, num_features)
    Returns: (num_features, num_features) correlation matrix
    """
    if features.shape[0] <= 1:
        return torch.zeros((features.shape[1], features.shape[1]), device=features.device)

    # Center the features
    centered_features = features - features.mean(dim=0, keepdim=True)

    # Compute covariance
    # (num_features, batch_size) @ (batch_size, num_features) -> (num_features, num_features)
    cov = (centered_features.t() @ centered_features) / (features.shape[0] - 1)

    # Compute standard deviations
    std = torch.sqrt(torch.diag(cov) + 1e-8)

    # Correlation matrix
    corr = cov / (std.unsqueeze(1) @ std.unsqueeze(0) + 1e-8)
    return corr

def decorrelation_loss(features):
    """
    Penalize the sum of squared off-diagonal elements of the correlation matrix.
    """
    corr = compute_correlations(features)
    n = corr.shape[0]
    if n <= 1:
        return torch.tensor(0.0, device=features.device)
    # Sum of squares of off-diagonal elements
    eye = torch.eye(n, device=corr.device)
    off_diag_sq = (corr - eye).pow(2)
    return off_diag_sq.sum() / (n * (n - 1) + 1e-8)

class DecorrManager:
    def __init__(self, model, mode='ESD'):
        self.model = model
        self.mode = mode # 'ESD' or 'Decorr'
        self.captured_tensors = []
        self.hooks = []
        self._setup_hooks()

    def _setup_hooks(self):
        def hook(module, input, output):
            # We capture the output of Linear layers (pre-activations)
            self.captured_tensors.append(output)

        for name, module in self.model.named_modules():
            # Only capture hidden layers, skip the last one (output layer)
            # This is a bit heuristic, but let's say we only care about hidden representations
            if isinstance(module, nn.Linear):
                # We'll identify output layer by looking at output_dim=10 if possible,
                # or just capture all and then decide.
                # For our 3-layer MLP, let's just capture all and we can skip the last one if we want.
                self.hooks.append(module.register_forward_hook(hook))

    def clear(self):
        self.captured_tensors = []

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def compute_loss(self, loss=None):
        if not self.captured_tensors:
            return torch.tensor(0.0, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # In our MLP, we capture fc1_out, fc2_out, fc3_out.
        # We probably want to skip the last one.
        tensors_to_use = self.captured_tensors[:-1]

        total_decorr_loss = 0.0

        if self.mode == 'ESD':
            if loss is None:
                raise ValueError("Loss must be provided for ESD mode")
            # Gradients of loss w.r.t captured tensors
            # loss is reduced (mean), but captured tensors are (B, N).
            # autograd.grad will give (B, N) gradients.
            grads = torch.autograd.grad(loss, tensors_to_use, retain_graph=True, create_graph=True)
            for g in grads:
                total_decorr_loss += decorrelation_loss(g)
        else: # Standard activation decorrelation
            for t in tensors_to_use:
                total_decorr_loss += decorrelation_loss(t)

        if not tensors_to_use:
            return torch.tensor(0.0, device=loss.device if loss is not None else self.captured_tensors[0].device)

        return total_decorr_loss / len(tensors_to_use)
