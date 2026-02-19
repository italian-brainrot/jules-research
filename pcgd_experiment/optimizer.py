import torch
from torch.func import vmap, grad, functional_call
import torch.nn.functional as F

def get_consensus_gradient(model, x, y, normalize_grads=True):
    """
    Computes the PCGD consensus gradient for a batch.
    """
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def loss_fn(p, b, x_sample, y_sample):
        state_dict = {**p, **b}
        logits = functional_call(model, state_dict, (x_sample.unsqueeze(0),))
        return F.cross_entropy(logits, y_sample.unsqueeze(0))

    grad_fn = grad(loss_fn)
    per_sample_grads_dict = vmap(grad_fn, in_dims=(None, None, 0, 0))(params, buffers, x, y)

    batch_size = x.shape[0]

    # Compute Gram matrix incrementally to save memory and potentially time
    K = torch.zeros((batch_size, batch_size), device=x.device, dtype=x.dtype)

    if normalize_grads:
        # We need the total norm squared for each sample first
        total_norm_sq = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        for g in per_sample_grads_dict.values():
            total_norm_sq += g.reshape(batch_size, -1).pow(2).sum(dim=1)
        total_norm = torch.sqrt(total_norm_sq + 1e-8)

        for g in per_sample_grads_dict.values():
            g_flat = g.reshape(batch_size, -1)
            # Normalize g_flat per sample
            g_normalized = g_flat / total_norm.unsqueeze(1)
            K += torch.matmul(g_normalized, g_normalized.t())
    else:
        for g in per_sample_grads_dict.values():
            g_flat = g.reshape(batch_size, -1)
            K += torch.matmul(g_flat, g_flat.t())

    # Non-negative top eigenvector via projected power method
    v = torch.ones(batch_size, device=x.device, dtype=x.dtype)
    v = v / torch.norm(v)

    for _ in range(10):
        v = torch.matmul(K, v)
        v = torch.clamp(v, min=0.0)
        v_norm = torch.norm(v)
        if v_norm > 1e-8:
            v = v / v_norm
        else:
            v = torch.ones(batch_size, device=K.device, dtype=K.dtype) / (batch_size**0.5)
            break

    # Consensus direction
    # Use weighted average of ORIGINAL gradients
    # Weight by v_i
    v_sum = v.sum()
    if v_sum > 1e-8:
        weights = v / v_sum * batch_size # scale so that if all v_i are equal, weights are 1
    else:
        weights = torch.ones_like(v)

    consensus_grads = {}
    total_consensus_norm_sq = 0.0
    total_mean_norm_sq = 0.0

    for name, g in per_sample_grads_dict.items():
        # Mean gradient for this parameter
        mean_g = g.mean(dim=0)
        total_mean_norm_sq += mean_g.pow(2).sum()

        # Consensus gradient for this parameter
        w_expanded = weights.view(-1, *([1] * (g.dim() - 1)))
        cons_g = (w_expanded * g).mean(dim=0)
        consensus_grads[name] = cons_g
        total_consensus_norm_sq += cons_g.pow(2).sum()

    # Re-scale consensus gradients to have the same total norm as the mean gradient
    # This ensures we only test the effect of the direction
    mean_norm = torch.sqrt(total_mean_norm_sq + 1e-8)
    cons_norm = torch.sqrt(total_consensus_norm_sq + 1e-8)
    scale = mean_norm / cons_norm

    for name in consensus_grads:
        consensus_grads[name] *= scale

    return consensus_grads

class PCGDOptimizer:
    def __init__(self, model, base_optimizer, normalize_grads=True):
        self.model = model
        self.base_optimizer = base_optimizer
        self.normalize_grads = normalize_grads
        self.param_groups = base_optimizer.param_groups
        self.state = base_optimizer.state

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, x, y):
        consensus_grads = get_consensus_gradient(self.model, x, y, self.normalize_grads)
        for name, p in self.model.named_parameters():
            if name in consensus_grads:
                p.grad = consensus_grads[name]
        self.base_optimizer.step()

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
