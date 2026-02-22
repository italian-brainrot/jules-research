import torch
from torch.func import vmap, grad, functional_call
import torch.nn.functional as F

def compute_sample_loss(params, buffers, model, x, y):
    state_dict = {**params, **buffers}
    logits = functional_call(model, state_dict, (x.unsqueeze(0),))
    return F.cross_entropy(logits, y.unsqueeze(0))

def get_cchw_gradients(model, x, y, beta, eps=1e-8):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def loss_fn(p, b, x_sample, y_sample):
        return compute_sample_loss(p, b, model, x_sample, y_sample)

    grad_fn = grad(loss_fn)
    per_sample_grads_dict = vmap(grad_fn, in_dims=(None, None, 0, 0))(params, buffers, x, y)

    batch_size = x.shape[0]
    num_classes = 10

    # Flatten grads efficiently
    # Instead of cat, we can work with a list of tensors if we just need dot products
    # But for cosine sim, we need the total norm.

    # Compute per-sample norms squared
    sample_norms_sq = torch.zeros(batch_size, device=x.device)
    for g in per_sample_grads_dict.values():
        sample_norms_sq += g.reshape(batch_size, -1).pow(2).sum(dim=1)
    sample_norms = torch.sqrt(sample_norms_sq + eps)

    # Compute class mean gradients (unnormalized)
    # class_means: dict of (num_classes, *param_shape)
    class_means_dict = {}
    class_counts = torch.zeros(num_classes, device=x.device)
    for c in range(num_classes):
        mask = (y == c)
        if mask.any():
            class_counts[c] = mask.sum()
            for name, g in per_sample_grads_dict.items():
                if name not in class_means_dict:
                    shape = g.shape[1:]
                    class_means_dict[name] = torch.zeros((num_classes, *shape), device=x.device)
                class_means_dict[name][c] = g[mask].mean(dim=0)

    # Compute class mean norms squared
    class_norms_sq = torch.zeros(num_classes, device=x.device)
    for g in class_means_dict.values():
        class_norms_sq += g.reshape(num_classes, -1).pow(2).sum(dim=1)
    class_norms = torch.sqrt(class_norms_sq + eps)

    # Compute dot products between each sample and each class mean
    # sims: (batch_size, num_classes)
    sims = torch.zeros((batch_size, num_classes), device=x.device)
    for name, g_samples in per_sample_grads_dict.items():
        g_samples_flat = g_samples.reshape(batch_size, -1)
        g_means_flat = class_means_dict[name].reshape(num_classes, -1)
        sims += torch.matmul(g_samples_flat, g_means_flat.t())

    # Final cosine similarity
    # sims[i, c] = dot(g_i, class_mean_c) / (norm(g_i) * norm(class_mean_c))
    sims = sims / (sample_norms.unsqueeze(1) * class_norms.unsqueeze(0) + eps)

    # h_i: average similarity with other valid classes
    y_mask = torch.zeros((batch_size, num_classes), device=x.device)
    y_mask.scatter_(1, y.unsqueeze(1), 1.0) # (B, 10) with 1 at (i, y_i)

    valid_classes = (class_counts > 0).float().unsqueeze(0) # (1, 10)
    other_valid_mask = valid_classes * (1.0 - y_mask) # (B, 10)

    h_sum = (sims * other_valid_mask).sum(dim=1)
    h_count = other_valid_mask.sum(dim=1)
    h = h_sum / (h_count + eps) # (B,)

    # Weights
    weights = torch.exp(beta * h)
    w_mean = weights.mean()
    if w_mean > eps:
        weights = weights / w_mean
    else:
        weights = torch.ones_like(weights)

    # Compute weighted gradients
    weighted_grads = {}
    for name, g in per_sample_grads_dict.items():
        w_expanded = weights.view(-1, *([1] * (g.dim() - 1)))
        weighted_grads[name] = (w_expanded * g).mean(dim=0)

    return weighted_grads
