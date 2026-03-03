import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, grad, functional_call

def compute_per_sample_grad_norms(model, x, y):
    params = dict(model.named_parameters())
    names = list(params.keys())
    values = tuple(params.values())

    def loss_fn(p_values, x_single, y_single):
        p_dict = {name: val for name, val in zip(names, p_values)}
        logits = functional_call(model, p_dict, (x_single.unsqueeze(0),))
        return F.cross_entropy(logits, y_single.unsqueeze(0))

    grad_fn = grad(loss_fn)
    v_grad_fn = vmap(grad_fn, in_dims=(None, 0, 0))

    # per_sample_grads is a tuple of tensors, each of shape (B, *param_shape)
    per_sample_grads = v_grad_fn(values, x, y)

    batch_size = x.shape[0]
    # Initialize squared norms for each sample in the batch
    sample_norms_sq = torch.zeros(batch_size, device=x.device)

    for g in per_sample_grads:
        # g has shape (B, ...)
        # We want to sum the squares of all elements in g except for the batch dimension
        sample_norms_sq += g.reshape(batch_size, -1).pow(2).sum(dim=1)

    return sample_norms_sq.sqrt()

def compute_gradient_kurtosis(norms, eps=1e-8):
    batch_size = norms.shape[0]
    if batch_size < 4: # Kurtosis is not very meaningful for very small batches
        return torch.tensor(0.0, device=norms.device)

    mu = norms.mean()
    diff = norms - mu
    var = (diff.pow(2).mean())

    if var < eps:
        return torch.tensor(0.0, device=norms.device)

    m4 = diff.pow(4).mean()
    kurtosis = m4 / (var.pow(2) + eps)
    return kurtosis - 3.0 # Excess kurtosis

def get_gkr_loss(model, x, y, lambda_gkr):
    if lambda_gkr <= 0:
        return torch.tensor(0.0, device=x.device)

    norms = compute_per_sample_grad_norms(model, x, y)
    kurtosis = compute_gradient_kurtosis(norms)
    return lambda_gkr * kurtosis
