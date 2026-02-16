import torch
from torch.func import vmap, grad, functional_call
import torch.nn.functional as F

def compute_sample_loss(params, buffers, model, x, y):
    # x: (input_dim,), y: ()
    # Combine params and buffers into a single dictionary for functional_call
    state_dict = {**params, **buffers}
    logits = functional_call(model, state_dict, (x.unsqueeze(0),))
    return F.cross_entropy(logits, y.unsqueeze(0))

def get_gasw_gradients(model, x, y, gamma, mode='GASW'):
    """
    Computes weighted gradients for a batch.
    mode: 'GASW' (Alignment) or 'GDSW' (Diversity)
    """
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    # Compute per-sample gradients using torch.func
    # We use a wrapper to match the expected signature of vmap
    def loss_fn(p, b, x_sample, y_sample):
        return compute_sample_loss(p, b, model, x_sample, y_sample)

    grad_fn = grad(loss_fn)
    # vmap over x and y (dim 0)
    per_sample_grads_dict = vmap(grad_fn, in_dims=(None, None, 0, 0))(params, buffers, x, y)

    batch_size = x.shape[0]

    # Flatten per-sample gradients to compute cosine similarity
    flattened_grads = []
    for name in params.keys():
        g = per_sample_grads_dict[name]
        flattened_grads.append(g.reshape(batch_size, -1))

    # G: (batch_size, total_params)
    G = torch.cat(flattened_grads, dim=1)

    # Average gradient across batch: (total_params,)
    G_avg = G.mean(dim=0)

    # Compute cosine similarity between each sample's gradient and the batch average
    eps = 1e-8
    G_norm = torch.norm(G, dim=1) # (batch_size,)
    G_avg_norm = torch.norm(G_avg) # ()
    dot_prod = torch.mv(G, G_avg) # (batch_size,)

    sim = dot_prod / (G_norm * G_avg_norm + eps)

    # Compute weights
    if mode == 'GASW':
        # Alignment: weight samples that align with the average
        weights = torch.clamp(sim, min=0.0) ** gamma
    else:
        # Diversity: weight samples that differ from the average
        # sim is in [-1, 1], so 1-sim is in [0, 2]
        weights = torch.clamp(1.0 - sim, min=0.0) ** gamma

    # Normalize weights so that their mean is 1.0 (equivalent to sum = batch_size)
    w_mean = weights.mean()
    if w_mean > eps:
        weights = weights / w_mean
    else:
        # Fallback to uniform weights if all weights are zero
        weights = torch.ones_like(weights)

    # Compute weighted average gradients for each parameter
    weighted_grads = {}
    for name in params.keys():
        g = per_sample_grads_dict[name] # (batch_size, ...)
        # Expand weights to match parameter dimensions
        w_expanded = weights.view(-1, *([1] * (g.dim() - 1)))
        weighted_grads[name] = (w_expanded * g).mean(dim=0)

    return weighted_grads
