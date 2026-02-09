import torch
from torch.func import vmap, grad, functional_call, grad_and_value

def compute_per_sample_grads_and_loss(model, x, y):
    params = dict(model.named_parameters())
    names = list(params.keys())
    values = list(params.values())

    def loss_fn(values, x, y):
        p_dict = {name: val for name, val in zip(names, values)}
        logits = functional_call(model, p_dict, (x.unsqueeze(0),))
        return torch.nn.functional.cross_entropy(logits, y.unsqueeze(0))

    # grads is a tuple of (B, *param_shape), losses is (B,)
    grads, losses = vmap(grad_and_value(loss_fn), in_dims=(None, 0, 0))(values, x, y)
    return {name: g for name, g in zip(names, grads)}, losses.mean().item()

def compute_batch_gsnr(grads_dict):
    # grads_dict: {name: tensor of shape (B, ...)}
    batch_size = next(iter(grads_dict.values())).shape[0]

    sum_sq_norm = 0.0
    sq_sum_norm = 0.0

    for g in grads_dict.values():
        # g: (B, ...)
        flat_g = g.flatten(start_dim=1) # (B, P_layer)
        sum_sq_norm += torch.sum(flat_g**2) # sum_i sum_j g_ij^2

        batch_sum = torch.sum(flat_g, dim=0) # sum_i g_ij (P_layer,)
        sq_sum_norm += torch.sum(batch_sum**2) # sum_j (sum_i g_ij)^2

    # G = ||sum g_i||^2 / sum ||g_i||^2
    gsnr = sq_sum_norm / (sum_sq_norm + 1e-8)
    return gsnr.item()
