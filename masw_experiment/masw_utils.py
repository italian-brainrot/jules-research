import torch
from torch.func import vmap, grad, functional_call

def compute_masw_gradients(model, optimizer, inputs, targets, gamma):
    params = dict(model.named_parameters())
    names = params.keys()
    values = tuple(params.values())

    # 1. Compute per-sample gradients
    def compute_loss(params_values, x, y):
        p_dict = {name: val for name, val in zip(names, params_values)}
        out = functional_call(model, p_dict, (x.unsqueeze(0),))
        loss = torch.nn.functional.cross_entropy(out, y.unsqueeze(0))
        return loss

    grad_fn = grad(compute_loss)
    v_grad_fn = vmap(grad_fn, in_dims=(None, 0, 0))
    per_sample_grads = v_grad_fn(values, inputs, targets)

    batch_size = inputs.shape[0]

    # 2. Extract momentum
    momentums = []
    has_momentum = True
    for p in model.parameters():
        if p in optimizer.state and 'exp_avg' in optimizer.state[p]:
            momentums.append(optimizer.state[p]['exp_avg'])
        else:
            has_momentum = False
            break

    if not has_momentum:
        # Fallback to mean gradients if no momentum available yet
        return [g.mean(dim=0) for g in per_sample_grads]

    # 3. Compute similarities
    flat_grads = []
    for g in per_sample_grads:
        flat_grads.append(g.reshape(batch_size, -1))
    flat_grads = torch.cat(flat_grads, dim=1) # (B, TotalParams)

    flat_momentum = []
    for m in momentums:
        flat_momentum.append(m.reshape(-1))
    flat_momentum = torch.cat(flat_momentum, dim=0) # (TotalParams,)

    eps = 1e-8
    dot_products = torch.mv(flat_grads, flat_momentum)
    grad_norms = torch.norm(flat_grads, dim=1)
    momentum_norm = torch.norm(flat_momentum)

    similarities = dot_products / (grad_norms * momentum_norm + eps)

    # Weighting: w_i = exp(gamma * s_i), then normalized to sum to B (mean to 1)
    weights = torch.exp(gamma * similarities)
    weights = weights / (weights.mean() + eps)

    # 4. Aggregate gradients
    final_grads = []
    for g in per_sample_grads:
        w_expanded = weights.view(batch_size, *([1] * (g.dim() - 1)))
        final_grads.append((g * w_expanded).mean(dim=0))

    return final_grads
