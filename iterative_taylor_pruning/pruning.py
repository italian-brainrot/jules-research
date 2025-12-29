import torch
import torch.nn as nn

def compute_taylor_saliency(model, loss_fn, data_loader):
    """
    Computes saliency for each weight based on a Taylor expansion approximation.
    Saliency = | -g*w + 0.5 * E[g^2] * w^2 |
    Gradients are averaged over the provided data.
    """
    model.eval()

    avg_grads = {name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad}
    avg_squared_grads = {name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad}
    num_samples = 0

    for inputs, targets in data_loader:
        model.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        batch_size = inputs.size(0)
        num_samples += batch_size

        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                avg_grads[name] += p.grad * batch_size
                avg_squared_grads[name] += (p.grad ** 2) * batch_size

    if num_samples == 0:
        return {name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad}

    for name in avg_grads:
        avg_grads[name] /= num_samples
        avg_squared_grads[name] /= num_samples

    saliency_scores = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            g = avg_grads[name]
            h_diag_approx = avg_squared_grads[name]
            w = p.data
            saliency = torch.abs(-g * w + 0.5 * h_diag_approx * (w ** 2))
            saliency_scores[name] = saliency

    return saliency_scores

def prune_model_with_saliency(model, saliency_scores, prune_ratio):
    """
    Prunes a model by removing a `prune_ratio` of weights with the lowest saliency scores.
    This function applies the pruning mask directly to the model's weights.
    It returns the total number of parameters and the number of pruned parameters.
    """
    all_scores = []
    for name, p in model.named_parameters():
        if p.requires_grad and name in saliency_scores:
            all_scores.append(saliency_scores[name].view(-1))

    if not all_scores:
        return 0, 0

    flat_scores = torch.cat(all_scores)
    total_params = len(flat_scores)
    num_params_to_prune = int(total_params * prune_ratio)

    if num_params_to_prune == 0:
        return total_params, 0

    if num_params_to_prune > total_params:
        num_params_to_prune = total_params

    threshold = torch.kthvalue(flat_scores, num_params_to_prune).values.item()

    pruned_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad and name in saliency_scores:
            scores = saliency_scores[name]
            mask = (scores >= threshold).float()
            p.data.mul_(mask)
            pruned_params += (mask == 0).sum().item()

    return total_params, pruned_params

def magnitude_prune(model, prune_ratio):
    """
    Baseline magnitude pruning.
    """
    all_weights = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            all_weights.append(p.data.abs().view(-1))

    if not all_weights:
        return 0, 0

    flat_weights = torch.cat(all_weights)
    total_params = len(flat_weights)
    num_params_to_prune = int(total_params * prune_ratio)

    if num_params_to_prune == 0:
        return total_params, 0

    if num_params_to_prune > total_params:
        num_params_to_prune = total_params

    threshold = torch.kthvalue(flat_weights, num_params_to_prune).values.item()

    pruned_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            mask = (p.data.abs() >= threshold).float()
            p.data.mul_(mask)
            pruned_params += (mask == 0).sum().item()

    return total_params, pruned_params
