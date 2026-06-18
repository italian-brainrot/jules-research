import torch
import torch.nn as nn
import torch.nn.functional as F

def nca_loss(embeddings, targets, temperature=1.0):
    """
    Computes the Neighborhood Components Analysis (NCA) loss.

    Args:
        embeddings: (batch_size, embedding_dim) tensor of features.
        targets: (batch_size,) tensor of class labels.
        temperature: scalar temperature for the softmax.

    Returns:
        A scalar loss value.
    """
    batch_size = embeddings.size(0)
    if batch_size <= 1:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # Compute pairwise squared Euclidean distances
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 <a, b>
    sq_norms = torch.sum(embeddings**2, dim=1, keepdim=True)
    dist_sq = sq_norms + sq_norms.t() - 2 * torch.mm(embeddings, embeddings.t())
    dist_sq = F.relu(dist_sq) # Ensure non-negative

    # Mask out self-distances (diagonal)
    mask = torch.eye(batch_size, device=embeddings.device).bool()
    dist_sq = dist_sq.masked_fill(mask, float('inf'))

    # Compute probabilities p_ij = exp(-d_ij^2 / T) / sum_{k!=i} exp(-d_ik^2 / T)
    # Use log-sum-exp for stability
    logits = -dist_sq / temperature

    # We need to compute log(p_ij)
    # log_p_ij = logits - logsumexp(logits, dim=1, keepdim=True)
    log_probs = F.log_softmax(logits, dim=1)

    # p_i = sum_{j: y_j = y_i, j!=i} p_ij
    # We want to minimize -sum_i log(p_i)

    # Create mask for same-class pairs (excluding self)
    target_mask = (targets.unsqueeze(0) == targets.unsqueeze(1)) & (~mask)

    # To compute log(sum exp(log_p_ij)), we can use logsumexp on the relevant entries
    # log_p_i = logsumexp(log_p_ij over j where y_j == y_i)

    # We need to handle the case where a class has only one sample in the batch
    # In that case, target_mask[i] will be all False.

    masked_log_probs = log_probs.masked_fill(~target_mask, -float('inf'))

    # logsumexp over dim 1
    log_p_i = torch.logsumexp(masked_log_probs, dim=1)

    # Filter out samples that have no other samples of the same class in the batch
    valid_mask = target_mask.any(dim=1)
    if not valid_mask.any():
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    loss = -log_p_i[valid_mask].mean()

    return loss

class NCALoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, targets):
        return nca_loss(embeddings, targets, self.temperature)
