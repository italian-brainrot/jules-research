import torch
from torch import nn
from torch.func import functional_call, vmap, grad

class MLP(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, output_size=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

def compute_loss(params, model, x, y):
    # x: (D,), y: scalar
    # functional_call expects a dict of parameters
    logits = functional_call(model, params, (x.unsqueeze(0),))
    return nn.functional.cross_entropy(logits, y.unsqueeze(0))

def get_sgo_penalty(params, model, x_batch, y_batch, class_aware=False):
    # params: dict of model parameters
    # x_batch: (B, D), y_batch: (B,)

    batch_size = x_batch.shape[0]

    # Define a wrapper for grad that takes single x and y
    def single_loss(p, x, y):
        return compute_loss(p, model, x, y)

    # vmap(grad) computes gradients for each sample in the batch
    # per_sample_grads_dict is a dict where each value has shape (B, *param_shape)
    per_sample_grads_dict = vmap(grad(single_loss), in_dims=(None, 0, 0))(params, x_batch, y_batch)

    # Flatten and concatenate gradients for each sample
    flat_grads = []
    for p in per_sample_grads_dict.values():
        flat_grads.append(p.reshape(batch_size, -1))

    G = torch.cat(flat_grads, dim=1) # (B, P)

    # Normalize gradients to compute cosine similarity
    G_norm = torch.norm(G, dim=1, keepdim=True) + 1e-8
    G_normalized = G / G_norm

    # Compute cosine similarity matrix: S[i, j] = cos(g_i, g_j)
    S = torch.mm(G_normalized, G_normalized.t()) # (B, B)

    device = S.device
    if class_aware:
        # Mask for pairs with different labels
        y_i = y_batch.unsqueeze(1)
        y_j = y_batch.unsqueeze(0)
        diff_class_mask = (y_i != y_j).float()

        # Upper triangle mask (excluding diagonal) to avoid double counting and self-similarity
        triu_mask = torch.triu(torch.ones(batch_size, batch_size, device=device), diagonal=1)
        final_mask = diff_class_mask * triu_mask

        num_pairs = final_mask.sum()
        if num_pairs > 0:
            # We penalize squared cosine similarity
            penalty = (S * final_mask).pow(2).sum() / num_pairs
        else:
            penalty = torch.tensor(0.0, device=device)
    else:
        # All distinct pairs
        triu_mask = torch.triu(torch.ones(batch_size, batch_size, device=device), diagonal=1)
        num_pairs = triu_mask.sum()
        if num_pairs > 0:
            penalty = (S * triu_mask).pow(2).sum() / num_pairs
        else:
            penalty = torch.tensor(0.0, device=device)

    return penalty
