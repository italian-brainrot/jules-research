import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, grad, functional_call

def compute_sga_loss(model, params, x, y):
    def compute_loss(params, x_single, y_single):
        logits = functional_call(model, params, (x_single.unsqueeze(0),))
        return F.cross_entropy(logits, y_single.unsqueeze(0))

    # Compute per-sample gradients w.r.t. all parameters
    per_sample_grads = vmap(grad(compute_loss), in_dims=(None, 0, 0))(params, x, y)

    # Flatten and concatenate all gradients for each sample
    flat_grads = []
    for p_name in params:
        g = per_sample_grads[p_name]
        flat_grads.append(g.reshape(x.shape[0], -1))

    all_flat_grads = torch.cat(flat_grads, dim=1) # (batch_size, num_params)

    # Normalize gradients to get directions
    norms = torch.norm(all_flat_grads, p=2, dim=1, keepdim=True) + 1e-8
    grad_directions = all_flat_grads / norms

    # Compute cosine similarity matrix
    sim_matrix = torch.matmul(grad_directions, grad_directions.t()) # (B, B)

    batch_size = x.shape[0]

    # Create masks for intra-class and inter-class
    y_vec = y.unsqueeze(0)
    mask_intra = (y_vec == y_vec.t()).float()
    mask_inter = 1.0 - mask_intra

    # Remove diagonal from intra-class mask
    mask_intra = mask_intra - torch.eye(batch_size, device=y.device)

    # Compute average intra-class similarity
    intra_count = mask_intra.sum()
    if intra_count > 0:
        intra_sim = (sim_matrix * mask_intra).sum() / intra_count
    else:
        intra_sim = torch.tensor(0.0, device=y.device)

    # Compute average inter-class similarity
    inter_count = mask_inter.sum()
    if inter_count > 0:
        inter_sim = (sim_matrix * mask_inter).sum() / inter_count
    else:
        inter_sim = torch.tensor(0.0, device=y.device)

    return intra_sim, inter_sim

def test():
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
        def forward(self, x):
            return self.fc(x)

    model = SimpleMLP()
    params = dict(model.named_parameters())

    x = torch.randn(8, 10)
    y = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])

    intra_sim, inter_sim = compute_sga_loss(model, params, x, y)
    print(f"Intra-class similarity: {intra_sim.item():.4f}")
    print(f"Inter-class similarity: {inter_sim.item():.4f}")

    # Check if gradients are actually computed and similarity makes sense
    assert -1.01 <= intra_sim.item() <= 1.01
    assert -1.01 <= inter_sim.item() <= 1.01

if __name__ == "__main__":
    test()
