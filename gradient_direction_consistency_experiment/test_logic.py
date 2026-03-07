import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, grad, functional_call

def compute_gdcr_loss(model, params, x, y):
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

    # Compute average cosine similarity efficiently
    batch_size = x.shape[0]
    sum_grads = grad_directions.sum(dim=0)
    sum_sq_norm = torch.sum(sum_grads**2)
    avg_cos_sim = (sum_sq_norm - batch_size) / (batch_size * (batch_size - 1))

    # GDCR loss: minimize (1 - avg_cos_sim)
    gdcr_loss = 1.0 - avg_cos_sim

    return gdcr_loss

# Test the logic
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 2)
    def forward(self, x):
        return self.fc(x)

model = SimpleModel()
params = dict(model.named_parameters())
x = torch.randn(4, 5)
y = torch.tensor([0, 1, 0, 1])

loss = compute_gdcr_loss(model, params, x, y)
print(f"GDCR loss: {loss.item()}")
assert loss.item() >= 0
print("Logic check passed!")
