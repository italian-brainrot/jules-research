import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, grad, functional_call
import sys
import os

# Add parent directory to path to import train
sys.path.append(os.getcwd())
from gradient_norm_variance_regularization.train import MLP, compute_gnvr_grads

def test_gnvr_logic():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use a smaller model for testing
    model = MLP(input_size=10, hidden_size=20, output_size=2).to(device)
    x = torch.randn(16, 10).to(device)
    y = torch.randint(0, 2, (16,)).to(device)

    # We want to test if the GNVR term reduces variance.
    # We'll use a very large lambda and a small task loss (by scaling logits)
    # Actually, let's just use lambd=100.0
    lambd = 100.0

    def get_norms_and_var(m, xb, yb):
        params = dict(m.named_parameters())
        param_names = list(params.keys())
        param_values = tuple(params.values())
        buffers = dict(m.named_buffers())

        def loss_fn(p_vals, b, x_s, y_s):
            p_dict = {name: val for name, val in zip(param_names, p_vals)}
            sd = {**p_dict, **b}
            logits = functional_call(m, sd, (x_s.unsqueeze(0),))
            return F.cross_entropy(logits, y_s.unsqueeze(0))

        def per_sample_grad_norm(p_vals, b, x_s, y_s):
            g = grad(loss_fn)(p_vals, b, x_s, y_s)
            norm_sq = sum(p_g.pow(2).sum() for p_g in g)
            return torch.sqrt(norm_sq + 1e-8)

        norms = vmap(per_sample_grad_norm, in_dims=(None, None, 0, 0))(param_values, buffers, xb, yb)
        return norms, torch.var(norms)

    # Initial state
    norms_init, var_init = get_norms_and_var(model, x, y)
    print(f"Initial norms: {norms_init.detach().cpu().numpy()}")
    print(f"Initial variance: {var_init.item():.8f}")

    # Compute GNVR grads
    grads = compute_gnvr_grads(model, x, y, lambd)

    # Take a small step
    lr = 0.01
    with torch.no_grad():
        for p, g in zip(model.parameters(), grads):
            p.copy_(p - lr * g)

    # New state
    norms_new, var_new = get_norms_and_var(model, x, y)
    print(f"New norms: {norms_new.detach().cpu().numpy()}")
    print(f"New variance: {var_new.item():.8f}")

    if var_new < var_init:
        print("Success: Variance decreased!")
    else:
        # It's possible that the task loss increased enough to offset the variance decrease
        # if lambd wasn't large enough, but with lambd=100 it should decrease.
        print("Failure: Variance did not decrease.")

    # Check if the grads are not zero
    total_grad_norm = sum(g.pow(2).sum() for g in grads).sqrt()
    print(f"Total grad norm: {total_grad_norm.item():.6f}")
    assert total_grad_norm > 0

if __name__ == "__main__":
    test_gnvr_logic()
