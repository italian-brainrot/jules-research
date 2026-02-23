import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, grad, functional_call
import pytest

def compute_per_sample_grads(model, x, y):
    params = dict(model.named_parameters())
    param_names = list(params.keys())
    param_values = tuple(params.values())

    def loss_fn(p_values, x_single, y_single):
        p_dict = {name: val for name, val in zip(param_names, p_values)}
        logits = functional_call(model, p_dict, (x_single.unsqueeze(0),))
        return F.cross_entropy(logits, y_single.unsqueeze(0))

    grad_fn = grad(loss_fn)
    v_grad_fn = vmap(grad_fn, in_dims=(None, 0, 0))
    return v_grad_fn(param_values, x, y)

def test_rogc_clipping():
    torch.manual_seed(42)
    model = nn.Linear(10, 2)
    x = torch.randn(5, 10)
    y = torch.randint(0, 2, (5,))

    per_sample_grads = compute_per_sample_grads(model, x, y)
    batch_size = x.size(0)

    # Compute norms
    norms = torch.zeros(batch_size)
    for g in per_sample_grads:
        norms += g.view(batch_size, -1).pow(2).sum(dim=1)
    norms = norms.sqrt()

    k = 1.5
    q1 = torch.quantile(norms, 0.25)
    q3 = torch.quantile(norms, 0.75)
    iqr = q3 - q1
    threshold = q3 + k * iqr

    scale = torch.clamp(threshold / (norms + 1e-8), max=1.0)

    # Check that scale works
    clipped_norms = norms * scale
    for val in clipped_norms:
        assert val <= threshold + 1e-6

    # If one norm is much larger than others, it should be clipped
    norms_with_outlier = torch.tensor([1.0, 1.1, 1.2, 0.9, 10.0])
    q1_o = torch.quantile(norms_with_outlier, 0.25) # 1.0
    q3_o = torch.quantile(norms_with_outlier, 0.75) # 1.2
    iqr_o = q3_o - q1_o # 0.2
    threshold_o = q3_o + 1.5 * iqr_o # 1.2 + 0.3 = 1.5

    scale_o = torch.clamp(threshold_o / (norms_with_outlier + 1e-8), max=1.0)
    assert scale_o[4] < 1.0
    assert scale_o[0] == 1.0
    assert norms_with_outlier[4] * scale_o[4] == pytest.approx(1.5)

if __name__ == '__main__':
    test_rogc_clipping()
