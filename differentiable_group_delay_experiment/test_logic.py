import torch
import numpy as np
from differentiable_group_delay_experiment.model import DifferentiableGroupDelay

def test_group_delay_impulse():
    seq_len = 40
    gd_layer = DifferentiableGroupDelay()

    for k in [5, 10, 20]:
        # Impulse at position k: x[n] = delta[n-k]
        x = torch.zeros((1, seq_len))
        x[0, k] = 1.0

        gd = gd_layer(x)
        # For a pure delay delta[n-k], group delay should be k for all frequencies
        mean_gd = gd.mean().item()
        print(f"Delay k={k}, Mean GD={mean_gd}")
        assert np.isclose(mean_gd, k, atol=1e-3)

def test_group_delay_differentiability():
    seq_len = 40
    gd_layer = DifferentiableGroupDelay()
    x = torch.randn((2, seq_len), requires_grad=True)

    gd = gd_layer(x)
    loss = gd.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print("Gradients computed successfully.")

if __name__ == "__main__":
    test_group_delay_impulse()
    test_group_delay_differentiability()
