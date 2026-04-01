import torch
import torch.nn as nn
from model import LVEN

def test_lven_logic():
    batch_size = 8
    input_dim = 40
    output_dim = 10
    latent_dim = 16
    num_experts = 32

    model = LVEN(input_dim, output_dim, latent_dim, num_experts)
    x = torch.randn(batch_size, input_dim)

    # Forward pass
    output = model(x)
    assert output.shape == (batch_size, output_dim), f"Expected shape {(batch_size, output_dim)}, got {output.shape}"

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check if all parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} does not have a gradient"
        assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaNs"
        print(f"Gradient for {name} verified.")

    print("LVEN logic and gradient flow verified.")

if __name__ == "__main__":
    test_lven_logic()
