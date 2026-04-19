import torch
from model import HermiteTransformLayer

def test_hermite_transform_logic():
    batch_size = 4
    input_dim = 40
    n_coeffs = 10

    layer = HermiteTransformLayer(input_dim, n_coeffs)
    x = torch.randn(batch_size, input_dim, requires_grad=True)

    # Check output shape
    coeffs = layer(x)
    assert coeffs.shape == (batch_size, n_coeffs)
    print(f"Output shape verified: {coeffs.shape}")

    # Check differentiability wrt input
    loss = coeffs.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print("Gradients wrt input verified.")

    # Check differentiability wrt scale parameter
    assert layer.log_scale.grad is not None
    assert not torch.isnan(layer.log_scale.grad).any()
    print("Gradients wrt log_scale verified.")

    # Check that basis functions are actually generated
    psi = layer.get_hermite_functions()
    assert psi.shape == (n_coeffs, input_dim)
    print(f"Basis functions shape verified: {psi.shape}")

if __name__ == "__main__":
    test_hermite_transform_logic()
