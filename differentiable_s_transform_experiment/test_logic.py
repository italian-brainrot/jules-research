import torch
from model import DifferentiableSTLayer

def test_st_differentiability():
    batch_size = 2
    channels = 1
    length = 40
    x = torch.randn(batch_size, channels, length, requires_grad=True)
    layer = DifferentiableSTLayer(sigma=1.0, learnable_sigma=True)

    st = layer(x)
    loss = torch.abs(st).sum()
    loss.backward()

    assert x.grad is not None
    assert layer.sigma.grad is not None
    print("Gradients with respect to input and sigma are present.")

def test_st_output_shape():
    batch_size = 2
    channels = 1
    length = 40
    x = torch.randn(batch_size, channels, length)
    layer = DifferentiableSTLayer(sigma=1.0)

    st = layer(x)
    # Output should be (B, C, N, N)
    assert st.shape == (batch_size, channels, length, length)
    print(f"Output shape verified: {st.shape}")

def test_st_dc_component():
    # DC component (n=0) should be constant across time and equal to mean
    length = 8
    x = torch.randn(1, 1, length)
    layer = DifferentiableSTLayer()
    st = layer(x)

    st_dc = st[0, 0, 0, :]
    expected_dc = torch.mean(x)

    assert torch.allclose(st_dc.real, expected_dc, atol=1e-5)
    assert torch.allclose(st_dc.imag, torch.tensor(0.0), atol=1e-5)
    print("DC component (n=0) verified.")

if __name__ == "__main__":
    test_st_differentiability()
    test_st_output_shape()
    test_st_dc_component()
