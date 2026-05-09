import torch
from model import DDMDLayer

def test_ddmd_layer_gradient():
    B, L, k = 4, 40, 15
    x = torch.randn(B, L, requires_grad=True)
    layer = DDMDLayer(L=L, k=k)

    out = layer(x)
    loss = out.pow(2).sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()
    assert torch.abs(x.grad).sum() > 0
    print("Gradient test passed!")

def test_ddmd_layer_output_shape():
    B, L, k = 8, 40, 10
    x = torch.randn(B, L)
    layer = DDMDLayer(L=L, k=k)
    out = layer(x)
    assert out.shape == (B, 2 * k)
    print("Shape test passed!")

def test_ddmd_determinism():
    B, L, k = 4, 40, 15
    x = torch.randn(B, L)
    layer = DDMDLayer(L=L, k=k)

    out1 = layer(x)
    out2 = layer(x)

    assert torch.allclose(out1, out2, atol=1e-6)
    print("Determinism test passed!")

if __name__ == "__main__":
    test_ddmd_layer_output_shape()
    test_ddmd_layer_gradient()
    test_ddmd_determinism()
