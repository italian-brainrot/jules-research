import torch
import pytest
from model import DifferentiableSAX

def test_sax_shapes():
    B, L = 10, 40
    num_segments = 8
    alphabet_size = 4
    x = torch.randn(B, L)
    sax_layer = DifferentiableSAX(num_segments=num_segments, alphabet_size=alphabet_size)
    out = sax_layer(x)
    assert out.shape == (B, num_segments, alphabet_size)
    # Probs should sum to 1
    assert torch.allclose(out.sum(dim=-1), torch.ones(B, num_segments), atol=1e-6)

def test_sax_differentiability():
    B, L = 5, 40
    num_segments = 8
    alphabet_size = 4
    x = torch.randn(B, L, requires_grad=True)
    sax_layer = DifferentiableSAX(num_segments=num_segments, alphabet_size=alphabet_size)
    out = sax_layer(x)
    loss = out.pow(2).sum()
    loss.backward()
    assert x.grad is not None
    assert sax_layer.breakpoints.grad is not None

def test_sax_breakpoints_order():
    # Even if breakpoints are initialized out of order, they should be sorted in forward
    B, L = 2, 20
    num_segments = 4
    alphabet_size = 3
    x = torch.randn(B, L)
    sax_layer = DifferentiableSAX(num_segments=num_segments, alphabet_size=alphabet_size)
    with torch.no_grad():
        sax_layer.breakpoints.data = torch.tensor([1.0, -1.0])
    out = sax_layer(x)
    assert out.shape == (B, num_segments, alphabet_size)
    assert torch.allclose(out.sum(dim=-1), torch.ones(B, num_segments), atol=1e-6)

if __name__ == "__main__":
    test_sax_shapes()
    test_sax_differentiability()
    test_sax_breakpoints_order()
    print("All tests passed!")
