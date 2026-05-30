import torch
import torch.nn as nn
from model import DifferentiableDelaySignature

def test_dds_output_shape():
    batch_size = 4
    input_dim = 40
    d = 3
    k = 2
    x = torch.randn(batch_size, input_dim)
    dds = DifferentiableDelaySignature(d=d, k=k)
    out = dds(x)
    # k=2 -> d + d*d = 3 + 9 = 12
    expected_dim = d + d*d
    assert out.shape == (batch_size, expected_dim)
    print("test_dds_output_shape passed!")

def test_dds_differentiability():
    batch_size = 4
    input_dim = 40
    x = torch.randn(batch_size, input_dim, requires_grad=True)
    dds = DifferentiableDelaySignature(d=3, k=2)
    out = dds(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert dds.log_tau.grad is not None
    print("test_dds_differentiability passed!")

def test_dds_tau_effect():
    # If tau changes, the output should change
    batch_size = 2
    input_dim = 40
    x = torch.randn(batch_size, input_dim)
    dds = DifferentiableDelaySignature(d=3, k=2, initial_tau=1.0)

    out1 = dds(x)

    with torch.no_grad():
        dds.log_tau.add_(0.5)

    out2 = dds(x)

    assert not torch.allclose(out1, out2)
    print("test_dds_tau_effect passed!")

if __name__ == "__main__":
    test_dds_output_shape()
    test_dds_differentiability()
    test_dds_tau_effect()
