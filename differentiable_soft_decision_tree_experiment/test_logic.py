import torch
from model import SoftDecisionTree

def test_sdt_differentiability():
    input_dim = 10
    output_dim = 5
    depth = 3
    model = SoftDecisionTree(input_dim, output_dim, depth=depth)

    x = torch.randn(8, input_dim, requires_grad=True)
    out = model(x)

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    print("Input gradient check passed.")

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"
    print("Parameters gradient check passed.")

def test_sdt_output_shape():
    batch_size = 16
    input_dim = 10
    output_dim = 5
    depth = 4
    model = SoftDecisionTree(input_dim, output_dim, depth=depth)

    x = torch.randn(batch_size, input_dim)
    out = model(x)

    assert out.shape == (batch_size, output_dim)
    print(f"Output shape check passed: {out.shape}")

if __name__ == "__main__":
    test_sdt_differentiability()
    test_sdt_output_shape()
    print("All tests passed!")
