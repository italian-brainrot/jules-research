import torch
from differentiable_ista_experiment.model import ISTALayer, soft_threshold

def test_soft_threshold():
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    threshold = 1.0
    expected = torch.tensor([-1.0, 0.0, 0.0, 0.0, 1.0])
    output = soft_threshold(x, threshold)
    assert torch.allclose(output, expected), f"Expected {expected}, got {output}"
    print("test_soft_threshold passed!")

def test_ista_differentiability():
    input_dim = 10
    sparse_dim = 20
    num_iterations = 5
    batch_size = 4

    layer = ISTALayer(input_dim, sparse_dim, num_iterations)
    x = torch.randn(batch_size, input_dim, requires_grad=True)

    output = layer(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None, "Gradient with respect to input is None"
    assert layer.W.grad is not None, "Gradient with respect to W is None"
    assert layer.log_eta.grad is not None, "Gradient with respect to log_eta is None"
    assert layer.log_lambda.grad is not None, "Gradient with respect to log_lambda is None"

    print("test_ista_differentiability passed!")

def test_ista_output_shape():
    input_dim = 10
    sparse_dim = 20
    num_iterations = 5
    batch_size = 4

    layer = ISTALayer(input_dim, sparse_dim, num_iterations)
    x = torch.randn(batch_size, input_dim)
    output = layer(x)

    assert output.shape == (batch_size, sparse_dim), f"Expected shape {(batch_size, sparse_dim)}, got {output.shape}"
    print("test_ista_output_shape passed!")

if __name__ == "__main__":
    test_soft_threshold()
    test_ista_differentiability()
    test_ista_output_shape()
