import torch
import pytest
from model import DOMPLayer

def test_domp_differentiability():
    batch_size = 4
    input_dim = 10
    dict_size = 20
    num_iterations = 5

    x = torch.randn(batch_size, input_dim, requires_grad=True)
    domp = DOMPLayer(input_dim, dict_size, num_iterations=num_iterations)

    coeffs = domp(x)
    loss = coeffs.sum()
    loss.backward()

    assert x.grad is not None
    assert domp.dictionary.grad is not None
    assert domp.log_beta.grad is not None

    print("Differentiability test passed.")

def test_domp_output_shape():
    batch_size = 4
    input_dim = 10
    dict_size = 20
    num_iterations = 5

    x = torch.randn(batch_size, input_dim)
    domp = DOMPLayer(input_dim, dict_size, num_iterations=num_iterations)

    coeffs = domp(x)
    assert coeffs.shape == (batch_size, dict_size)
    print("Output shape test passed.")

if __name__ == "__main__":
    test_domp_differentiability()
    test_domp_output_shape()
