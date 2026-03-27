import torch
import torch.nn as nn
from model import NystromKernelLayer

def test_nystrom_kernel_layer_forward():
    input_dim = 10
    num_landmarks = 5
    batch_size = 2

    layer = NystromKernelLayer(input_dim, num_landmarks)
    x = torch.randn(batch_size, input_dim)

    phi_x = layer(x)

    assert phi_x.shape == (batch_size, num_landmarks)
    assert not torch.isnan(phi_x).any()

def test_nystrom_kernel_layer_backward():
    input_dim = 10
    num_landmarks = 5
    batch_size = 2

    layer = NystromKernelLayer(input_dim, num_landmarks)
    x = torch.randn(batch_size, input_dim, requires_grad=True)

    phi_x = layer(x)
    loss = phi_x.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.landmarks.grad is not None
    assert layer.log_gamma.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(layer.landmarks.grad).any()
    assert not torch.isnan(layer.log_gamma.grad).any()

def test_nystrom_kernel_layer_stability():
    # Test stability with identical inputs or near-zero eigenvalues
    input_dim = 10
    num_landmarks = 5
    batch_size = 2

    layer = NystromKernelLayer(input_dim, num_landmarks)
    # Make landmarks all zero
    nn.init.zeros_(layer.landmarks)

    x = torch.randn(batch_size, input_dim, requires_grad=True)
    phi_x = layer(x)
    loss = phi_x.sum()
    loss.backward()

    assert not torch.isnan(phi_x).any()
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(layer.landmarks.grad).any()

if __name__ == "__main__":
    test_nystrom_kernel_layer_forward()
    test_nystrom_kernel_layer_backward()
    test_nystrom_kernel_layer_stability()
    print("Logic tests passed!")
