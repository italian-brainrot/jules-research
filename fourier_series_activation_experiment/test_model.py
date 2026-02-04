import torch
import torch.nn as nn
from fourier_series_activation_experiment.model import LFSA, ORelu, Snake, MLP

def test_lfsa_forward():
    x = torch.randn(10, 20)
    lfsa = LFSA(num_parameters=20, K=4)
    out = lfsa(x)
    assert out.shape == x.shape

    # Test per-neuron
    lfsa_shared = LFSA(num_parameters=1, K=4)
    out_shared = lfsa_shared(x)
    assert out_shared.shape == x.shape

def test_orelu_forward():
    x = torch.randn(10, 20)
    orelu = ORelu(num_parameters=20)
    out = orelu(x)
    assert out.shape == x.shape

def test_snake_forward():
    x = torch.randn(10, 20)
    snake = Snake(num_parameters=20)
    out = snake(x)
    assert out.shape == x.shape

def test_mlp_creation():
    model = MLP(40, [128, 64], 10, activation_type='lfsa', num_params='per_neuron')
    x = torch.randn(5, 40)
    out = model(x)
    assert out.shape == (5, 10)

if __name__ == "__main__":
    test_lfsa_forward()
    test_orelu_forward()
    test_snake_forward()
    test_mlp_creation()
    print("All tests passed!")
