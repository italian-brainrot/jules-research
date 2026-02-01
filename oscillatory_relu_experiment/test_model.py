import torch
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'oscillatory_relu_experiment'))
from model import ORelu, MLP

def test_orelu():
    x = torch.randn(10, 5)
    orelu = ORelu(5)
    y = orelu(x)
    assert y.shape == (10, 5)
    assert not torch.isnan(y).any()
    print("ORelu test passed!")

def test_mlp():
    x = torch.randn(10, 40)
    mlp = MLP(40, [256, 256], 10, activation_type='orelu')
    y = mlp(x)
    assert y.shape == (10, 10)
    assert not torch.isnan(y).any()
    print("MLP (orelu) test passed!")

    mlp_snake = MLP(40, [256, 256], 10, activation_type='snake')
    y_snake = mlp_snake(x)
    assert y_snake.shape == (10, 10)
    assert not torch.isnan(y_snake).any()
    print("MLP (snake) test passed!")

if __name__ == "__main__":
    test_orelu()
    test_mlp()
