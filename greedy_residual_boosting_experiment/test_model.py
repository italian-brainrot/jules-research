import torch
from model import GreedyMLP

def test_model_shapes():
    input_dim = 40
    hidden_dim = 128
    num_classes = 10
    num_layers = 3
    model = GreedyMLP(input_dim, hidden_dim, num_classes, num_layers)
    x = torch.randn(16, input_dim)

    # Test forward_layer
    h = model.forward_layer(x, 0)
    assert h.shape == (16, hidden_dim)

    h = model.forward_layer(x, 2)
    assert h.shape == (16, hidden_dim)

    # Test forward_head
    y = model.forward_head(x, 1)
    assert y.shape == (16, num_classes)

    # Test forward_boost
    y = model.forward_boost(x, 2)
    assert y.shape == (16, num_classes)

    # Test forward (last head)
    y = model.forward(x)
    assert y.shape == (16, num_classes)

    print("Model shape tests passed!")

if __name__ == "__main__":
    test_model_shapes()
