import torch
from model import VolterraLayer, VolterraMLP

def test_volterra_layer_shape():
    in_features = 10
    out_features = 5
    batch_size = 3
    layer = VolterraLayer(in_features, out_features)
    x = torch.randn(batch_size, in_features)
    out = layer(x)
    assert out.shape == (batch_size, out_features)

def test_volterra_layer_grad():
    in_features = 10
    out_features = 5
    batch_size = 3
    layer = VolterraLayer(in_features, out_features)
    x = torch.randn(batch_size, in_features, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert layer.linear.weight.grad is not None
    assert layer.quad_weight.grad is not None

def test_volterra_mlp_shape():
    in_dim = 40
    hidden_dim = 20
    out_dim = 10
    batch_size = 4
    model = VolterraMLP(in_dim, hidden_dim, out_dim)
    x = torch.randn(batch_size, in_dim)
    out = model(x)
    assert out.shape == (batch_size, out_dim)

def test_volterra_quadratic_logic():
    # Simple case: 2 inputs, 1 output, only one quadratic term x0*x1
    # num_quad for 2 inputs is 2*(3)/2 = 3 (x0^2, x0*x1, x1^2)
    in_features = 2
    out_features = 1
    layer = VolterraLayer(in_features, out_features)

    # Zero out linear part
    nn = torch.nn
    nn.init.zeros_(layer.linear.weight)
    nn.init.zeros_(layer.linear.bias)

    # triu_indices for (2,2) are (0,0), (0,1), (1,1)
    # Set quad_weight for x0*x1 to 1.0, others to 0
    with torch.no_grad():
        layer.quad_weight.zero_()
        layer.quad_weight[0, 1] = 1.0

    x = torch.tensor([[2.0, 3.0]])
    out = layer(x)
    # x0*x1 = 2*3 = 6
    assert torch.allclose(out, torch.tensor([[6.0]]))

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
