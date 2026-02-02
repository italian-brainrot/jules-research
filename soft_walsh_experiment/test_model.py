import torch
from model import SoftWalshLayer, SoftWalshNetwork, MLP

def test_swl_shape():
    batch_size = 16
    in_features = 40
    num_terms = 128
    swl = SoftWalshLayer(in_features, num_terms)
    x = torch.randn(batch_size, in_features)
    y = swl(x)
    assert y.shape == (batch_size, num_terms)

def test_swn_shape():
    batch_size = 16
    in_features = 40
    num_terms = 128
    out_features = 10
    swn = SoftWalshNetwork(in_features, num_terms, out_features)
    x = torch.randn(batch_size, in_features)
    y = swn(x)
    assert y.shape == (batch_size, out_features)

def test_mlp_shape():
    batch_size = 16
    in_features = 40
    hidden_size = 128
    out_features = 10
    mlp = MLP(in_features, hidden_size, out_features)
    x = torch.randn(batch_size, in_features)
    y = mlp(x)
    assert y.shape == (batch_size, out_features)

if __name__ == "__main__":
    test_swl_shape()
    test_swn_shape()
    test_mlp_shape()
    print("All tests passed!")
