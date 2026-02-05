import torch
from model import DSBLayer, FSBLayer, DSBMLP, FSBMLP

def test_dsb_layer():
    dim = 256
    layer = DSBLayer(dim)
    x = torch.randn(10, dim)
    y = layer(x)
    assert y.shape == x.shape
    # Check if sparsity is possible
    # (Since we initialized bias to -2.0, some values should be zero)
    # But it depends on x. Let's force it.
    layer.tau_predictor.bias.data.fill_(10.0) # High threshold
    y = layer(x)
    assert (y == 0).all()

def test_fsb_layer():
    dim = 256
    layer = FSBLayer(dim)
    x = torch.randn(10, dim)
    y = layer(x)
    assert y.shape == x.shape

    layer.tau.data.fill_(10.0) # High threshold
    y = layer(x)
    assert (y == 0).all()

def test_models():
    dsb = DSBMLP()
    fsb = FSBMLP()
    x = torch.randn(5, 40)
    assert dsb(x).shape == (5, 10)
    assert fsb(x).shape == (5, 10)

if __name__ == "__main__":
    test_dsb_layer()
    test_fsb_layer()
    test_models()
    print("All local tests passed!")
