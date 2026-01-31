import torch
from model import MixtureActivation, AdaptiveMLP

def test_mixture_activation():
    m = MixtureActivation()
    x = torch.randn(10, 5)
    y = m(x)
    assert y.shape == (10, 5)

    # Test gradient
    y.sum().backward()
    assert m.weights.grad is not None
    assert m.omega.grad is not None

def test_adaptive_mlp():
    m = AdaptiveMLP(40, [10, 10], 5)
    x = torch.randn(2, 40)
    y = m(x)
    assert y.shape == (2, 5)

    weights = m.get_mixture_weights()
    assert len(weights) == 2
    assert weights[0].shape == (4,)

if __name__ == "__main__":
    test_mixture_activation()
    test_adaptive_mlp()
    print("Tests passed!")
