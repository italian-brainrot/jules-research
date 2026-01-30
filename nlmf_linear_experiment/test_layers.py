import torch
from layers import NLMFLinear, LowRankLinear, KroneckerLinear, DenseLinear

def test_layers():
    x = torch.randn(5, 40)

    # Test Dense
    l = DenseLinear(40, 100)
    y = l(x)
    assert y.shape == (5, 100)
    print("Dense test passed")

    # Test LowRank
    l = LowRankLinear(40, 100, rank=4)
    y = l(x)
    assert y.shape == (5, 100)
    print("LowRank test passed")

    # Test NLMF
    l = NLMFLinear(40, 100, rank=4)
    y = l(x)
    assert y.shape == (5, 100)
    print("NLMF test passed")

    # Test Kronecker
    l = KroneckerLinear(10, 4, 10, 10) # in=40, out=100
    y = l(x)
    assert y.shape == (5, 100)
    print("Kronecker test passed")

if __name__ == "__main__":
    test_layers()
