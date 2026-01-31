import torch
from models import QuantileFunction

def test_gradients():
    from torch.autograd import gradcheck

    # Test with double precision for gradcheck
    x = torch.randn(2, 3, 5, dtype=torch.double, requires_grad=True)
    q = torch.tensor([[0.3, 0.3, 0.3], [0.7, 0.7, 0.7]], dtype=torch.double, requires_grad=True)
    alpha = torch.tensor([[5.0, 5.0, 5.0], [10.0, 10.0, 10.0]], dtype=torch.double, requires_grad=True)

    # We need a wrapper to handle the iterations argument
    def func(x, q, alpha):
        return QuantileFunction.apply(x, q, alpha, 40)

    print("Running gradcheck...")
    test = gradcheck(func, (x, q, alpha), eps=1e-6, atol=1e-4)
    print("Gradcheck passed:", test)

if __name__ == "__main__":
    test_gradients()
