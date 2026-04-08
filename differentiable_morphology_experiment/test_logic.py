import torch
import pytest
import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import Dilation1d, Erosion1d, Opening1d, Closing1d

def test_dilation_forward():
    # Constant input, constant kernel
    x = torch.ones(2, 1, 10)
    tau = 0.01
    dilation = Dilation1d(1, 1, 3, tau=tau)
    dilation.kernel.data.fill_(0.5)

    y = dilation(x)
    # 1.0 + 0.5 = 1.5. LSE adds tau * log(N)
    # N = in_channels * kernel_size = 1 * 3 = 3
    expected = 1.5 + tau * torch.log(torch.tensor(3.0))
    assert torch.allclose(y, expected, atol=1e-5)

def test_erosion_forward():
    x = torch.ones(2, 1, 10)
    tau = 0.01
    erosion = Erosion1d(1, 1, 3, tau=tau)
    erosion.kernel.data.fill_(0.5)

    y = erosion(x)
    # 1.0 - 0.5 = 0.5. LSE subtracts tau * log(N)
    expected = 0.5 - tau * torch.log(torch.tensor(3.0))
    assert torch.allclose(y, expected, atol=1e-5)

def test_dilation_gradient():
    x = torch.randn(2, 1, 10, requires_grad=True)
    dilation = Dilation1d(1, 4, 5, tau=0.1)
    y = dilation(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert dilation.kernel.grad is not None
    assert dilation.tau.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(dilation.kernel.grad).any()

def test_erosion_gradient():
    x = torch.randn(2, 1, 10, requires_grad=True)
    erosion = Erosion1d(1, 4, 5, tau=0.1)
    y = erosion(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert erosion.kernel.grad is not None
    assert erosion.tau.grad is not None

def test_opening_closing():
    x = torch.randn(2, 1, 10)
    op = Opening1d(1, 4, 3)
    cl = Closing1d(1, 4, 3)

    assert op(x).shape == (2, 4, 10)
    assert cl(x).shape == (2, 4, 10)

if __name__ == "__main__":
    # If run as script
    test_dilation_forward()
    test_erosion_forward()
    test_dilation_gradient()
    test_erosion_gradient()
    test_opening_closing()
    print("All logic tests passed!")
