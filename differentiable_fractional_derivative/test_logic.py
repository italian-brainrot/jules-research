import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FractionalDerivativeLayer(nn.Module):
    def __init__(self, num_orders=1, init_orders=None):
        super().__init__()
        if init_orders is None:
            # Initialize orders around 0.5, 1.0, 1.5 etc.
            init_orders = torch.linspace(0.1, 2.0, num_orders)
        else:
            init_orders = torch.tensor(init_orders)

        self.orders = nn.Parameter(init_orders)

    def forward(self, x):
        # x: (batch, length) or (batch, channels, length)
        if x.dim() == 2:
            x = x.unsqueeze(1) # (batch, 1, length)

        batch_size, channels, length = x.shape

        # Compute Grunwald-Letnikov coefficients for each order
        # c_k = c_{k-1} * (1 - (q+1)/k) = c_{k-1} * (k - q - 1) / k

        # k: 1, 2, ..., length-1
        ks = torch.arange(1, length, device=x.device, dtype=x.dtype)

        # multipliers: (num_orders, length-1)
        multipliers = (ks.unsqueeze(0) - self.orders.unsqueeze(1) - 1) / ks.unsqueeze(0)

        # c0 = 1 for all orders
        ones = torch.ones(self.orders.shape[0], 1, device=x.device, dtype=x.dtype)

        # coefficients: (num_orders, length)
        coeffs = torch.cat([ones, multipliers], dim=1)
        coeffs = torch.cumprod(coeffs, dim=1)

        # kernel: (num_orders, 1, length)
        # We need to flip the coefficients for F.conv1d to act as a causal filter
        kernel = torch.flip(coeffs, dims=[1]).unsqueeze(1)

        # Padding: Since it's causal, we pad left with (length-1)
        x_padded = F.pad(x, (length - 1, 0))

        num_orders = self.orders.shape[0]
        # Repeat kernel for each channel to use grouped convolution
        kernel_repeated = kernel.repeat(channels, 1, 1)

        # Output: (batch, channels * num_orders, length)
        out = F.conv1d(x_padded, kernel_repeated, groups=channels)

        # result: (batch, channels, num_orders, length)
        result = out.view(batch_size, channels, num_orders, length)

        return result

def test_logic():
    print("Testing DFD logic...")

    # 1. Test integer orders
    # q=1 should be backward difference: D^1 x_t = x_t - x_{t-1}
    # q=0 should be identity: D^0 x_t = x_t

    layer = FractionalDerivativeLayer(init_orders=[0.0, 1.0, 2.0])
    x = torch.tensor([[1.0, 2.0, 4.0, 7.0, 11.0]]) # (1, 5)

    output = layer(x) # (batch, channels, num_orders, length) = (1, 1, 3, 5)
    output = output.squeeze(0).squeeze(0) # (3, 5)

    print(f"Input: {x}")
    print(f"q=0: {output[0]}")
    print(f"q=1: {output[1]}")
    print(f"q=2: {output[2]}")

    # Assert q=0
    assert torch.allclose(output[0], x[0]), f"q=0 failed: {output[0]}"

    # Assert q=1
    expected_q1 = torch.tensor([1.0, 1.0, 2.0, 3.0, 4.0]) # [1-0, 2-1, 4-2, 7-4, 11-7]
    assert torch.allclose(output[1], expected_q1), f"q=1 failed: {output[1]}"

    # Assert q=2
    # D^2 x_t = x_t - 2x_{t-1} + x_{t-2}
    expected_q2 = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0])
    assert torch.allclose(output[2], expected_q2), f"q=2 failed: {output[2]}"

    print("Integer order tests passed!")

def test_gradients():
    print("Testing gradients with respect to order q...")
    layer = FractionalDerivativeLayer(num_orders=1, init_orders=[0.5])
    x = torch.randn(2, 1, 40, requires_grad=True)

    output = layer(x)
    loss = output.pow(2).sum()
    loss.backward()

    assert x.grad is not None
    assert layer.orders.grad is not None
    print(f"Order: {layer.orders.item()}, Gradient: {layer.orders.grad.item()}")
    print("Gradient tests passed!")

if __name__ == "__main__":
    test_logic()
    test_gradients()
