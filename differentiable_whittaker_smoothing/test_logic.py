import torch
import torch.nn as nn
from layer import WhittakerSmoothing, get_difference_matrix
import matplotlib.pyplot as plt
import os

def test_smoothing_effect():
    print("Testing smoothing effect...")
    n = 100
    t = torch.linspace(0, 1, n)
    clean_signal = torch.sin(2 * torch.pi * 5 * t)
    noise = torch.randn(n) * 0.2
    noisy_signal = clean_signal + noise

    # Reshape for the layer (batch, features)
    x = noisy_signal.unsqueeze(0)

    orders = [1, 2, 3]
    lambdas = [1.0, 100.0, 10000.0]

    plt.figure(figsize=(15, 10))
    plt.plot(t, noisy_signal, label="Noisy", alpha=0.3, color='gray')
    plt.plot(t, clean_signal, label="Clean", linewidth=2, color='black')

    for d in orders:
        for l in lambdas:
            layer = WhittakerSmoothing(n, order=d, initial_lambda=l, learnable_lambda=False)
            with torch.no_grad():
                smoothed = layer(x).squeeze(0)
            plt.plot(t, smoothed, label=f"Order {d}, lambda {l}")

    plt.legend()
    plt.title("Whittaker-Eilers Smoothing at Various Scales and Orders")
    os.makedirs("differentiable_whittaker_smoothing", exist_ok=True)
    plt.savefig("differentiable_whittaker_smoothing/smoothing_test.png")
    print("Saved smoothing_test.png")

def test_gradients():
    print("Testing gradients...")
    n = 20
    batch_size = 2
    x = torch.randn(batch_size, n, dtype=torch.float64, requires_grad=True)

    layer = WhittakerSmoothing(n, order=2, initial_lambda=10.0, learnable_lambda=True)
    layer = layer.to(torch.float64)

    # Check gradients with respect to input
    from torch.autograd import gradcheck

    def func(input_tensor):
        return layer(input_tensor)

    test = gradcheck(func, (x,), eps=1e-6, atol=1e-4)
    print(f"Gradcheck for input: {test}")

    # Check gradients with respect to log_lambda
    def func_lambda(log_lambda_param):
        # Temporarily replace parameter
        old_log_lambda = layer.log_lambda
        layer.log_lambda = log_lambda_param
        out = layer(x)
        layer.log_lambda = old_log_lambda
        return out

    log_lambda = nn.Parameter(torch.tensor([1.0], dtype=torch.float64, requires_grad=True))
    test_lambda = gradcheck(func_lambda, (log_lambda,), eps=1e-6, atol=1e-4)
    print(f"Gradcheck for lambda: {test_lambda}")

def test_per_channel():
    print("Testing per-channel smoothing...")
    batch_size = 2
    channels = 3
    n = 40
    x = torch.randn(batch_size, channels, n)

    layer = WhittakerSmoothing(n, order=2, initial_lambda=10.0, per_channel=True, channels=channels)
    output = layer(x)

    assert output.shape == (batch_size, channels, n)
    print(f"Per-channel output shape correct: {output.shape}")

    # Different lambdas for different channels
    with torch.no_grad():
        layer.log_lambda.data[0] = 0.0 # lambda = 1
        layer.log_lambda.data[1] = 10.0 # lambda = exp(10)
        layer.log_lambda.data[2] = 20.0 # lambda = exp(20)

    output = layer(x)
    # Channel 2 should be much smoother than channel 0
    var0 = torch.var(torch.diff(output[:, 0, :]))
    var2 = torch.var(torch.diff(output[:, 2, :]))
    print(f"Variance of diffs: Ch0={var0.item():.6f}, Ch2={var2.item():.6f}")
    assert var2 < var0
    print("Per-channel smoothing verified.")

if __name__ == "__main__":
    test_smoothing_effect()
    test_gradients()
    test_per_channel()
