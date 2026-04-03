import torch
from model import DifferentiableTV1D, TVDenoisingMLP, BaselineMLP

def test_tv_layer_shapes():
    batch_size = 10
    seq_len = 40
    x = torch.randn(batch_size, seq_len)
    tv = DifferentiableTV1D(n_iters=5)
    out = tv(x)
    assert out.shape == x.shape

def test_tv_layer_grad():
    seq_len = 20
    x = torch.randn(2, seq_len, requires_grad=True)
    tv = DifferentiableTV1D(n_iters=10, learnable=True)
    out = tv(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert tv.log_lambda.grad is not None

def test_tv_denoising_effect():
    # Constant signal with noise
    x = torch.zeros(1, 100)
    x_noisy = x + 0.1 * torch.randn(1, 100)
    tv = DifferentiableTV1D(n_iters=50, initial_lambda=0.5)
    x_denoised = tv(x_noisy)

    # TV should reduce total variation
    def tv_norm(signal):
        return torch.abs(signal[:, 1:] - signal[:, :-1]).sum()

    orig_tv = tv_norm(x_noisy)
    new_tv = tv_norm(x_denoised)
    print(f"Original TV: {orig_tv.item():.4f}, Denoised TV: {new_tv.item():.4f}")
    assert new_tv < orig_tv

if __name__ == "__main__":
    test_tv_layer_shapes()
    test_tv_layer_grad()
    test_tv_denoising_effect()
    print("All logic tests passed!")
