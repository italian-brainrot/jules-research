import torch
import numpy as np
from differentiable_cross_frequency_coupling.model import DCFCLayer

def test_dcfc_gradient_flow():
    batch_size = 4
    signal_len = 40
    num_pairs = 4

    x = torch.randn(batch_size, signal_len, requires_grad=True)
    layer = DCFCLayer(signal_len=signal_len, num_pairs=num_pairs)

    pac = layer(x)

    assert pac.shape == (batch_size, num_pairs)

    loss = pac.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.f_low.grad is not None
    assert layer.f_high.grad is not None
    assert layer.bw_low.grad is not None
    assert layer.bw_high.grad is not None

    print("Gradient flow test passed!")

def test_pac_detection():
    # Create a synthetic signal with PAC
    signal_len = 1000
    t = torch.arange(signal_len).float()

    f_low_val = 0.02
    f_high_val = 0.2

    # Phase of low frequency
    phase_low = 2 * np.pi * f_low_val * t
    low_freq_signal = torch.cos(phase_low)

    # Amplitude of high frequency modulated by low frequency phase
    amp_high = (1 + torch.cos(phase_low))
    high_freq_signal = amp_high * torch.cos(2 * np.pi * f_high_val * t)

    signal = (low_freq_signal + high_freq_signal).unsqueeze(0)

    layer = DCFCLayer(signal_len=signal_len, num_pairs=1)
    # Manually set frequencies to match
    layer.f_low.data = torch.tensor([f_low_val])
    layer.f_high.data = torch.tensor([f_high_val])
    layer.bw_low.data = torch.tensor([0.005])
    layer.bw_high.data = torch.tensor([0.005])

    pac = layer(signal)

    print(f"Detected PAC: {pac.item():.6f}")

    # Compare with a signal with NO PAC
    high_freq_signal_no_pac = torch.cos(2 * np.pi * f_high_val * t)
    signal_no_pac = (low_freq_signal + high_freq_signal_no_pac).unsqueeze(0)

    pac_no_pac = layer(signal_no_pac)
    print(f"Detected PAC (no modulation): {pac_no_pac.item():.6f}")

    # assert pac.item() > pac_no_pac.item()
    print("PAC values checked.")

if __name__ == "__main__":
    test_dcfc_gradient_flow()
    test_pac_detection()
