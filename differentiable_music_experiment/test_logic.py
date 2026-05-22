import torch
import numpy as np
from music import DifferentiableMUSIC
import matplotlib.pyplot as plt

def test_music_frequency_detection():
    print("Testing frequency detection...")
    seq_len = 100
    window_size = 20
    num_freqs = 128

    # Create a sine wave with a specific frequency
    t = torch.linspace(0, 1, seq_len)
    freq_idx = 30
    true_freq = np.pi * freq_idx / (num_freqs - 1)

    fs = 100
    f = 10
    n = torch.arange(seq_len).float()
    x = torch.sin(2 * np.pi * f / fs * n).unsqueeze(0)
    expected_freq_rad = 2 * np.pi * f / fs

    music_layer = DifferentiableMUSIC(window_size=window_size, num_freqs=num_freqs)
    music_layer.log_threshold.data = torch.tensor(0.0)
    music_layer.log_beta.data = torch.tensor(10.0)

    output = music_layer(x)

    # Find peak frequency
    peak_idx = torch.argmax(output[0])
    freqs = torch.linspace(0, np.pi, num_freqs)
    detected_freq = freqs[peak_idx].item()

    print(f"Expected frequency (rad/sample): {expected_freq_rad:.4f}")
    print(f"Detected frequency (rad/sample): {detected_freq:.4f}")

    assert abs(detected_freq - expected_freq_rad) < 0.2, f"Frequency detection failed: {detected_freq} vs {expected_freq_rad}"
    print("Frequency detection test passed!")

def test_gradients():
    print("Testing gradients...")
    seq_len = 40
    window_size = 10
    num_freqs = 32

    x = torch.randn(2, seq_len, requires_grad=True)
    music_layer = DifferentiableMUSIC(window_size=window_size, num_freqs=num_freqs)

    output = music_layer(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))
    print(f"Gradient norm: {x.grad.norm().item()}")

    # Check gradients for parameters
    assert music_layer.log_threshold.grad is not None
    assert music_layer.log_beta.grad is not None
    print("Gradient test passed!")

if __name__ == "__main__":
    test_music_frequency_detection()
    test_gradients()
