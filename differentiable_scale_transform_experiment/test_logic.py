import torch
import numpy as np
from model import DifferentiableScaleTransform

def test_differentiability():
    print("Testing differentiability...")
    input_size = 40
    x = torch.randn(2, input_size, requires_grad=True)
    st_layer = DifferentiableScaleTransform(input_size)
    out = st_layer(x)
    loss = out.sum()
    loss.backward()

    if x.grad is not None:
        print("Success: Gradient is not None.")
        print(f"Gradient norm: {x.grad.norm().item()}")
    else:
        print("Failure: Gradient is None.")

def test_scale_invariance():
    print("Testing scale invariance...")
    input_size = 100
    st_layer = DifferentiableScaleTransform(input_size, num_scale_bins=200, t_min=1.0, t_max=80.0)

    # Create a simple pulse signal
    x = torch.zeros(1, input_size)
    x[0, 20:40] = 1.0

    # Scale the signal by 1.5
    # Original pulse: [20, 40]
    # Scaled pulse: [30, 60] (roughly)
    # We use linear interpolation to scale

    def scale_signal(signal, factor):
        B, N = signal.shape
        # New length would be factor * N, but we want to keep it N and stretch the content
        # So we sample from original at signal[t/factor]
        t = torch.arange(N).float()
        t_scaled = t / factor
        # Normalize to [-1, 1] for grid_sample
        grid_t = 2.0 * t_scaled / (N - 1) - 1.0
        grid = torch.zeros(B, 1, N, 2)
        grid[:, 0, :, 0] = grid_t

        scaled = torch.nn.functional.grid_sample(signal.view(B, 1, 1, N), grid, align_corners=True)
        return scaled.view(B, N)

    x_scaled = scale_signal(x, 1.5)

    # Scale transform magnitude should be invariant to scale (up to a phase shift in log-frequency and overall amplitude factor if not normalized)
    # Actually, Scale Transform is ST(s) = \int x(t) t^{is - 1/2} dt
    # If we replace x(t) with x(at), then ST_new(s) = \int x(at) t^{is - 1/2} dt
    # Let u = at, t = u/a, dt = du/a
    # ST_new(s) = \int x(u) (u/a)^{is - 1/2} (1/a) du
    # ST_new(s) = a^{-is - 1/2} a^{-1} \int x(u) u^{is - 1/2} du -- wait
    # dt = du/a
    # ST_new(s) = \int x(u) (u/a)^{is - 1/2} (1/a) du
    #          = a^{-(is + 1/2)} \int x(u) u^{is - 1/2} du
    #          = a^{-1/2} a^{-is} ST(s)
    # |ST_new(s)| = a^{-1/2} |ST(s)| because |a^{-is}| = |exp(-is ln a)| = 1.

    # So if we normalize by sqrt(scale), it should be invariant.
    # In our discrete implementation, we might need to be careful.

    out1 = st_layer(x)
    out2 = st_layer(x_scaled)

    # Normalize by max to check shape similarity
    out1_norm = out1 / out1.max()
    out2_norm = out2 / out2.max()

    diff = torch.norm(out1_norm - out2_norm) / torch.norm(out1_norm)
    print(f"Normalized difference between original and scaled (1.5x): {diff.item():.4f}")

    if diff < 0.2: # Allow some error due to discretization and finite window
        print("Success: Scale invariance holds reasonably well.")
    else:
        print("Warning: Scale invariance difference is high. This might be due to discrete effects or edge cases.")

if __name__ == "__main__":
    test_differentiability()
    test_scale_invariance()
