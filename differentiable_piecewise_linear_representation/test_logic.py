import torch
from model import DPLRLayer

def test_dplr_gradients():
    print("Testing DPLR gradients...")
    batch_size = 4
    input_dim = 40
    num_segments = 4

    x = torch.randn(batch_size, input_dim, requires_grad=True)
    layer = DPLRLayer(input_dim, num_segments)

    out = layer(x)
    loss = out.sum()
    loss.backward()

    if x.grad is not None:
        print("Gradient with respect to input: SUCCESS")
    else:
        print("Gradient with respect to input: FAILED")

    if layer.breakpoints.grad is not None:
        print("Gradient with respect to breakpoints: SUCCESS")
    else:
        print("Gradient with respect to breakpoints: FAILED")

def test_dplr_fitting():
    print("\nTesting DPLR fitting accuracy...")
    input_dim = 40
    num_segments = 2

    # Create a piecewise linear signal
    # Segment 1 (0-20): y = 2t + 5
    # Segment 2 (20-40): y = -t + 60
    t = torch.arange(input_dim, dtype=torch.float32)
    x = torch.zeros(1, input_dim)
    x[0, :20] = 2 * t[:20] + 5
    x[0, 20:] = -1 * t[20:] + 65

    layer = DPLRLayer(input_dim, num_segments, temperature=100.0)
    # Manually set breakpoint near 20
    with torch.no_grad():
        layer.breakpoints[0] = 20.0

    out = layer(x)
    # Features are (a, c, mse) for each segment
    # Segment 1: a=2, c=5, mse=0
    # Segment 2: a=-1, c=65, mse=0

    print(f"Segment 1 features: {out[0, :3].detach().numpy()}")
    print(f"Segment 2 features: {out[0, 3:6].detach().numpy()}")

    assert torch.allclose(out[0, 0], torch.tensor(2.0), atol=1e-1)
    assert torch.allclose(out[0, 1], torch.tensor(5.0), atol=1e-1)
    assert torch.allclose(out[0, 3], torch.tensor(-1.0), atol=1e-1)
    assert torch.allclose(out[0, 4], torch.tensor(65.0), atol=1e-1)
    print("Fitting accuracy: SUCCESS")

if __name__ == "__main__":
    test_dplr_gradients()
    test_dplr_fitting()
