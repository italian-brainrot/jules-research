import torch
from model import DAWTLayer

def test_reconstruction():
    print("Testing reconstruction (zero threshold)...")
    input_len = 40
    levels = 3
    layer = DAWTLayer(levels=levels, input_len=input_len)

    # Set thresholds to zero
    layer.thresholds.data.fill_(0.0)

    x = torch.randn(5, input_len)
    y = layer(x)

    diff = torch.abs(x - y).max().item()
    print(f"Max reconstruction error: {diff:.2e}")
    assert diff < 1e-6, "Reconstruction error too high!"
    print("Reconstruction test passed.")

def test_differentiability():
    print("\nTesting differentiability...")
    input_len = 40
    levels = 2
    layer = DAWTLayer(levels=levels, input_len=input_len)

    x = torch.randn(5, input_len, requires_grad=True)
    y = layer(x)

    loss = y.pow(2).sum()
    loss.backward()

    assert x.grad is not None, "Gradients not flowing to input!"
    assert layer.thresholds.grad is not None, "Gradients not flowing to thresholds!"

    print(f"Thresholds grad: {layer.thresholds.grad}")
    print("Differentiability test passed.")

if __name__ == "__main__":
    try:
        test_reconstruction()
        test_differentiability()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        exit(1)
