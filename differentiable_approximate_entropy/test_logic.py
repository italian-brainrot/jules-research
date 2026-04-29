import torch
from model import DSampEnLayer

def test_dsampen_differentiability():
    batch_size = 4
    seq_len = 40
    x = torch.randn(batch_size, seq_len, requires_grad=True)
    layer = DSampEnLayer(m=2, r=0.2, gamma=10.0, learnable=True)

    output = layer(x)
    assert output.shape == (batch_size, 1)

    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.r.grad is not None
    assert layer.gamma.grad is not None
    print("Differentiability test passed!")

def test_dsampen_values():
    # Regular signal (sine wave)
    t = torch.linspace(0, 10, 40)
    x_reg = torch.sin(t).unsqueeze(0)

    # Irregular signal (noise)
    x_irreg = torch.randn(1, 40)

    layer = DSampEnLayer(m=2, r=0.2, gamma=10.0, learnable=False)

    en_reg = layer(x_reg).item()
    en_irreg = layer(x_irreg).item()

    print(f"Entropy regular: {en_reg:.4f}")
    print(f"Entropy irregular: {en_irreg:.4f}")

    # Noise should generally have higher entropy than a sine wave
    assert en_irreg > en_reg
    print("Entropy values test passed!")

if __name__ == "__main__":
    test_dsampen_differentiability()
    test_dsampen_values()
