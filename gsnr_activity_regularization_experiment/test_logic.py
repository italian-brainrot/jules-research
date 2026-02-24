import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GWARMLP

def test_gwar_logic():
    batch_size = 4
    input_dim = 10
    hidden_dim = 20
    output_dim = 5

    model = GWARMLP(input_dim, hidden_dim, output_dim)
    x = torch.randn(batch_size, input_dim)
    y = torch.randint(0, output_dim, (batch_size,))

    # Forward pass
    logits = model(x)

    # Compute GWAR loss
    lambda_gwar = 0.1
    gwar_loss = model.compute_gwar_loss(y, lambda_gwar)

    print(f"GWAR loss: {gwar_loss.item()}")
    assert gwar_loss.item() >= 0

    # Check if we can backprop through it (even though GWAR loss is detached in compute_gwar_loss)
    # Wait, in my implementation, I used torch.no_grad() for compute_gwar_loss.
    # So the loss itself should NOT have grad.
    # But it is added to the CE loss which has grad.

    ce_loss = F.cross_entropy(logits, y)
    total_loss = ce_loss + gwar_loss
    total_loss.backward()

    # Check if gradients are populated
    for p in model.parameters():
        assert p.grad is not None

    print("Test passed!")

if __name__ == '__main__':
    test_gwar_logic()
