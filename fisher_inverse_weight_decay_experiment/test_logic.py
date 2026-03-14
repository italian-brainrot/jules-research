import torch
import torch.nn as nn
import torch.optim as optim
from optimizer import FIWDAdamW

def test_fiwd_logic():
    # Simple model
    model = nn.Sequential(nn.Linear(10, 10))
    p = model[0].weight

    # Initialize with some values
    with torch.no_grad():
        p.fill_(1.0)

    # Optimizer with FIWD
    # Use high weight_decay and gamma to see the effect clearly
    # Small tau so that v_hat / tau is significant
    optimizer = FIWDAdamW(model.parameters(), lr=0.1, weight_decay=1.0, gamma=1.0, tau=1.0)

    # Step 1: Zero gradient
    optimizer.zero_grad()
    p.grad = torch.zeros_like(p)
    optimizer.step()

    # Since grad is 0, v_hat is 0.
    # Scaling = (1 + 0/1)^(-1) = 1.0
    # Expected weight after decay: 1.0 * (1 - 0.1 * 1.0 * 1.0) = 0.9
    print(f"Weight after step 1 (zero grad): {p[0, 0].item():.4f}")
    assert torch.allclose(p[0, 0], torch.tensor(0.9), atol=1e-4)

    # Reset weight
    with torch.no_grad():
        p.fill_(1.0)

    # Step 2: Large gradient
    optimizer.zero_grad()
    # Mock some history in v_hat. FIWDAdamW uses exp_avg_sq.
    # In one step with grad=10, exp_avg_sq will be (1-beta2)*100 = 0.001 * 100 = 0.1
    p.grad = torch.ones_like(p) * 10.0
    optimizer.step()

    # v_hat = 0.1 / (1 - 0.999) = 100.0 (bias corrected)
    # scaling = (1 + 100/1)^(-1) = 1/101 approx 0.0099
    # Expected weight decay much smaller than before.
    # p_new = p - lr * update - lr * wd * scaling * p
    # The update part will also change p, but we care about the weight decay part.
    # Let's check if the weight is > 0.9 (meaning less decay than the zero-grad case)
    # Actually, the update will be lr * m_hat / sqrt(v_hat) = 0.1 * 10 / 10 = 0.1
    # So p_new approx 1.0 - 0.1 - 0.1 * 1.0 * 0.0099 * 1.0 = 0.89901
    # If it was standard AdamW, p_new = 1.0 - 0.1 - 0.1 * 1.0 * 1.0 * 1.0 = 0.8
    print(f"Weight after step 2 (large grad): {p[0, 0].item():.4f}")
    assert p[0, 0] > 0.85, f"Weight decay should be reduced for large gradients. Got {p[0, 0].item()}"

if __name__ == "__main__":
    try:
        test_fiwd_logic()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        exit(1)
