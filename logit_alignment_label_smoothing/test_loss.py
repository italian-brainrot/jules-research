import torch
import torch.nn.functional as F
from logit_alignment_label_smoothing.loss import LGALSLoss

def test_lgals_loss_basic():
    B, C = 4, 3
    logits = torch.randn(B, C)
    targets = torch.tensor([0, 1, 2, 0])

    criterion = LGALSLoss(epsilon_max=0.1, gamma=1.0)
    loss = criterion(logits, targets)

    assert loss.shape == torch.Size([])
    assert not torch.isnan(loss)
    print("test_lgals_loss_basic passed")

def test_lgals_alignment():
    # If all logit-gradients are the same, s should be 1, epsilon should be 0
    C = 3
    # Same logits and same targets -> same logit gradients
    logits = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    targets = torch.tensor([0, 0])

    criterion = LGALSLoss(epsilon_max=0.1, gamma=1.0)

    # We need to compute s manually inside or check if loss matches CE
    probs = F.softmax(logits, dim=1)
    y_oh = F.one_hot(targets, num_classes=C).float()
    g = probs - y_oh
    g_mean = g.mean(dim=0, keepdim=True)
    s = F.cosine_similarity(g, g_mean, dim=1)

    assert torch.allclose(s, torch.ones_like(s))

    loss = criterion(logits, targets)
    log_probs = F.log_softmax(logits, dim=1)
    ce_loss = -log_probs.gather(dim=1, index=targets.unsqueeze(1)).mean()

    # Since s=1, epsilon=0, loss should be CE
    assert torch.allclose(loss, ce_loss)
    print("test_lgals_alignment (s=1) passed")

def test_lgals_anti_alignment():
    # If gradients are opposite
    C = 2
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    targets = torch.tensor([0, 1])
    # p1 = [1, 0], y1 = [1, 0] -> g1 = [0, 0]
    # This is not a good example because g will be zero.

    # Try different logits
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    targets = torch.tensor([1, 0])
    # p1 = [0.88, 0.12], y1 = [0, 1] -> g1 = [0.88, -0.88]
    # p2 = [0.12, 0.88], y2 = [1, 0] -> g2 = [-0.88, 0.88]
    # mean g = [0, 0]
    # s will be undefined or depend on epsilon in code

    criterion = LGALSLoss(epsilon_max=0.1, gamma=1.0)
    loss = criterion(logits, targets)
    assert not torch.isnan(loss)
    print("test_lgals_anti_alignment passed")

if __name__ == "__main__":
    test_lgals_loss_basic()
    test_lgals_alignment()
    test_lgals_anti_alignment()
