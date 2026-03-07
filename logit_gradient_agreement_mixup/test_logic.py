import torch
import torch.nn as nn
import torch.nn.functional as F
from logit_gradient_agreement_mixup.model import MLP
from logit_gradient_agreement_mixup.train_utils import lgam_mixup_data, get_logit_gradients

def test_lgam_logic():
    batch_size = 4
    input_size = 40
    num_classes = 10
    model = MLP(input_size=input_size, hidden_size=64, output_size=num_classes)
    x = torch.randn(batch_size, input_size)
    y = torch.randint(0, num_classes, (batch_size,))

    # 1. Test get_logit_gradients
    logit_grads = get_logit_gradients(model, x, y)
    assert logit_grads.shape == (batch_size, num_classes)

    # Check if sum of probs is 1
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        y_onehot = F.one_hot(y, num_classes=num_classes).float()
        expected_grads = probs - y_onehot
        assert torch.allclose(logit_grads, expected_grads)

    # 2. Test lgam_mixup_data
    alpha = 1.0
    gamma = 2.0
    mixed_x, y_a, y_b, lams = lgam_mixup_data(model, x, y, alpha=alpha, gamma=gamma)

    assert mixed_x.shape == x.shape
    assert y_a.shape == y.shape
    assert y_b.shape == y.shape
    assert lams.shape == (batch_size, 1)
    assert (lams >= 0).all() and (lams <= 1).all()

    print("Logic test passed!")

if __name__ == "__main__":
    test_lgam_logic()
