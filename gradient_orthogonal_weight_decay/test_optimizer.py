import torch
from gradient_orthogonal_weight_decay.optimizer import GOWD

def test_gowd_orthogonal():
    # p is [1, 1], grad is [1, 0]
    # projection of p onto d is [1, 0]
    # orthogonal part is [0, 1]
    # weight decay should only affect the second component
    p = torch.tensor([1.0, 1.0], requires_grad=True)
    p.grad = torch.tensor([1.0, 0.0])

    # lr=0.1, wd=0.5, betas=(0.0, 0.0)
    # exp_avg = grad = [1, 0]
    # exp_avg_sq = grad^2 = [1, 0]
    # denom = sqrt(exp_avg_sq) + eps = [1, 1e-8]
    # update_dir = exp_avg / denom = [1, 0]
    # alpha = dot(p, update_dir) / dot(update_dir, update_dir) = 1.0 / 1.0 = 1.0
    # weight decay step:
    # p = p * (1 - lr * wd) + lr * wd * alpha * update_dir
    # p = [1, 1] * 0.95 + 0.05 * [1, 0] = [1.0, 0.95]
    # Adam update step:
    # p = p - lr * update_dir = [1.0, 0.95] - [0.1, 0] = [0.9, 0.95]

    opt = GOWD([p], lr=0.1, weight_decay=0.5, betas=(0.0, 0.999), eps=1e-8)
    # Note: I need beta1=0 to make exp_avg = grad immediately
    # But beta2 needs to be something or I can set it to 0 too for simplicity
    opt.param_groups[0]['betas'] = (0.0, 0.0)

    opt.step()

    print(f"p after step: {p.detach()}")
    # Check orthogonal part (component 1)
    assert torch.allclose(p[1], torch.tensor(0.95)), f"Expected 0.95, got {p[1]}"
    # Check aligned part (component 0)
    assert torch.allclose(p[0], torch.tensor(0.9)), f"Expected 0.9, got {p[0]}"

if __name__ == "__main__":
    test_gowd_orthogonal()
    print("Test passed!")
