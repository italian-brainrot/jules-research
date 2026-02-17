import torch
import torch.nn as nn
from adaptive_gradient_alignment_warmup.optimizer import AGAWOptimizer

def test_agaw_optimizer_lr_increase():
    # Simple linear model
    model = nn.Linear(10, 1)
    target_lr = 0.1
    initial_lr = 1e-7
    base_opt = torch.optim.SGD(model.parameters(), lr=target_lr)
    optimizer = AGAWOptimizer(
        base_opt,
        target_lr=target_lr,
        initial_lr=initial_lr,
        warmup_steps_nominal=10
    )

    # First step
    optimizer.zero_grad()
    # Mock some gradients
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()

    # After 1st step, lr should still be initial_lr because prev_grad was None
    assert abs(optimizer.current_lr - initial_lr) < 1e-10

    # Second step with SAME gradient direction
    optimizer.zero_grad()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()

    # After 2nd step with perfectly aligned gradients, lr should increase
    assert optimizer.current_lr > initial_lr

    # Check that it reaches target_lr eventually
    for _ in range(20):
        optimizer.zero_grad()
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        optimizer.step()

    assert abs(optimizer.current_lr - target_lr) < 1e-5

def test_agaw_optimizer_misalignment():
    model = nn.Linear(10, 1)
    target_lr = 0.1
    initial_lr = 1e-7
    base_opt = torch.optim.SGD(model.parameters(), lr=target_lr)
    optimizer = AGAWOptimizer(
        base_opt,
        target_lr=target_lr,
        initial_lr=initial_lr,
        warmup_steps_nominal=10
    )

    # 1st step
    optimizer.zero_grad()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()

    lr_after_1 = optimizer.current_lr

    # 2nd step with OPPOSITE gradient direction
    optimizer.zero_grad()
    for p in model.parameters():
        p.grad = -torch.ones_like(p)
    optimizer.step()

    # lr should NOT increase because cos_sim is -1
    assert abs(optimizer.current_lr - lr_after_1) < 1e-10

if __name__ == "__main__":
    test_agaw_optimizer_lr_increase()
    test_agaw_optimizer_misalignment()
    print("Tests passed!")
