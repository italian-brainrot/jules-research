import torch
import torch.nn as nn
from optimizer import SortedGradientOptimizer

def test_gradient_sorting():
    """
    Tests if the SortedGradientOptimizer correctly sorts gradients before the base optimizer's step.
    """
    # 1. Setup a simple model and data
    model = nn.Linear(10, 1)
    # Ensure weights are not uniform for a non-uniform gradient
    torch.nn.init.normal_(model.weight)
    inputs = torch.randn(1, 10)
    targets = torch.randn(1, 1)

    # 2. Setup the optimizer
    # Use SGD for simplicity, as we only care about the gradient manipulation
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer = SortedGradientOptimizer(base_optimizer)

    # 3. Generate gradients
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = nn.MSELoss()(outputs, targets)
    loss.backward()

    # 4. Store the original, unsorted gradient
    original_grad = model.weight.grad.clone().flatten()

    # 5. Perform the optimization step
    # This is where the sorting happens
    optimizer.step()

    # 6. Verification
    # After the step, the gradient on the parameter should have been modified in-place
    # to its sorted version by our wrapper before the base optimizer used it.
    grad_after_step = model.weight.grad.clone().flatten()

    # The gradient that was actually used by SGD should be the sorted version
    # of the original gradient. We can check this by sorting the original
    # gradient ourselves and comparing.
    expected_sorted_grad = torch.sort(original_grad)[0]

    # Assert that the gradient was indeed sorted
    assert not torch.equal(original_grad, grad_after_step), "Gradient was not modified by the optimizer."
    assert torch.allclose(expected_sorted_grad, grad_after_step), "Gradient was not sorted correctly."

if __name__ == "__main__":
    test_gradient_sorting()
    print("Test passed!")
