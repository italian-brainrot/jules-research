import torch
import torch.nn as nn
import copy
from gns_adam_experiment.optimizer import GNSAdam

def test_gns_adam_step():
    """
    Tests if the GNSAdam optimizer takes a step and updates parameters.
    """
    # Arrange: Create a simple model, optimizer, and dummy data
    model = nn.Linear(10, 1)

    # Store a deep copy of the initial parameters for later comparison
    initial_params = copy.deepcopy(list(model.parameters()))

    optimizer = GNSAdam(model.parameters(), base_optimizer=torch.optim.Adam, lr=0.001)
    criterion = nn.MSELoss()

    inputs = torch.randn(5, 10)
    targets = torch.randn(5, 1)

    # Act: Perform a single optimization step
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    # Assert: Check that the parameters have changed from their initial values
    params_have_changed = False
    for i, param in enumerate(model.parameters()):
        if not torch.equal(initial_params[i], param):
            params_have_changed = True
            break

    assert params_have_changed, "The optimizer step did not change the model parameters."

def test_gns_adam_zero_grad():
    """
    Tests if the optimizer's zero_grad method works correctly.
    """
    # Arrange
    model = nn.Linear(10, 1)
    optimizer = GNSAdam(model.parameters(), base_optimizer=torch.optim.Adam, lr=0.001)
    inputs = torch.randn(5, 10)
    targets = torch.randn(5, 1)
    criterion = nn.MSELoss()

    # Act: Compute gradients
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    # Assert that gradients exist before zero_grad
    grads_exist_before = any(p.grad is not None for p in model.parameters())
    assert grads_exist_before, "Gradients should exist after backward pass."

    # Act: Clear gradients
    optimizer.zero_grad()

    # Assert that gradients are gone after zero_grad
    grads_exist_after = any(p.grad is not None for p in model.parameters())
    assert not grads_exist_after, "Gradients should not exist after zero_grad."
