import torch
import torch.nn as nn
import pytest
import sys
import os

# Add the parent directory to the path to allow for package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from eigen_normalized_gradient_descent.optimizer import ENGD

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

def test_engd_optimizer_step():
    """Tests that the ENGD optimizer can perform a step without crashing."""
    model = SimpleModel()
    # Use a copy of initial params to check if they change after the step
    initial_params = [p.clone() for p in model.parameters()]

    optimizer = ENGD(model.parameters(), lr=0.01)

    # Dummy data and loss
    input_tensor = torch.randn(5, 10)
    target = torch.randn(5, 1)
    criterion = nn.MSELoss()

    def closure():
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward(create_graph=True)
        return loss

    try:
        optimizer.step(closure)
    except Exception as e:
        pytest.fail(f"ENGD optimizer step raised an exception: {e}")

    # Check that parameters have been updated
    for i, p in enumerate(model.parameters()):
        assert not torch.equal(p, initial_params[i]), "Model parameters were not updated after optimizer step."
