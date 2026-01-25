import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import os

def get_model():
    """Initializes and returns a simple MLP model."""
    return nn.Sequential(
        nn.Linear(40, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

def get_data():
    """Loads and returns the mnist1d dataset."""
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    try:
        data = make_dataset(defaults)
    except FileNotFoundError:
        # Handle case where data is not downloaded
        defaults.download = True
        data = make_dataset(defaults)

    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.int64)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.int64)

    train_loader = TensorDataLoader((X_train, y_train), batch_size=256, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=256, shuffle=False)
    return train_loader, test_loader

def evaluate(model, loader):
    """Evaluates the model on the given data loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total

def train_polynomial(model, train_loader, test_loader, lr, reg_strength, epochs=20):
    """Trains the model using Polynomial Regularization."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Stores W_{t-2}
    weights_two_steps_ago = [p.clone().detach() for p in model.parameters()]

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            # Store W_{t-1} before the update, to be used in the next iteration
            weights_one_step_ago = [p.clone().detach() for p in model.parameters()]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Regularization term is || W_{t-1} - W_{t-2} ||^2
            # Here, `p` is W_{t-1} and `p_old` is W_{t-2}
            reg_loss = 0
            for p, p_old in zip(model.parameters(), weights_two_steps_ago):
                reg_loss += torch.sum((p - p_old) ** 2)

            total_loss = loss + reg_strength * reg_loss
            total_loss.backward()
            optimizer.step() # model parameters become W_t

            # Update the history for the next iteration
            weights_two_steps_ago = weights_one_step_ago

    return evaluate(model, test_loader)

def train_adam(model, train_loader, test_loader, lr, epochs=20):
    """Trains the model using the Adam optimizer as a baseline."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    return evaluate(model, test_loader)
