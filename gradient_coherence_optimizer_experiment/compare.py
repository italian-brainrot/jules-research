import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import get_dataset, get_dataset_args
import optuna

from optimizer import GradientCoherence

def get_model():
    return nn.Sequential(
        nn.Linear(40, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

def train(optimizer_class, lr, train_loader, test_loader, num_epochs=10):
    model = get_model()
    optimizer = optimizer_class(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total
    return val_loss / len(test_loader)

def objective(trial, train_loader, test_loader):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer_name = trial.study.user_attrs['optimizer_name']

    if optimizer_name == 'Adam':
        optimizer_class = optim.Adam
    elif optimizer_name == 'GradientCoherence':
        optimizer_class = GradientCoherence
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return train(optimizer_class, lr, train_loader, test_loader)

if __name__ == '__main__':
    args = get_dataset_args()
    data = get_dataset(args)
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'], dtype=torch.long)
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'], dtype=torch.long)

    train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    # Study for Adam
    study_adam = optuna.create_study(direction='minimize')
    study_adam.set_user_attr('optimizer_name', 'Adam')
    study_adam.optimize(lambda trial: objective(trial, train_loader, test_loader), n_trials=50)

    # Study for Gradient Coherence
    study_gc = optuna.create_study(direction='minimize')
    study_gc.set_user_attr('optimizer_name', 'GradientCoherence')
    study_gc.optimize(lambda trial: objective(trial, train_loader, test_loader), n_trials=50)

    print("--- Results ---")
    print(f"Adam best validation loss: {study_adam.best_value:.4f}")
    print(f"Adam best learning rate: {study_adam.best_params['lr']:.6f}")
    print(f"Gradient Coherence best validation loss: {study_gc.best_value:.4f}")
    print(f"Gradient Coherence best learning rate: {study_gc.best_params['lr']:.6f}")

    if study_gc.best_value < study_adam.best_value:
        print("\nConclusion: Gradient Coherence optimizer performed better.")
    else:
        print("\nConclusion: Adam optimizer performed better or comparably.")
