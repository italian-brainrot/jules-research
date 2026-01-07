import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import get_dataset, get_dataset_args
import matplotlib.pyplot as plt
import optuna
import copy

from optimizer import HessianLRAdaptation

# Define the model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(40, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.layers(x)

# Training and validation functions
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

# Optuna objective function
def objective(trial, optimizer_name, train_loader, val_loader, initial_model_state):
    model = MLP()
    model.load_state_dict(initial_model_state)
    criterion = nn.CrossEntropyLoss()

    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'HessianLRAdaptation':
        base_optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer = HessianLRAdaptation(base_optimizer)
    else:
        raise ValueError("Unknown optimizer")

    for epoch in range(10):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)

    return val_loss

def main():
    # Load data
    args = get_dataset_args()
    data = get_dataset(args)
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'])
    X_val, y_val = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'])

    train_loader = TensorDataLoader((X_train, y_train), batch_size=32, shuffle=True)
    val_loader = TensorDataLoader((X_val, y_val), batch_size=32)

    # Store initial model state for fair comparison
    initial_model = MLP()
    initial_model_state = copy.deepcopy(initial_model.state_dict())

    # Tune hyperparameters
    study_adam = optuna.create_study(direction='minimize')
    study_adam.optimize(lambda trial: objective(trial, 'Adam', train_loader, val_loader, initial_model_state), n_trials=20)

    study_hessian = optuna.create_study(direction='minimize')
    study_hessian.optimize(lambda trial: objective(trial, 'HessianLRAdaptation', train_loader, val_loader, initial_model_state), n_trials=20)

    best_lr_adam = study_adam.best_trial.params['lr']
    best_lr_hessian = study_hessian.best_trial.params['lr']

    print(f"Best LR for Adam: {best_lr_adam}")
    print(f"Best LR for HessianLRAdaptation: {best_lr_hessian}")

    # Train final models with best hyperparameters
    model_adam = MLP()
    model_adam.load_state_dict(initial_model_state)
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=best_lr_adam)

    model_hessian = MLP()
    model_hessian.load_state_dict(initial_model_state)
    base_optimizer_hessian = optim.Adam(model_hessian.parameters(), lr=best_lr_hessian)
    optimizer_hessian = HessianLRAdaptation(base_optimizer_hessian)

    criterion = nn.CrossEntropyLoss()

    adam_val_losses = []
    hessian_val_losses = []

    for epoch in range(20):
        train_epoch(model_adam, train_loader, optimizer_adam, criterion)
        val_loss_adam, _ = validate(model_adam, val_loader, criterion)
        adam_val_losses.append(val_loss_adam)

        train_epoch(model_hessian, train_loader, optimizer_hessian, criterion)
        val_loss_hessian, _ = validate(model_hessian, val_loader, criterion)
        hessian_val_losses.append(val_loss_hessian)

        print(f"Epoch {epoch+1}: Adam Val Loss: {val_loss_adam:.4f}, Hessian Val Loss: {val_loss_hessian:.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(adam_val_losses, label='Adam')
    plt.plot(hessian_val_losses, label='HessianLRAdaptation')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('hessian_lr_adaptation_experiment/comparison.png')
    plt.show()

if __name__ == '__main__':
    main()
