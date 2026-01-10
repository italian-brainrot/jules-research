
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import os
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

def rms_norm(tensor):
    return torch.sqrt(torch.mean(tensor**2))

def rmsln_regularizer(model, target_norm=1.0):
    penalty = 0.0
    for param in model.parameters():
        if param.dim() > 1:  # Only apply to weight matrices
            penalty += (rms_norm(param) - target_norm)**2
    return penalty

class MLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_and_evaluate(model, optimizer, criterion, dl_train, dl_test, num_epochs, reg_fn=None, reg_lambda=0.0):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for inputs, targets in dl_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if reg_fn:
                loss += reg_lambda * reg_fn(model)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        train_losses.append(epoch_train_loss / len(dl_train))

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for inputs, targets in dl_test:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()
        val_losses.append(epoch_val_loss / len(dl_test))
    return train_losses, val_losses

def objective_rmsln(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    reg_lambda = trial.suggest_float('reg_lambda', 1e-4, 1e2, log=True)

    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    _, val_losses = train_and_evaluate(model, optimizer, criterion, dl_train, dl_test, num_epochs=10, reg_fn=rmsln_regularizer, reg_lambda=reg_lambda)

    return min(val_losses)

def objective_baseline(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    _, val_losses = train_and_evaluate(model, optimizer, criterion, dl_train, dl_test, num_epochs=10)

    return min(val_losses)

if __name__ == '__main__':
    # Load data
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'], dtype=torch.long)
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'], dtype=torch.long)
    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=128)

    # Hyperparameter tuning for RMSLN
    study_rmsln = optuna.create_study(direction='minimize')
    study_rmsln.optimize(objective_rmsln, n_trials=25)
    best_params_rmsln = study_rmsln.best_params
    print(f"Best params for RMSLN: {best_params_rmsln}")

    # Hyperparameter tuning for baseline
    study_baseline = optuna.create_study(direction='minimize')
    study_baseline.optimize(objective_baseline, n_trials=25)
    best_params_baseline = study_baseline.best_params
    print(f"Best params for baseline: {best_params_baseline}")

    # Final training with best hyperparameters
    num_epochs_final = 100

    # Train RMSLN model
    model_rmsln = MLP()
    optimizer_rmsln = optim.Adam(model_rmsln.parameters(), lr=best_params_rmsln['lr'])
    criterion = nn.CrossEntropyLoss()
    _, val_losses_rmsln = train_and_evaluate(model_rmsln, optimizer_rmsln, criterion, dl_train, dl_test, num_epochs_final, reg_fn=rmsln_regularizer, reg_lambda=best_params_rmsln['reg_lambda'])

    # Train baseline model
    model_baseline = MLP()
    optimizer_baseline = optim.Adam(model_baseline.parameters(), lr=best_params_baseline['lr'])
    _, val_losses_baseline = train_and_evaluate(model_baseline, optimizer_baseline, criterion, dl_train, dl_test, num_epochs_final)

    print(f"Final validation loss (RMSLN): {val_losses_rmsln[-1]}")
    print(f"Final validation loss (Baseline): {val_losses_baseline[-1]}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(val_losses_rmsln, label='Adam + RMSLN')
    plt.plot(val_losses_baseline, label='Adam (Baseline)')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'rmsln_comparison.png'))
    plt.show()
