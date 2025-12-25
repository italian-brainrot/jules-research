import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import optuna
import mnist1d
from optimizer import ENGD
import os
import numpy as np
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --- Configuration ---
DEVICE = torch.device("cpu")
EPOCHS = 10
N_TRIALS_OPTUNA = 20
N_SAMPLES_DATA = 1000  # Using a subset of data for faster tuning

# --- Data Loading ---
args = mnist1d.data.get_dataset_args()
data = mnist1d.data.get_dataset(args, path='./mnist1d_data.pkl', download=True)
X_train = torch.from_numpy(data['x']).float()[:N_SAMPLES_DATA]
y_train = torch.from_numpy(data['y']).long()[:N_SAMPLES_DATA]
X_val = torch.from_numpy(data['x_test']).float()
y_val = torch.from_numpy(data['y_test']).long()

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val, y_val), batch_size=128, shuffle=False)


# --- Model Definition ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(X_train.shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layers(x)

# --- Training & Validation ---
criterion = nn.CrossEntropyLoss()

def train_and_validate(optimizer_name, lr):
    model = SimpleMLP().to(DEVICE)
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = ENGD(model.parameters(), lr=lr)

    val_losses = []
    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            def closure():
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward(create_graph=True) # Important for ENGD
                return loss

            if isinstance(optimizer, ENGD):
                optimizer.step(closure)
            else:
                loss = closure()
                optimizer.step()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

    return val_losses

# --- Optuna Objective ---
def objective(trial, optimizer_name):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    val_losses = train_and_validate(optimizer_name, lr)
    return min(val_losses)

# --- Main Execution ---
if __name__ == '__main__':
    # Tune Adam
    study_adam = optuna.create_study(direction='minimize')
    study_adam.optimize(lambda trial: objective(trial, 'Adam'), n_trials=N_TRIALS_OPTUNA)
    best_lr_adam = study_adam.best_params['lr']
    print(f"Best LR for Adam: {best_lr_adam}")

    # Tune ENGD
    study_engd = optuna.create_study(direction='minimize')
    study_engd.optimize(lambda trial: objective(trial, 'ENGD'), n_trials=N_TRIALS_OPTUNA)
    best_lr_engd = study_engd.best_params['lr']
    print(f"Best LR for ENGD: {best_lr_engd}")

    # Final comparison run
    print("Running final comparison...")
    adam_losses = train_and_validate('Adam', best_lr_adam)
    engd_losses = train_and_validate('ENGD', best_lr_engd)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(adam_losses, label=f'Adam (Best LR: {best_lr_adam:.4f})')
    plt.plot(engd_losses, label=f'ENGD (Best LR: {best_lr_engd:.4f})')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Adam vs. ENGD Optimizer Performance')
    plt.legend()
    plt.grid(True)

    # Save the plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'comparison.png'))
    print(f"Plot saved to {os.path.join(script_dir, 'comparison.png')}")
