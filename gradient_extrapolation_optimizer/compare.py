import torch
import torch.nn as nn
import torch.optim as optim
import mnist1d
from light_dataloader import TensorDataLoader as DataLoader
import optuna
import matplotlib.pyplot as plt
import numpy as np
import os

from optimizer import GPE

# --- Model Definition ---
class MLP(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, output_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# --- Data Loading ---
def get_data():
    args = mnist1d.data.get_dataset_args()
    data = mnist1d.data.get_dataset(args)
    # Convert to float32
    data['x'] = data['x'].astype(np.float32)
    data['x_test'] = data['x_test'].astype(np.float32)
    data['y'] = data['y'].astype(np.int64)
    data['y_test'] = data['y_test'].astype(np.int64)

    # Create TensorDataLoaders
    train_loader = DataLoader(
        (torch.from_numpy(data['x']), torch.from_numpy(data['y'])),
        batch_size=128,
        shuffle=True
    )
    val_loader = DataLoader(
        (torch.from_numpy(data['x_test']), torch.from_numpy(data['y_test'])),
        batch_size=128,
        shuffle=False
    )
    return train_loader, val_loader

# --- Training and Evaluation ---
def train_model(model, optimizer, train_loader, criterion, device):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# --- Optuna Objective ---
def objective(trial, optimizer_name, device, train_loader, val_loader):
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()

    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    if optimizer_name == 'GPE':
        history_size = trial.suggest_int('history_size', 5, 15)
        # Ensure degree is valid
        max_degree = history_size - 1
        degree = trial.suggest_int('degree', 1, max_degree)
        base_optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer = GPE(base_optimizer, history_size=history_size, degree=degree)
    else: # Adam
        optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 15 # Reduced epochs for faster tuning
    for epoch in range(epochs):
        train_loss = train_model(model, optimizer, train_loader, criterion, device)
        val_loss = evaluate_model(model, val_loader, criterion, device)

    trial.report(val_loss, epoch)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return val_loss

# --- Main Comparison ---
def main():
    device = torch.device("cpu")
    train_loader, val_loader = get_data()

    # --- Tune GPE(Adam) ---
    study_gpe = optuna.create_study(direction='minimize')
    study_gpe.optimize(lambda t: objective(t, 'GPE', device, train_loader, val_loader), n_trials=30)
    best_params_gpe = study_gpe.best_params
    print(f"Best GPE params: {best_params_gpe}")

    # --- Tune Adam ---
    study_adam = optuna.create_study(direction='minimize')
    study_adam.optimize(lambda t: objective(t, 'Adam', device, train_loader, val_loader), n_trials=30)
    best_params_adam = study_adam.best_params
    print(f"Best Adam params: {best_params_adam}")

    # --- Run Final Comparison ---
    epochs = 50

    # GPE Model
    model_gpe = MLP().to(device)
    base_optimizer_gpe = optim.Adam(model_gpe.parameters(), lr=best_params_gpe['lr'])
    optimizer_gpe = GPE(base_optimizer_gpe,
                        history_size=best_params_gpe['history_size'],
                        degree=best_params_gpe['degree'])

    # Adam Model
    model_adam = MLP().to(device)
    model_adam.load_state_dict(model_gpe.state_dict()) # Use same initial weights
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=best_params_adam['lr'])

    criterion = nn.CrossEntropyLoss()

    history_gpe = {'train_loss': [], 'val_loss': []}
    history_adam = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Train and evaluate GPE
        train_loss_gpe = train_model(model_gpe, optimizer_gpe, train_loader, criterion, device)
        val_loss_gpe = evaluate_model(model_gpe, val_loader, criterion, device)
        history_gpe['train_loss'].append(train_loss_gpe)
        history_gpe['val_loss'].append(val_loss_gpe)

        # Train and evaluate Adam
        train_loss_adam = train_model(model_adam, optimizer_adam, train_loader, criterion, device)
        val_loss_adam = evaluate_model(model_adam, val_loader, criterion, device)
        history_adam['train_loss'].append(train_loss_adam)
        history_adam['val_loss'].append(val_loss_adam)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"GPE Val Loss: {val_loss_gpe:.4f} | "
              f"Adam Val Loss: {val_loss_adam:.4f}")

    # --- Plot Results ---
    plt.figure(figsize=(12, 6))
    plt.plot(history_gpe['val_loss'], label='GPE(Adam) Validation Loss')
    plt.plot(history_adam['val_loss'], label='Adam Validation Loss', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('GPE(Adam) vs Adam Validation Loss')
    plt.legend()
    plt.grid(True)

    # Ensure the directory exists
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'comparison.png'))
    print(f"Plot saved to {os.path.join(output_dir, 'comparison.png')}")


if __name__ == '__main__':
    main()
