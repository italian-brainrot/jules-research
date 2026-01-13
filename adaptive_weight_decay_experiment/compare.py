
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from light_dataloader import TensorDataLoader
from mnist1d.data import get_dataset_args, get_dataset
import numpy as np
import copy

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- Model Definition ---
class MLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# --- Data Loading ---
def load_data():
    args = get_dataset_args()
    data = get_dataset(args, path='./mnist1d_data.pkl', download=False)
    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)
    return X_train, y_train, X_test, y_test

# --- Training and Evaluation ---
def train_and_evaluate(model, optimizer, train_loader, val_loader, epochs, adaptive_wd_strength=0.0):
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')

    initial_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Apply adaptive weight decay if strength > 0
            if adaptive_wd_strength > 0:
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.add_(param.data.abs() * param.data, alpha=adaptive_wd_strength)

            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    # Restore initial model state for the next trial
    model.load_state_dict(initial_state)

    return best_val_loss

# --- Optuna Objective ---
def objective(trial, X_train, y_train, X_test, y_test, use_adaptive_wd):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    # For baseline, tune a standard weight_decay
    # For the new method, tune the adaptive_wd_strength
    if use_adaptive_wd:
        adaptive_wd_strength = trial.suggest_float('adaptive_wd_strength', 1e-5, 1e-1, log=True)
        weight_decay = 0.0
    else:
        adaptive_wd_strength = 0.0
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)

    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    val_loader = TensorDataLoader((X_test, y_test), batch_size=128)

    # Use a smaller number of epochs for faster tuning
    final_val_loss = train_and_evaluate(model, optimizer, train_loader, val_loader, epochs=10, adaptive_wd_strength=adaptive_wd_strength)

    return final_val_loss

# --- Main Execution ---
if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()

    # --- Baseline: Adam with standard weight decay ---
    study_baseline = optuna.create_study(direction='minimize')
    study_baseline.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test, use_adaptive_wd=False), n_trials=50)

    print("--- Baseline (Adam with L2) ---")
    print(f"Best validation loss: {study_baseline.best_value}")
    print(f"Best hyperparameters: {study_baseline.best_params}")

    # --- New Method: Adam with Adaptive Weight Decay ---
    study_adaptive = optuna.create_study(direction='minimize')
    study_adaptive.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test, use_adaptive_wd=True), n_trials=50)

    print("\n--- New Method (Adam with Adaptive WD) ---")
    print(f"Best validation loss: {study_adaptive.best_value}")
    print(f"Best hyperparameters: {study_adaptive.best_params}")

    # --- Final Conclusion ---
    print("\n--- Conclusion ---")
    if study_adaptive.best_value < study_baseline.best_value:
        print("Adaptive Weight Decay performed BETTER than the baseline.")
    else:
        print("Adaptive Weight Decay performed WORSE or SIMILAR to the baseline.")

    print(f"\nBaseline Best Validation Loss: {study_baseline.best_value:.4f}")
    print(f"Adaptive WD Best Validation Loss: {study_adaptive.best_value:.4f}")
