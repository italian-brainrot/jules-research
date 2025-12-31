import torch
import torch.nn as nn
import optuna
import mnist1d
from light_dataloader import TensorDataLoader
import matplotlib.pyplot as plt
import copy
import os
import numpy as np

from optimizer import SVDGradientOptimizer

# --- Environment Setup ---
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cpu")
EPOCHS = 20
N_TRIALS = 30

# --- Data Loading ---
args = mnist1d.data.get_dataset_args()
data = mnist1d.data.get_dataset(args)
train_loader = TensorDataLoader(
    (torch.from_numpy(data['x']).float(), torch.from_numpy(data['y']).long()),
    batch_size=128,
    shuffle=True
)
val_loader = TensorDataLoader(
    (torch.from_numpy(data['x_test']).float(), torch.from_numpy(data['y_test']).long()),
    batch_size=512
)

# --- Model Definition ---
def create_model():
    return nn.Sequential(
        nn.Linear(40, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(DEVICE)

# --- Training and Evaluation ---
def train_and_evaluate(model, optimizer, criterion):
    val_losses = []
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                output = model(x)
                total_loss += criterion(output, y).item()
        val_losses.append(total_loss / len(val_loader.data[0]))
    return val_losses

# --- Optuna Objectives ---
def objective_adam(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    val_losses = train_and_evaluate(model, optimizer, criterion)
    return min(val_losses)

def objective_svd(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    k = trial.suggest_int("k", 1, 10)
    model = create_model()
    base_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = SVDGradientOptimizer(base_optimizer, k=k)
    criterion = nn.CrossEntropyLoss()
    val_losses = train_and_evaluate(model, optimizer, criterion)
    return min(val_losses)

# --- Main Execution ---
if __name__ == "__main__":
    # --- Tune Adam ---
    print("--- Tuning Adam Optimizer ---")
    study_adam = optuna.create_study(direction="minimize")
    study_adam.optimize(objective_adam, n_trials=N_TRIALS)
    best_lr_adam = study_adam.best_trial.params['lr']
    print(f"Best Adam lr: {best_lr_adam}")

    # --- Tune SVD(Adam) ---
    print("\n--- Tuning SVD(Adam) Optimizer ---")
    study_svd = optuna.create_study(direction="minimize")
    study_svd.optimize(objective_svd, n_trials=N_TRIALS)
    best_lr_svd = study_svd.best_trial.params['lr']
    best_k_svd = study_svd.best_trial.params['k']
    print(f"Best SVD(Adam) lr: {best_lr_svd}, k: {best_k_svd}")

    # --- Final Comparison ---
    print("\n--- Running Final Comparison ---")

    # Adam with best params
    model_adam = create_model()
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=best_lr_adam)
    adam_losses = train_and_evaluate(model_adam, optimizer_adam, nn.CrossEntropyLoss())

    # SVD(Adam) with best params
    model_svd = create_model()
    base_optimizer_svd = torch.optim.Adam(model_svd.parameters(), lr=best_lr_svd)
    optimizer_svd = SVDGradientOptimizer(base_optimizer_svd, k=best_k_svd)
    svd_losses = train_and_evaluate(model_svd, optimizer_svd, nn.CrossEntropyLoss())

    # --- Plotting Results ---
    plt.figure(figsize=(10, 6))
    plt.plot(adam_losses, label=f"Adam (lr={best_lr_adam:.4f})")
    plt.plot(svd_losses, label=f"SVD(Adam) (lr={best_lr_svd:.4f}, k={best_k_svd})")
    plt.title("Validation Loss Comparison (Fairly Tuned)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, "comparison.png"))

    print(f"\nPlot saved to {os.path.join(script_dir, 'comparison.png')}")
