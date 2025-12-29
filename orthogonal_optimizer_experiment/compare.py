import torch
import torch.nn as nn
import optuna
import matplotlib.pyplot as plt
import os
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from optimizer import OrthoAdam
import copy

# --- Configuration ---
DEVICE = torch.device("cpu")
N_TRIALS = 20
EPOCHS = 20
BATCH_SIZE = 128
N_SAMPLES = 4000

# --- Data Loading ---
def get_dataloaders(batch_size):
    defaults = get_dataset_args()
    defaults.num_samples = N_SAMPLES
    data = make_dataset(defaults)
    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_val = torch.tensor(data["x_test"], dtype=torch.float32)
    y_val = torch.tensor(data['y_test'], dtype=torch.long)

    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = TensorDataLoader((X_val, y_val), batch_size=batch_size)
    return train_loader, val_loader

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
def train_model(model, optimizer, train_loader, val_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    val_losses = []
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item() * x_batch.size(0)

        avg_loss = total_loss / len(val_loader.data[0])
        val_losses.append(avg_loss)
    return val_losses


# --- Optuna Objective Functions ---
def objective_adam(trial):
    train_loader, val_loader = get_dataloaders(BATCH_SIZE)
    model = create_model()
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    val_losses = train_model(model, optimizer, train_loader, val_loader, EPOCHS)
    return min(val_losses)

def objective_orthoadam(trial):
    train_loader, val_loader = get_dataloaders(BATCH_SIZE)
    model = create_model()
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    ortho_strength = trial.suggest_float("ortho_strength", 1e-6, 1e-2, log=True)
    optimizer = OrthoAdam(model.parameters(), lr=lr, ortho_strength=ortho_strength)

    val_losses = train_model(model, optimizer, train_loader, val_loader, EPOCHS)
    return min(val_losses)

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Tuning Adam ---")
    study_adam = optuna.create_study(direction="minimize")
    study_adam.optimize(objective_adam, n_trials=N_TRIALS)
    best_lr_adam = study_adam.best_params["lr"]
    print(f"Best LR for Adam: {best_lr_adam}")

    print("\n--- Tuning OrthoAdam ---")
    study_orthoadam = optuna.create_study(direction="minimize")
    study_orthoadam.optimize(objective_orthoadam, n_trials=N_TRIALS)
    best_lr_orthoadam = study_orthoadam.best_params["lr"]
    best_ortho_strength = study_orthoadam.best_params["ortho_strength"]
    print(f"Best LR for OrthoAdam: {best_lr_orthoadam}")
    print(f"Best Ortho Strength: {best_ortho_strength}")

    # --- Final Comparison ---
    print("\n--- Running Final Comparison ---")
    # Ensure fair comparison with same initial weights
    initial_model = create_model()

    model_adam = copy.deepcopy(initial_model)
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=best_lr_adam)

    model_orthoadam = copy.deepcopy(initial_model)
    optimizer_orthoadam = OrthoAdam(model_orthoadam.parameters(), lr=best_lr_orthoadam, ortho_strength=best_ortho_strength)

    train_loader, val_loader = get_dataloaders(BATCH_SIZE)

    adam_losses = train_model(model_adam, optimizer_adam, train_loader, val_loader, EPOCHS)
    orthoadam_losses = train_model(model_orthoadam, optimizer_orthoadam, train_loader, val_loader, EPOCHS)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(adam_losses, label=f'Adam (Best LR: {best_lr_adam:.4f})')
    plt.plot(orthoadam_losses, label=f'OrthoAdam (Best LR: {best_lr_orthoadam:.4f}, Strength: {best_ortho_strength:.4f})')
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.title("Adam vs. OrthoAdam Performance on MNIST-1D")
    plt.legend()
    plt.grid(True)

    # Save the plot in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, "convergence_comparison.png"))

    print("\nComparison complete. Plot saved to 'convergence_comparison.png'")
