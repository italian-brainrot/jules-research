import torch
import torch.nn as nn
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import matplotlib.pyplot as plt
import copy
import os
import optuna

from optimizer import PPE

# --- 1. Dataset Setup ---
defaults = get_dataset_args()
defaults.num_samples = 10000
data = make_dataset(defaults)

X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'], dtype=torch.long)
X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'], dtype=torch.long)

dl_train = TensorDataLoader((X_train, y_train), batch_size=256, shuffle=True)
dl_test = TensorDataLoader((X_test, y_test), batch_size=1024)

# --- 2. Model Definition ---
def create_model():
    return nn.Sequential(
        nn.Linear(X_train.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

# --- 3. Training and Evaluation Loop ---
def train_and_evaluate(optimizer, model, epochs=50, verbose=True):
    criterion = nn.CrossEntropyLoss()
    val_losses = []

    for epoch in range(epochs):
        model.train()
        for inputs, targets in dl_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in dl_test:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)

        val_loss = total_loss / len(dl_test.data[0])
        val_losses.append(val_loss)
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}")

    return val_losses

# --- 4. Hyperparameter Tuning with Optuna ---
def objective(trial, optimizer_name, initial_model):
    model = copy.deepcopy(initial_model)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'PPE':
        base_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer = PPE(model.parameters(), base_optimizer, history_size=10, degree=2, alpha=0.4)
    else:
        raise ValueError("Unknown optimizer")

    # Use fewer epochs for tuning to speed up the process
    val_losses = train_and_evaluate(optimizer, model, epochs=20, verbose=False)
    return min(val_losses) # Return the best validation loss

def tune_optimizer_lr(optimizer_name, initial_model, n_trials=20):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, optimizer_name, initial_model), n_trials=n_trials)
    print(f"Best LR for {optimizer_name}: {study.best_params['lr']:.6f}")
    return study.best_params['lr']


# --- 5. Optimizer Comparison ---
if __name__ == "__main__":
    # Ensure fair comparison by starting with the same initial weights
    initial_model = create_model()

    # --- Tune Learning Rates ---
    print("--- Tuning Adam Learning Rate ---")
    best_lr_adam = tune_optimizer_lr('Adam', initial_model)

    print("\n--- Tuning PPE(Adam) Learning Rate ---")
    best_lr_ppe = tune_optimizer_lr('PPE', initial_model)


    # --- Train with Best Learning Rates ---
    print("\n--- Training with Adam (Tuned LR) ---")
    model_adam = copy.deepcopy(initial_model)
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=best_lr_adam)
    losses_adam = train_and_evaluate(optimizer_adam, model_adam)

    print("\n--- Training with PPE(Adam) (Tuned LR) ---")
    model_ppe = copy.deepcopy(initial_model)
    base_optimizer_ppe = torch.optim.Adam(model_ppe.parameters(), lr=best_lr_ppe)
    optimizer_ppe = PPE(model_ppe.parameters(), base_optimizer_ppe, history_size=10, degree=2, alpha=0.4)
    losses_ppe = train_and_evaluate(optimizer_ppe, model_ppe)

    # --- 6. Plotting Results ---
    plt.figure(figsize=(10, 6))
    plt.plot(losses_adam, label=f'Adam (LR={best_lr_adam:.4f})')
    plt.plot(losses_ppe, label=f'PPE(Adam) (LR={best_lr_ppe:.4f})')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Tuned Adam vs. Tuned PPE(Adam) Optimizer Performance')
    plt.legend()
    plt.grid(True)

    # Save the plot in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'comparison_plot.png'))

    print("\nComparison plot saved to 'comparison_plot.png'")
