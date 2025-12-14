import torch
import torch.nn as nn
import optuna
import matplotlib.pyplot as plt
import os
import copy

from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from .optimizer import GradientAgreementOptimizer

# --- 1. Device and Data Setup ---
DEVICE = torch.device("cpu")
defaults = get_dataset_args()
defaults.num_samples = 4000
data = make_dataset(defaults)

X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'])
X_val, y_val = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'])

dl_train = TensorDataLoader((X_train.to(DEVICE), y_train.to(DEVICE)), batch_size=128, shuffle=True)
dl_val = TensorDataLoader((X_val.to(DEVICE), y_val.to(DEVICE)), batch_size=128, shuffle=False)

# --- 2. Model Definition ---
def create_model():
    return nn.Sequential(
        nn.Linear(40, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(DEVICE)

# --- 3. Training and Evaluation Loop ---
def train_and_evaluate(model, optimizer, dl_train, dl_val, epochs):
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
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in dl_val:
                outputs = model(inputs)
                total_val_loss += criterion(outputs, targets).item()
        val_losses.append(total_val_loss / len(dl_val))
    return val_losses

# --- 4. Optuna Objective Function ---
def objective(trial, optimizer_name):
    model = create_model()
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        base_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer = GradientAgreementOptimizer(model.parameters(), base_optimizer)

    val_losses = train_and_evaluate(model, optimizer, dl_train, dl_val, epochs=10)
    return min(val_losses) # Optimize for best validation loss

# --- 5. Main Execution ---
if __name__ == "__main__":
    # --- Tune Hyperparameters ---
    print("Tuning Adam...")
    study_adam = optuna.create_study(direction='minimize')
    study_adam.optimize(lambda trial: objective(trial, 'Adam'), n_trials=20)
    best_lr_adam = study_adam.best_params['lr']
    print(f"Best LR for Adam: {best_lr_adam}")

    print("\nTuning GradientAgreementOptimizer(Adam)...")
    study_ga = optuna.create_study(direction='minimize')
    study_ga.optimize(lambda trial: objective(trial, 'GA_Adam'), n_trials=20)
    best_lr_ga = study_ga.best_params['lr']
    print(f"Best LR for GradientAgreementOptimizer(Adam): {best_lr_ga}")

    # --- Final Comparison ---
    print("\nRunning final comparison...")
    initial_model = create_model()

    # Adam
    model_adam = copy.deepcopy(initial_model)
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=best_lr_adam)
    history_adam = train_and_evaluate(model_adam, optimizer_adam, dl_train, dl_val, epochs=50)

    # Gradient Agreement Optimizer
    model_ga = copy.deepcopy(initial_model)
    base_optimizer_ga = torch.optim.Adam(model_ga.parameters(), lr=best_lr_ga)
    optimizer_ga = GradientAgreementOptimizer(model_ga.parameters(), base_optimizer_ga)
    history_ga = train_and_evaluate(model_ga, optimizer_ga, dl_train, dl_val, epochs=50)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(history_adam, label=f'Adam (LR={best_lr_adam:.4f})')
    plt.plot(history_ga, label=f'GA(Adam) (LR={best_lr_ga:.4f})')
    plt.title('Validation Loss: Adam vs. GradientAgreementOptimizer(Adam)')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'comparison.png'))
    print(f"\nPlot saved to {os.path.join(script_dir, 'comparison.png')}")
