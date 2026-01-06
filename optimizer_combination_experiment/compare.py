import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import matplotlib.pyplot as plt
import optuna
import copy
import os

from optimizer import CombinedOptimizer

# Set random seed for reproducibility
torch.manual_seed(42)

# --- 1. Data Loading ---
defaults = get_dataset_args()
defaults.num_samples = 10000
data = make_dataset(defaults)

X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'])
X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'])

dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

# --- 2. Model Definition ---
def create_model():
    return nn.Sequential(
        nn.Linear(40, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

# --- 3. Training and Evaluation ---
def train_epoch(model, optimizer, dataloader, criterion):
    model.train()
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# --- 4. Optuna Objective ---
N_EPOCHS_OPTUNA = 10
N_EPOCHS_FINAL = 30
TOTAL_STEPS_FINAL = N_EPOCHS_FINAL * len(dl_train)

def objective(trial, optimizer_name):
    model = create_model()
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "Combined":
        lr1 = trial.suggest_float("lr1", 1e-5, 1e-1, log=True)
        lr2 = trial.suggest_float("lr2", 1e-5, 1e-1, log=True)
        base_optimizer1 = optim.Adam(model.parameters(), lr=lr1)
        base_optimizer2 = optim.SGD(model.parameters(), lr=lr2)
        alpha_schedule = lambda t: min(1.0, t / (N_EPOCHS_OPTUNA * len(dl_train)))
        optimizer = CombinedOptimizer(base_optimizer1, base_optimizer2, alpha_schedule)
    else:
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(N_EPOCHS_OPTUNA):
        train_epoch(model, optimizer, dl_train, criterion)

    val_loss = evaluate(model, dl_test, criterion)
    return val_loss

# --- 5. Run Studies and Final Training ---
optimizers_to_test = ["Adam", "SGD", "Combined"]
results = {}

print("--- Starting Hyperparameter Tuning ---")
for optimizer_name in optimizers_to_test:
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, optimizer_name), n_trials=20, timeout=180)
    print(f"Best params for {optimizer_name}: {study.best_params}")

    # Train final model with best params
    model = create_model()
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "Combined":
        lr1 = study.best_params['lr1']
        lr2 = study.best_params['lr2']
        base_optimizer1 = optim.Adam(model.parameters(), lr=lr1)
        base_optimizer2 = optim.SGD(model.parameters(), lr=lr2)
        alpha_schedule = lambda t: min(1.0, t / TOTAL_STEPS_FINAL)
        optimizer = CombinedOptimizer(base_optimizer1, base_optimizer2, alpha_schedule)
    else:
        lr = study.best_params['lr']
        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr)

    history = []
    for epoch in range(N_EPOCHS_FINAL):
        train_epoch(model, optimizer, dl_train, criterion)
        val_loss = evaluate(model, dl_test, criterion)
        history.append(val_loss)
        print(f"[{optimizer_name}] Epoch {epoch+1}/{N_EPOCHS_FINAL}, Val Loss: {val_loss:.4f}")

    results[optimizer_name] = history

# --- 6. Plot Results ---
plt.figure(figsize=(10, 6))
for optimizer_name, history in results.items():
    plt.plot(history, label=optimizer_name)

plt.title("Optimizer Comparison on mnist1d")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("optimizer_combination_experiment/comparison.png")
print("\nPlot saved to optimizer_combination_experiment/comparison.png")
