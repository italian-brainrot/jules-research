import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import matplotlib.pyplot as plt
import optuna
import numpy as np
import random
import os
import copy

from optimizer import NormAdaptiveAdam

# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# --- Data Loading ---
defaults = get_dataset_args()
defaults.num_samples = 4000
data = make_dataset(defaults)

X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'])
X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'])

dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

# --- Model Definition ---
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)

# --- Training and Evaluation ---
def train_eval(optimizer, model, epochs=20):
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
            for inputs, targets in dl_test:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        val_losses.append(total_val_loss / len(dl_test))
    return val_losses

def train_eval_baseline(optimizer, model, clip_value, epochs=20):
    criterion = nn.CrossEntropyLoss()
    val_losses = []
    for epoch in range(epochs):
        model.train()
        for inputs, targets in dl_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
            optimizer.step()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in dl_test:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        val_losses.append(total_val_loss / len(dl_test))
    return val_losses


# --- Optuna Objective Functions ---
initial_model_state = copy.deepcopy(MLP().state_dict())

def objective_baseline(trial):
    model = MLP()
    model.load_state_dict(initial_model_state)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    clip_value = trial.suggest_float('clip_value', 1e-2, 1e1, log=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    val_losses = train_eval_baseline(optimizer, model, clip_value, epochs=10)
    return min(val_losses)


def objective_norm_adaptive(trial):
    model = MLP()
    model.load_state_dict(initial_model_state)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    clip_factor = trial.suggest_float('clip_factor', 1e-2, 1e1, log=True)

    base_optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = NormAdaptiveAdam(model.parameters(), base_optimizer=base_optimizer, clip_factor=clip_factor)

    val_losses = train_eval(optimizer, model, epochs=10) # Shorter tuning phase
    return min(val_losses)

# --- Main Comparison ---
if __name__ == '__main__':
    print("Tuning Baseline Adam with Fixed Clipping...")
    study_baseline = optuna.create_study(direction='minimize')
    study_baseline.optimize(objective_baseline, n_trials=30)
    best_params_baseline = study_baseline.best_params
    print(f"Best Baseline Params: {best_params_baseline}")

    print("\nTuning Norm-Adaptive Adam...")
    study_norm_adaptive = optuna.create_study(direction='minimize')
    study_norm_adaptive.optimize(objective_norm_adaptive, n_trials=30)
    best_params_norm_adaptive = study_norm_adaptive.best_params
    print(f"Best Norm-Adaptive Adam Params: {best_params_norm_adaptive}")

    # Final training runs with best hyperparameters
    print("\nRunning final comparison...")

    # Baseline
    model_baseline = MLP()
    model_baseline.load_state_dict(initial_model_state)
    optimizer_baseline = optim.Adam(model_baseline.parameters(), lr=best_params_baseline['lr'])
    baseline_losses = train_eval_baseline(optimizer_baseline, model_baseline, best_params_baseline['clip_value'], epochs=25)

    print("Baseline Final Run:")
    for i, loss in enumerate(baseline_losses):
        print(f"Epoch {i+1}, Val Loss: {loss:.4f}")

    # Norm-Adaptive
    model_norm_adaptive = MLP()
    model_norm_adaptive.load_state_dict(initial_model_state)
    base_optimizer_na = optim.Adam(model_norm_adaptive.parameters(), lr=best_params_norm_adaptive['lr'])
    optimizer_norm_adaptive = NormAdaptiveAdam(
        model_norm_adaptive.parameters(),
        base_optimizer=base_optimizer_na,
        clip_factor=best_params_norm_adaptive['clip_factor']
    )
    norm_adaptive_losses = train_eval(optimizer_norm_adaptive, model_norm_adaptive, epochs=25)
    print("\nNorm-Adaptive Final Run:")
    for i, loss in enumerate(norm_adaptive_losses):
        print(f"Epoch {i+1}, Val Loss: {loss:.4f}")


    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_losses, label=f'Baseline Adam (tuned clip={best_params_baseline["clip_value"]:.2f})')
    plt.plot(norm_adaptive_losses, label=f'Norm-Adaptive Adam (tuned factor={best_params_norm_adaptive["clip_factor"]:.2f})')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Optimizer Comparison')
    plt.legend()
    plt.grid(True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'comparison.png'))

    print(f"\nPlot saved to {os.path.join(script_dir, 'comparison.png')}")
