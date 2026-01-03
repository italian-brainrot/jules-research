import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import optuna
import matplotlib.pyplot as plt
import numpy as np
import os
from optimizer import Lookaround

# --- 1. Dataset and Model Definition ---

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 2000
    defaults.num_test = 1000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'], dtype=torch.long)
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'], dtype=torch.long)
    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=256)
    return dl_train, dl_test

class MLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# --- 2. Training and Evaluation Loop ---

def train_eval(model, optimizer, dl_train, dl_test, n_epochs):
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    val_loss_history = []

    for epoch in range(n_epochs):
        model.train()
        for inputs, targets in dl_train:
            def closure():
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                return loss

            if isinstance(optimizer, Lookaround):
                optimizer.step(closure)
            else:
                closure()
                optimizer.step()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in dl_test:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(dl_test)
        val_loss_history.append(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    return best_val_loss, val_loss_history

# --- 3. Optuna Objective Function ---

def objective(trial):
    dl_train, dl_test = get_data()

    # Use the same initial model weights for a fair comparison
    torch.manual_seed(42)
    model = MLP()
    initial_state_dict = model.state_dict()

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "Lookaround"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    model.load_state_dict(initial_state_dict) # Reset model to initial state

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        lookaround_alpha = trial.suggest_float("lookaround_alpha", 0.1, 0.9)
        base_optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer = Lookaround(base_optimizer, lookaround_alpha=lookaround_alpha)

    best_val_loss, _ = train_eval(model, optimizer, dl_train, dl_test, n_epochs=15)
    return best_val_loss

# --- 4. Main Experiment Execution ---

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, timeout=600)

    # Separate best trials for each optimizer
    trials = study.trials
    best_adam_trial = min([t for t in trials if t.params['optimizer'] == 'Adam' and t.state == optuna.trial.TrialState.COMPLETE], key=lambda t: t.value)
    best_lookaround_trial = min([t for t in trials if t.params['optimizer'] == 'Lookaround' and t.state == optuna.trial.TrialState.COMPLETE], key=lambda t: t.value)

    print("Best Adam trial:", best_adam_trial.params, "Value:", best_adam_trial.value)
    print("Best Lookaround trial:", best_lookaround_trial.params, "Value:", best_lookaround_trial.value)

    # Retrain with best hyperparameters to get loss curves
    dl_train, dl_test = get_data()
    torch.manual_seed(42)
    model_adam = MLP()
    adam_optimizer = optim.Adam(model_adam.parameters(), lr=best_adam_trial.params['lr'])
    _, adam_history = train_eval(model_adam, adam_optimizer, dl_train, dl_test, n_epochs=50)

    torch.manual_seed(42)
    model_lookaround = MLP()
    base_opt = optim.Adam(model_lookaround.parameters(), lr=best_lookaround_trial.params['lr'])
    lookaround_optimizer = Lookaround(base_opt, lookaround_alpha=best_lookaround_trial.params['lookaround_alpha'])
    _, lookaround_history = train_eval(model_lookaround, lookaround_optimizer, dl_train, dl_test, n_epochs=50)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(adam_history, label=f"Adam (Best val loss: {min(adam_history):.4f})")
    plt.plot(lookaround_history, label=f"Lookaround(Adam) (Best val loss: {min(lookaround_history):.4f})")
    plt.title("Validation Loss Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)

    # Save the plot in the experiment's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'comparison.png'))

    print(f"Plot saved to {os.path.join(script_dir, 'comparison.png')}")
