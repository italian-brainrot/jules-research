import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import matplotlib.pyplot as plt
import optuna
import os
import copy

# Import the custom optimizer
from optimizer import SortedGradientOptimizer

# --- Model Definition ---
class MLP(nn.Module):
    def __init__(self, input_size=40, hidden_size=256, num_classes=10):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    def forward(self, x):
        return self.layers(x)

# --- Data Loading ---
def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.int64)
    X_val = torch.tensor(data['x_test'], dtype=torch.float32)
    y_val = torch.tensor(data['y_test'], dtype=torch.int64)

    train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    val_loader = TensorDataLoader((X_val, y_val), batch_size=512)
    return train_loader, val_loader

# --- Training and Evaluation ---
def train_and_evaluate(model, optimizer, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    val_loss_history = []

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item() * inputs.size(0)

        avg_val_loss = total_val_loss / len(val_loader.data[0])
        val_loss_history.append(avg_val_loss)

    return val_loss_history

# --- Optuna Objective ---
def objective(trial, optimizer_name, initial_state_dict):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    model = MLP()
    model.load_state_dict(copy.deepcopy(initial_state_dict))

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SortedAdam':
        base_optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer = SortedGradientOptimizer(base_optimizer)
    else:
        raise ValueError("Unknown optimizer")

    train_loader, val_loader = get_data()

    # Short training for hyperparameter tuning
    val_loss_history = train_and_evaluate(model, optimizer, train_loader, val_loader, epochs=5)

    return min(val_loss_history)

# --- Main Comparison ---
def main():
    print("Starting hyperparameter tuning...")

    # For fair comparison, start both models from the same initial weights
    initial_model = MLP()
    initial_state_dict = initial_model.state_dict()

    # Tune Adam
    study_adam = optuna.create_study(direction='minimize')
    study_adam.optimize(lambda trial: objective(trial, 'Adam', initial_state_dict), n_trials=15)
    best_lr_adam = study_adam.best_params['lr']
    print(f"Best LR for Adam: {best_lr_adam:.6f}")

    # Tune SortedGradientOptimizer(Adam)
    study_sorted = optuna.create_study(direction='minimize')
    study_sorted.optimize(lambda trial: objective(trial, 'SortedAdam', initial_state_dict), n_trials=15)
    best_lr_sorted = study_sorted.best_params['lr']
    print(f"Best LR for SortedAdam: {best_lr_sorted:.6f}")

    print("\nRunning final comparison with best learning rates...")

    # Train Adam with best LR
    model_adam = MLP()
    model_adam.load_state_dict(copy.deepcopy(initial_state_dict))
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=best_lr_adam)
    train_loader, val_loader = get_data()
    history_adam = train_and_evaluate(model_adam, optimizer_adam, train_loader, val_loader, epochs=25)

    # Train SortedAdam with best LR
    model_sorted = MLP()
    model_sorted.load_state_dict(copy.deepcopy(initial_state_dict))
    base_optimizer_sorted = optim.Adam(model_sorted.parameters(), lr=best_lr_sorted)
    optimizer_sorted = SortedGradientOptimizer(base_optimizer_sorted)
    history_sorted = train_and_evaluate(model_sorted, optimizer_sorted, train_loader, val_loader, epochs=25)

    print(f"Final validation loss (Adam): {history_adam[-1]:.4f}")
    print(f"Final validation loss (SortedAdam): {history_sorted[-1]:.4f}")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(history_adam, label=f'Adam (LR={best_lr_adam:.4f})')
    plt.plot(history_sorted, label=f'SortedAdam (LR={best_lr_sorted:.4f})')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'comparison_plot.png'))
    print(f"\nPlot saved to {os.path.join(script_dir, 'comparison_plot.png')}")

if __name__ == '__main__':
    main()
