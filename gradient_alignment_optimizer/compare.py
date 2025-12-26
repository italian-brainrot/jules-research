import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
import os
import numpy as np

from optimizer import GradientAlignmentOptimizer
from mnist1d.data import get_dataset
from light_dataloader import TensorDataLoader as DataLoader

# --- 1. Dataset and Model ---
def get_data():
    args = type('Args', (), {})()
    args.seed = 0
    args.path = '.'
    data = get_dataset(args)
    return data

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.layers(x)

# --- 2. Training and Evaluation ---
def train_and_evaluate(optimizer, model, train_loader, val_loader, epochs=50):
    criterion = nn.CrossEntropyLoss()
    val_losses = []

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs = model(x_batch)
                val_loss += criterion(outputs, y_batch).item()
        val_losses.append(val_loss / len(val_loader))

    return val_losses

# --- 3. Optuna Objective ---
def objective(trial, optimizer_name, train_loader, val_loader, initial_state_dict):
    model = SimpleMLP(input_size=40, hidden_size=128, output_size=10)
    model.load_state_dict(initial_state_dict)

    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    if optimizer_name == 'Adam':
        base_optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer = base_optimizer
    elif optimizer_name == 'GaoAdam':
        beta = trial.suggest_float('beta', 0.8, 0.99)
        base_optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer = GradientAlignmentOptimizer(base_optimizer, beta=beta)
    else:
        raise ValueError("Unknown optimizer")

    val_losses = train_and_evaluate(optimizer, model, train_loader, val_loader, epochs=25)
    return min(val_losses)

# --- 4. Main Execution ---
if __name__ == '__main__':
    # Setup
    torch.manual_seed(42)
    np.random.seed(42)

    data = get_data()
    train_loader = DataLoader((torch.from_numpy(data['x']).float(), torch.from_numpy(data['y']).long()), batch_size=64, shuffle=True)
    val_loader = DataLoader((torch.from_numpy(data['x_test']).float(), torch.from_numpy(data['y_test']).long()), batch_size=256)

    # Save initial model state for fair comparison
    initial_model = SimpleMLP(input_size=40, hidden_size=128, output_size=10)
    initial_state_dict = initial_model.state_dict()

    # Hyperparameter tuning
    print("Tuning Adam...")
    study_adam = optuna.create_study(direction='minimize')
    study_adam.optimize(lambda trial: objective(trial, 'Adam', train_loader, val_loader, initial_state_dict), n_trials=30)

    print("\nTuning GradientAlignmentOptimizer(Adam)...")
    study_gao = optuna.create_study(direction='minimize')
    study_gao.optimize(lambda trial: objective(trial, 'GaoAdam', train_loader, val_loader, initial_state_dict), n_trials=30)

    best_lr_adam = study_adam.best_params['lr']
    best_params_gao = study_gao.best_params

    print(f"\nBest LR for Adam: {best_lr_adam}")
    print(f"Best params for GAO(Adam): {best_params_gao}")

    # Final training with best hyperparameters
    print("\nTraining final models...")
    model_adam = SimpleMLP(input_size=40, hidden_size=128, output_size=10)
    model_adam.load_state_dict(initial_state_dict)
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=best_lr_adam)

    model_gao = SimpleMLP(input_size=40, hidden_size=128, output_size=10)
    model_gao.load_state_dict(initial_state_dict)
    base_optimizer_gao = optim.Adam(model_gao.parameters(), lr=best_params_gao['lr'])
    optimizer_gao = GradientAlignmentOptimizer(base_optimizer_gao, beta=best_params_gao['beta'])

    epochs = 100
    losses_adam = train_and_evaluate(optimizer_adam, model_adam, train_loader, val_loader, epochs=epochs)
    losses_gao = train_and_evaluate(optimizer_gao, model_gao, train_loader, val_loader, epochs=epochs)

    # Plotting results
    plt.figure(figsize=(10, 6))
    plt.plot(losses_adam, label=f'Adam (Best LR: {best_lr_adam:.4f})')
    plt.plot(losses_gao, label=f'GAO(Adam) (Best LR: {best_params_gao["lr"]:.4f}, Beta: {best_params_gao["beta"]:.2f})')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    # Ensure the script's directory exists for saving the plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(script_dir, exist_ok=True)
    plt.savefig(os.path.join(script_dir, 'convergence_comparison.png'))

    print("\nComparison complete. Plot saved to 'convergence_comparison.png'.")
