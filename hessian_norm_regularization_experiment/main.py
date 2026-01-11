import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import get_dataset, get_dataset_args
import numpy as np
import matplotlib.pyplot as plt
import optuna
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def get_data():
    """Loads the mnist1d dataset."""
    args = get_dataset_args()
    args.num_samples = 2000
    data = get_dataset(args, path='./mnist1d_data.pkl', download=False)
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'], dtype=torch.long)
    X_test, y_test = torch.tensor(data['x_test'], dtype=torch.float32), torch.tensor(data['y_test'], dtype=torch.long)
    return X_train, y_train, X_test, y_test

class SimpleMLP(nn.Module):
    """A simple MLP model."""
    def __init__(self, input_dim=40, hidden_dim=32, output_dim=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

def get_hessian_norm_approximation(model, criterion, inputs, targets):
    """
    Computes an approximation of the Frobenius norm of the Hessian using Hutchinson's method.
    """
    model.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # Generate a random Rademacher vector
    v = [torch.randint_like(p, 0, 2) * 2 - 1 for p in params]

    # Compute Hessian-vector product
    grad_v_prod = sum((g * v_i).sum() for g, v_i in zip(grads, v))
    h_v = torch.autograd.grad(grad_v_prod, params, retain_graph=False)

    # Approximate squared Frobenius norm: E[||Hv||^2]
    hessian_norm_sq = sum((h_v_i**2).sum() for h_v_i in h_v)
    return hessian_norm_sq

def train(model, train_loader, test_loader, lr, epochs, regularization_strength=0.0):
    """Trains the model with optional Hessian norm regularization."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if regularization_strength > 0:
                hessian_norm_sq = get_hessian_norm_approximation(model, criterion, inputs, targets)
                loss += regularization_strength * hessian_norm_sq

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        train_losses.append(epoch_train_loss / len(train_loader))

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()

        val_losses.append(epoch_val_loss / len(test_loader))

    return train_losses, val_losses

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = get_data()
    train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=128)

    # --- Optuna Objective for Regularized Model ---
    def objective_regularized(trial):
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        regularization_strength = trial.suggest_float('regularization_strength', 1e-5, 1e-1, log=True)

        model = SimpleMLP()
        _, val_losses = train(model, train_loader, test_loader, lr, epochs=10, regularization_strength=regularization_strength)
        return min(val_losses)

    # --- Optuna Objective for Baseline Model ---
    def objective_baseline(trial):
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

        model = SimpleMLP()
        _, val_losses = train(model, train_loader, test_loader, lr, epochs=10, regularization_strength=0.0)
        return min(val_losses)

    # --- Run Optuna Studies ---
    print("--- Tuning Regularized Model ---")
    study_regularized = optuna.create_study(direction='minimize')
    study_regularized.optimize(objective_regularized, n_trials=10)
    best_params_regularized = study_regularized.best_params
    print(f"Best params for regularized model: {best_params_regularized}")

    print("\n--- Tuning Baseline Model ---")
    study_baseline = optuna.create_study(direction='minimize')
    study_baseline.optimize(objective_baseline, n_trials=10)
    best_params_baseline = study_baseline.best_params
    print(f"Best params for baseline model: {best_params_baseline}")

    # --- Final Comparison ---
    print("\n--- Running Final Comparison ---")

    n_seeds = 2
    all_baseline_losses = []
    all_regularized_losses = []

    for i in range(n_seeds):
        print(f"--- Seed {i+1}/{n_seeds} ---")

        # Train baseline model
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)
        model_baseline = SimpleMLP()
        _, val_losses_baseline = train(model_baseline, train_loader, test_loader, best_params_baseline['lr'], epochs=20)
        all_baseline_losses.append(val_losses_baseline)

        # Train regularized model
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)
        model_regularized = SimpleMLP()
        _, val_losses_regularized = train(model_regularized, train_loader, test_loader, best_params_regularized['lr'], epochs=20, regularization_strength=best_params_regularized['regularization_strength'])
        all_regularized_losses.append(val_losses_regularized)

    # --- Plotting and Saving Results ---
    baseline_losses_mean = np.mean(all_baseline_losses, axis=0)
    regularized_losses_mean = np.mean(all_regularized_losses, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(baseline_losses_mean, label='Baseline')
    plt.plot(regularized_losses_mean, label='Regularized')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Baseline vs. Hessian Norm Regularization')
    plt.legend()
    plt.grid(True)

    # Save the plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'comparison.png'))

    print("\n--- Final Results ---")
    print(f"Baseline Final Validation Loss: {baseline_losses_mean[-1]:.4f}")
    print(f"Regularized Final Validation Loss: {regularized_losses_mean[-1]:.4f}")
