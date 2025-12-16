import torch
import torch.nn as nn
import torch.optim as optim
from .utils import get_model_and_data, train_and_evaluate
import numpy as np
import os
import matplotlib.pyplot as plt
import optuna
import torch

# --- Configuration ---
N_EPOCHS = 20
N_TRIALS_OPTUNA = 20

# --- Load Evolved Schedule ---
class PolynomialSchedule:
    """Represents a polynomial learning rate schedule."""
    def __init__(self, coeffs, lr_scale=1e-4):
        self.coeffs = np.array(coeffs, dtype=np.float32)
        self.lr_scale = lr_scale

    def get_lr(self, epoch_normalized):
        """Calculate LR for a given normalized epoch (0 to 1)."""
        res = 0
        for c in reversed(self.coeffs):
            res = res * epoch_normalized + c
        return np.abs(res) * self.lr_scale

# --- Schedulers to Compare ---
def get_schedulers(best_coeffs):
    schedulers = {
        'evolved_polynomial': lambda lr: PolynomialSchedule(best_coeffs, lr_scale=lr).get_lr,
        'constant': lambda lr: lambda epoch_normalized: lr,
        'linear_decay': lambda lr: lambda epoch_normalized: lr * (1 - epoch_normalized),
        'cosine_annealing': lambda lr: lambda epoch_normalized: lr * (0.5 * (1 + np.cos(epoch_normalized * np.pi)))
    }
    return schedulers

# --- Optuna Objective for LR Tuning ---
def create_objective(scheduler_name, schedulers, model_template, train_loader, val_loader):
    def objective(trial):
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        schedule_func_factory = schedulers[scheduler_name]
        schedule_func = schedule_func_factory(lr)

        val_accuracies = train_and_evaluate(model_template, train_loader, val_loader, schedule_func, N_EPOCHS)
        return val_accuracies[-1]
    return objective

# --- Main Comparison Logic ---
def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))

    coeffs_path = os.path.join(output_dir, 'best_coeffs.npy')
    if not os.path.exists(coeffs_path):
        print(f"Error: `best_coeffs.npy` not found. Please run `main.py` first.")
        return
    best_coeffs = np.load(coeffs_path)
    print(f"Loaded best coefficients: {best_coeffs}")

    model_template, train_loader, val_loader = get_model_and_data()
    schedulers = get_schedulers(best_coeffs)
    results = {}

    for name, scheduler_factory in schedulers.items():
        print(f"\n--- Benchmarking {name} ---")

        study = optuna.create_study(direction='maximize')
        objective_func = create_objective(name, schedulers, model_template, train_loader, val_loader)
        study.optimize(objective_func, n_trials=N_TRIALS_OPTUNA)

        best_lr = study.best_trial.params['lr']
        print(f"Best LR for {name}: {best_lr:.6f}")

        final_schedule = scheduler_factory(best_lr)
        accuracy_history = train_and_evaluate(model_template, train_loader, val_loader, final_schedule, N_EPOCHS)
        results[name] = accuracy_history
        print(f"Final accuracy for {name}: {accuracy_history[-1]:.4f}")

    # Plot the results
    plt.figure(figsize=(12, 8))
    for name, history in results.items():
        plt.plot(range(1, N_EPOCHS + 1), history, label=name)

    plt.title('Comparison of Learning Rate Schedulers')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_dir, 'comparison_plot.png')
    plt.savefig(plot_path)
    print(f"\nSaved comparison plot to {plot_path}")

if __name__ == '__main__':
    # Set a seed for reproducibility of model initialization
    torch.manual_seed(42)
    np.random.seed(42)
    main()
