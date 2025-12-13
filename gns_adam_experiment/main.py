import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import matplotlib.pyplot as plt
import copy
import optuna

from .model import MLP
from .optimizer import GNSAdam

def run_experiment(optimizer_class, model, train_loader, test_loader, epochs, verbose=True, **optimizer_kwargs):
    """Runs a training and evaluation experiment for a given optimizer."""
    # A fresh copy of the model is needed for each run to ensure fair comparison
    model_instance = copy.deepcopy(model)
    criterion = nn.CrossEntropyLoss()

    # Special handling for GNSAdam which wraps another optimizer
    if optimizer_class == GNSAdam:
        base_optimizer_class = optimizer_kwargs.pop('base_optimizer', optim.Adam)
        optimizer_kwargs['base_optimizer'] = base_optimizer_class
        optimizer = optimizer_class(model_instance.parameters(), **optimizer_kwargs)
    else:
        optimizer = optimizer_class(model_instance.parameters(), **optimizer_kwargs)

    test_accuracies = []

    for epoch in range(epochs):
        model_instance.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model_instance(inputs.float())
            loss = criterion(outputs, targets.long())
            loss.backward()
            optimizer.step()

        model_instance.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model_instance(inputs.float())
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Optimizer: {optimizer_class.__name__}, Test Accuracy: {accuracy:.2f}%")

    return test_accuracies

def objective(trial, optimizer_class, model_prototype, train_loader, test_loader, epochs, **optimizer_kwargs_static):
    """Optuna objective function for hyperparameter tuning."""
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    optimizer_kwargs = optimizer_kwargs_static.copy()
    optimizer_kwargs['lr'] = lr

    # Run experiment without verbose logging for cleaner tuning output
    accuracies = run_experiment(
        optimizer_class,
        model_prototype,
        train_loader,
        test_loader,
        epochs,
        verbose=False,
        **optimizer_kwargs
    )
    return accuracies[-1]

def main():
    # --- Hyperparameters ---
    EPOCHS = 15
    BATCH_SIZE = 128
    N_TRIALS = 30  # Number of Optuna trials for tuning

    # --- Dataset ---
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)

    X_train, y_train = torch.tensor(data['x']), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"]), torch.tensor(data['y_test'])

    dl_train = TensorDataLoader((X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # --- Model Initialization ---
    # Create a single model prototype to ensure all trials start with the same initial weights
    model_prototype = MLP()

    # --- Optuna Study for Adam ---
    print("--- Tuning Adam ---")
    study_adam = optuna.create_study(direction="maximize")
    study_adam.optimize(
        lambda trial: objective(trial, optim.Adam, model_prototype, dl_train, dl_test, EPOCHS),
        n_trials=N_TRIALS
    )
    best_lr_adam = study_adam.best_params["lr"]
    print(f"Best LR for Adam: {best_lr_adam}")

    # --- Optuna Study for GNS-Adam ---
    print("\n--- Tuning GNS-Adam ---")
    study_gns = optuna.create_study(direction="maximize")
    study_gns.optimize(
        lambda trial: objective(trial, GNSAdam, model_prototype, dl_train, dl_test, EPOCHS, base_optimizer=optim.Adam),
        n_trials=N_TRIALS
    )
    best_lr_gns = study_gns.best_params["lr"]
    print(f"Best LR for GNS-Adam: {best_lr_gns}")

    # --- Run Final Experiments with Best Hyperparameters ---
    print("\n--- Running Final Comparison with Tuned Learning Rates ---")
    adam_history = run_experiment(
        optim.Adam, model_prototype, dl_train, dl_test, EPOCHS, lr=best_lr_adam
    )

    print("\n--- Starting GNS-Adam Experiment ---")
    gns_adam_history = run_experiment(
        GNSAdam, model_prototype, dl_train, dl_test, EPOCHS, base_optimizer=optim.Adam, lr=best_lr_gns
    )

    # --- Plot Results ---
    plt.figure(figsize=(10, 6))
    plt.plot(adam_history, label=f"Adam (lr={best_lr_adam:.4f})", marker='o')
    plt.plot(gns_adam_history, label=f"GNS-Adam (lr={best_lr_gns:.4f})", marker='x')
    plt.title("Tuned Optimizer Comparison: Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig("gns_adam_experiment/comparison_plot.png")
    print("\nComparison plot saved to gns_adam_experiment/comparison_plot.png")

if __name__ == "__main__":
    main()
