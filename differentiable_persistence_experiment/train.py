import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import BaselineMLP, PersistenceAugmentedMLP, PersistenceMLP
import os

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'], dtype=torch.long)
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'], dtype=torch.long)
    return X_train, y_train, X_test, y_test

def train_model(model, train_loader, val_loader, lr, epochs=20):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def objective(trial, model_name, X_train, y_train, X_val, y_val):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    if model_name == "baseline":
        model = BaselineMLP()
    elif model_name == "persistence_augmented":
        model = PersistenceAugmentedMLP()
    elif model_name == "persistence_only":
        model = PersistenceMLP()

    train_loader = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)
    val_loader = TensorDataLoader((X_val, y_val), batch_size=64, shuffle=False)

    accuracy = train_model(model, train_loader, val_loader, lr, epochs=15)
    return accuracy

def main():
    X_train_full, y_train_full, X_test, y_test = get_data()

    # Split for tuning
    n_train = int(0.8 * len(X_train_full))
    X_train, y_train = X_train_full[:n_train], y_train_full[:n_train]
    X_val, y_val = X_train_full[n_train:], y_train_full[n_train:]

    results = {}
    model_names = ["baseline", "persistence_augmented", "persistence_only"]

    for model_name in model_names:
        print(f"Tuning {model_name}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_name, X_train, y_train, X_val, y_val), n_trials=10)

        best_lr = study.best_params["lr"]
        print(f"Best LR for {model_name}: {best_lr}")

        # Final evaluation with 3 seeds
        test_accs = []
        for seed in range(3):
            torch.manual_seed(seed)
            if model_name == "baseline":
                model = BaselineMLP()
            elif model_name == "persistence_augmented":
                model = PersistenceAugmentedMLP()
            elif model_name == "persistence_only":
                model = PersistenceMLP()

            train_loader = TensorDataLoader((X_train_full, y_train_full), batch_size=64, shuffle=True)
            test_loader = TensorDataLoader((X_test, y_test), batch_size=64, shuffle=False)

            acc = train_model(model, train_loader, test_loader, best_lr, epochs=30)
            test_accs.append(acc)

        results[model_name] = {
            "mean": np.mean(test_accs),
            "std": np.std(test_accs),
            "best_lr": best_lr
        }
        print(f"Results for {model_name}: {results[model_name]['mean']:.4f} +- {results[model_name]['std']:.4f}")

    with open("differentiable_persistence_experiment/results.txt", "w") as f:
        for model_name, res in results.items():
            f.write(f"{model_name}: {res['mean']:.4f} +- {res['std']:.4f} (best_lr: {res['best_lr']})\n")

    # Plot results
    names = list(results.keys())
    means = [results[n]["mean"] for n in names]
    stds = [results[n]["std"] for n in names]

    plt.figure(figsize=(10, 6))
    plt.bar(names, means, yerr=stds, capsize=5)
    plt.ylabel("Accuracy")
    plt.title("MNIST-1D Classification Results")
    plt.savefig("differentiable_persistence_experiment/comparison.png")

if __name__ == "__main__":
    main()
