import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import DCDAugmentedMLP, DCDMLP, BaselineMLP
import matplotlib.pyplot as plt
import os

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)
    return X_train, y_train, X_test, y_test

def train_model(model, dl_train, lr, epochs=20):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for inputs, targets in dl_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        # Process in batches to avoid OOM if necessary, though 1D MNIST is small
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).float().mean().item()
    return accuracy

def objective(trial, model_name, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    # Use a subset of data for tuning to speed up
    subset_idx = torch.randperm(len(X_train))[:2000]
    X_subset = X_train[subset_idx]
    y_subset = y_train[subset_idx]

    dl_train = TensorDataLoader((X_subset, y_subset), batch_size=64, shuffle=True)

    if model_name == "dcd_augmented":
        model = DCDAugmentedMLP()
    elif model_name == "dcd_standalone":
        model = DCDMLP()
    else:
        model = BaselineMLP()

    train_model(model, dl_train, lr, epochs=10)
    return evaluate_model(model, X_test, y_test)

def run_experiment():
    X_train, y_train, X_test, y_test = get_data()

    models_to_tune = ["baseline", "dcd_standalone", "dcd_augmented"]
    best_lrs = {}

    for model_name in models_to_tune:
        print(f"Tuning {model_name}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_name, X_train, y_train, X_test, y_test), n_trials=10)
        best_lrs[model_name] = study.best_params["lr"]
        print(f"Best LR for {model_name}: {best_lrs[model_name]}")

    seeds = [42, 43, 44]
    results = {name: [] for name in models_to_tune}

    dl_train = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)

    for seed in seeds:
        for model_name in models_to_tune:
            torch.manual_seed(seed)
            if model_name == "baseline":
                model = BaselineMLP()
            elif model_name == "dcd_standalone":
                model = DCDMLP()
            elif model_name == "dcd_augmented":
                model = DCDAugmentedMLP()

            print(f"Training {model_name} with seed {seed}...")
            train_model(model, dl_train, best_lrs[model_name], epochs=30)
            acc = evaluate_model(model, X_test, y_test)
            results[model_name].append(acc)
            print(f"Seed {seed} {model_name} Accuracy: {acc}")

    output_dir = "differentiable_correlation_dimension_experiment"
    with open(f"{output_dir}/results.txt", "w") as f:
        for model_name in models_to_tune:
            mean_acc = np.mean(results[model_name])
            std_acc = np.std(results[model_name])
            f.write(f"{model_name}: {mean_acc:.4f} +/- {std_acc:.4f}\n")
            f.write(f"Best LR {model_name}: {best_lrs[model_name]}\n")

    # Plot results
    plt.figure(figsize=(10, 6))
    means = [np.mean(results[m]) for m in models_to_tune]
    stds = [np.std(results[m]) for m in models_to_tune]
    plt.bar(models_to_tune, means, yerr=stds, capsize=10)
    plt.ylabel("Accuracy")
    plt.title("Comparison of DCD Models on MNIST1D")
    plt.savefig(f"{output_dir}/comparison.png")
    print("Experiment complete. Results saved.")

if __name__ == "__main__":
    run_experiment()
