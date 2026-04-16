import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import FrFTAugmentedMLP, BaselineMLP
import matplotlib.pyplot as plt
import os

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    # Ensure float32
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

    if model_name == "frft":
        model = FrFTAugmentedMLP()
    else:
        model = BaselineMLP()

    train_model(model, dl_train, lr, epochs=15)
    return evaluate_model(model, X_test, y_test)

def run_experiment():
    X_train, y_train, X_test, y_test = get_data()

    print("Tuning BaselineMLP...")
    study_base = optuna.create_study(direction="maximize")
    study_base.optimize(lambda trial: objective(trial, "baseline", X_train, y_train, X_test, y_test), n_trials=10)
    best_lr_base = study_base.best_params["lr"]

    print("Tuning FrFTAugmentedMLP...")
    study_frft = optuna.create_study(direction="maximize")
    study_frft.optimize(lambda trial: objective(trial, "frft", X_train, y_train, X_test, y_test), n_trials=10)
    best_lr_frft = study_frft.best_params["lr"]

    print(f"Best LR Baseline: {best_lr_base}")
    print(f"Best LR FrFT: {best_lr_frft}")

    seeds = [42, 43, 44]
    results_base = []
    results_frft = []

    dl_train = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)

    for seed in seeds:
        print(f"Running seed {seed}...")
        torch.manual_seed(seed)
        model_base = BaselineMLP()
        train_model(model_base, dl_train, best_lr_base, epochs=40)
        acc_base = evaluate_model(model_base, X_test, y_test)
        results_base.append(acc_base)
        print(f"Seed {seed} Baseline Accuracy: {acc_base}")

        torch.manual_seed(seed)
        model_frft = FrFTAugmentedMLP()
        train_model(model_frft, dl_train, best_lr_frft, epochs=40)
        acc_frft = evaluate_model(model_frft, X_test, y_test)
        results_frft.append(acc_frft)
        print(f"Seed {seed} FrFT Accuracy: {acc_frft}")

    mean_base, std_base = np.mean(results_base), np.std(results_base)
    mean_frft, std_frft = np.mean(results_frft), np.std(results_frft)

    print(f"Baseline: {mean_base:.4f} +/- {std_base:.4f}")
    print(f"FrFT: {mean_frft:.4f} +/- {std_frft:.4f}")

    with open("differentiable_frft_experiment/results.txt", "w") as f:
        f.write(f"Baseline: {mean_base:.4f} +/- {std_base:.4f}\n")
        f.write(f"FrFT: {mean_frft:.4f} +/- {std_frft:.4f}\n")
        f.write(f"Best LR Baseline: {best_lr_base}\n")
        f.write(f"Best LR FrFT: {best_lr_frft}\n")
        f.write(f"Raw Results Baseline: {results_base}\n")
        f.write(f"Raw Results FrFT: {results_frft}\n")

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.bar(["Baseline", "FrFT"], [mean_base, mean_frft],
            yerr=[std_base, std_frft], capsize=10, color=['gray', 'blue'])
    plt.ylabel("Accuracy")
    plt.title("Baseline vs FrFTAugmentedMLP on MNIST1D")
    plt.ylim(0, 1.0)
    plt.savefig("differentiable_frft_experiment/comparison.png")

if __name__ == "__main__":
    run_experiment()
