import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import RQAAugmentedMLP, BaselineMLP
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

    if model_name == "rqa":
        model = RQAAugmentedMLP()
    else:
        model = BaselineMLP()

    train_model(model, dl_train, lr, epochs=10)
    return evaluate_model(model, X_test, y_test)

def run_experiment():
    X_train, y_train, X_test, y_test = get_data()

    print("Tuning BaselineMLP...")
    study_base = optuna.create_study(direction="maximize")
    study_base.optimize(lambda trial: objective(trial, "baseline", X_train, y_train, X_test, y_test), n_trials=10)
    best_lr_base = study_base.best_params["lr"]

    print("Tuning RQAAugmentedMLP...")
    study_rqa = optuna.create_study(direction="maximize")
    study_rqa.optimize(lambda trial: objective(trial, "rqa", X_train, y_train, X_test, y_test), n_trials=10)
    best_lr_rqa = study_rqa.best_params["lr"]

    print(f"Best LR Baseline: {best_lr_base}")
    print(f"Best LR RQA: {best_lr_rqa}")

    seeds = [42, 43, 44]
    results_base = []
    results_rqa = []

    dl_train = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)

    for seed in seeds:
        torch.manual_seed(seed)
        model_base = BaselineMLP()
        train_model(model_base, dl_train, best_lr_base, epochs=30)
        acc_base = evaluate_model(model_base, X_test, y_test)
        results_base.append(acc_base)
        print(f"Seed {seed} Baseline Accuracy: {acc_base}")

        torch.manual_seed(seed)
        model_rqa = RQAAugmentedMLP()
        train_model(model_rqa, dl_train, best_lr_rqa, epochs=30)
        acc_rqa = evaluate_model(model_rqa, X_test, y_test)
        results_rqa.append(acc_rqa)
        print(f"Seed {seed} RQA Accuracy: {acc_rqa}")

    print(f"Baseline: {np.mean(results_base):.4f} +/- {np.std(results_base):.4f}")
    print(f"RQA: {np.mean(results_rqa):.4f} +/- {np.std(results_rqa):.4f}")

    with open("differentiable_rqa_experiment/results.txt", "w") as f:
        f.write(f"Baseline: {np.mean(results_base):.4f} +/- {np.std(results_base):.4f}\n")
        f.write(f"RQA: {np.mean(results_rqa):.4f} +/- {np.std(results_rqa):.4f}\n")
        f.write(f"Best LR Baseline: {best_lr_base}\n")
        f.write(f"Best LR RQA: {best_lr_rqa}\n")

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.bar(["Baseline", "RQA"], [np.mean(results_base), np.mean(results_rqa)],
            yerr=[np.std(results_base), np.std(results_rqa)], capsize=10)
    plt.ylabel("Accuracy")
    plt.title("Baseline vs RQAAugmentedMLP on MNIST1D")
    plt.savefig("differentiable_rqa_experiment/comparison.png")

if __name__ == "__main__":
    run_experiment()
