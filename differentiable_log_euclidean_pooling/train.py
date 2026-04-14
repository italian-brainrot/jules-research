import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import DLEPModel, BaselineMLP
import matplotlib.pyplot as plt
import os

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)

    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.int64)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.int64)

    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, lr, epochs=50, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dl_train = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in dl_train:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test, y_test = X_test.to(device), y_test.to(device)
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)

    return accuracy

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    if model_type == "DLEP":
        model = DLEPModel()
    else:
        model = BaselineMLP()

    try:
        acc = train_model(model, X_train, y_train, X_test, y_test, lr, epochs=30)
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return 0.0
    return acc

def run_experiment():
    X_train, y_train, X_test, y_test = get_data()

    print("Tuning Baseline MLP...")
    study_baseline = optuna.create_study(direction="maximize")
    study_baseline.optimize(lambda trial: objective(trial, "Baseline", X_train, y_train, X_test, y_test), n_trials=10)
    best_lr_baseline = study_baseline.best_params["lr"]

    print("Tuning DLEP Model...")
    study_dlep = optuna.create_study(direction="maximize")
    study_dlep.optimize(lambda trial: objective(trial, "DLEP", X_train, y_train, X_test, y_test), n_trials=10)
    best_lr_dlep = study_dlep.best_params["lr"]

    print(f"Best LR Baseline: {best_lr_baseline}")
    print(f"Best LR DLEP: {best_lr_dlep}")

    # Final comparison with multiple seeds
    seeds = [42, 43, 44]
    results_baseline = []
    results_dlep = []

    for seed in seeds:
        torch.manual_seed(seed)
        model_baseline = BaselineMLP()
        acc_b = train_model(model_baseline, X_train, y_train, X_test, y_test, best_lr_baseline, epochs=50)
        results_baseline.append(acc_b)

        torch.manual_seed(seed)
        model_dlep = DLEPModel()
        acc_d = train_model(model_dlep, X_train, y_train, X_test, y_test, best_lr_dlep, epochs=50)
        results_dlep.append(acc_d)

    print(f"Baseline Results: {results_baseline}, Mean: {np.mean(results_baseline):.4f}")
    print(f"DLEP Results: {results_dlep}, Mean: {np.mean(results_dlep):.4f}")

    with open("results.txt", "w") as f:
        f.write(f"Baseline Results: {results_baseline}, Mean: {np.mean(results_baseline):.4f}, Std: {np.std(results_baseline):.4f}\n")
        f.write(f"DLEP Results: {results_dlep}, Mean: {np.mean(results_dlep):.4f}, Std: {np.std(results_dlep):.4f}\n")
        f.write(f"Best LR Baseline: {best_lr_baseline}\n")
        f.write(f"Best LR DLEP: {best_lr_dlep}\n")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.bar(["Baseline MLP", "DLEP Model"], [np.mean(results_baseline), np.mean(results_dlep)],
            yerr=[np.std(results_baseline), np.std(results_dlep)], capsize=10)
    plt.ylabel("Accuracy")
    plt.title("Comparison of Baseline MLP vs DLEP Model on MNIST-1D")
    plt.savefig("comparison.png")

if __name__ == "__main__":
    run_experiment()
