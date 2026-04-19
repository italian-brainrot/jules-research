import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from model import BaselineMLP, HermiteAugmentedMLP
import json
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

def train_model(model, X_train, y_train, X_test, y_test, lr, epochs=30, batch_size=64):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    train_accs = []
    test_accs = []

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        train_acc = 100. * correct / total
        train_accs.append(train_acc)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = outputs.max(1)
            total += y_test.size(0)
            correct += predicted.eq(y_test).sum().item()

        test_acc = 100. * correct / total
        test_accs.append(test_acc)

    return train_accs, test_accs

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    input_dim = X_train.shape[1]
    output_dim = 10
    hidden_dim = 256

    if model_type == "baseline":
        model = BaselineMLP(input_dim, hidden_dim, output_dim)
    else:
        n_coeffs = 20
        model = HermiteAugmentedMLP(input_dim, n_coeffs, hidden_dim, output_dim)

    _, test_accs = train_model(model, X_train, y_train, X_test, y_test, lr, epochs=15)
    return max(test_accs)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()

    results = {}

    for model_type in ["baseline", "hermite"]:
        print(f"Tuning {model_type}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_type, X_train, y_train, X_test, y_test), n_trials=10)

        best_lr = study.best_params["lr"]
        print(f"Best LR for {model_type}: {best_lr}")

        # Train for real
        input_dim = X_train.shape[1]
        output_dim = 10
        hidden_dim = 256

        seeds = [0, 1, 2]
        all_train_accs = []
        all_test_accs = []

        for seed in seeds:
            torch.manual_seed(seed)
            if model_type == "baseline":
                model = BaselineMLP(input_dim, hidden_dim, output_dim)
            else:
                n_coeffs = 20
                model = HermiteAugmentedMLP(input_dim, n_coeffs, hidden_dim, output_dim)

            train_accs, test_accs = train_model(model, X_train, y_train, X_test, y_test, best_lr, epochs=50)
            all_train_accs.append(train_accs)
            all_test_accs.append(test_accs)

        results[model_type] = {
            "train": np.mean(all_train_accs, axis=0).tolist(),
            "test": np.mean(all_test_accs, axis=0).tolist(),
            "test_std": np.std(all_test_accs, axis=0).tolist(),
            "best_lr": best_lr
        }

    # Plot results
    plt.figure(figsize=(10, 6))
    for model_type in results:
        mean_test = np.array(results[model_type]["test"])
        std_test = np.array(results[model_type]["test_std"])
        epochs = range(1, len(mean_test) + 1)
        plt.plot(epochs, mean_test, label=f"{model_type} (test)")
        plt.fill_between(epochs, mean_test - std_test, mean_test + std_test, alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Hermite Augmented MLP vs Baseline MLP on MNIST-1D")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison.png")

    with open("results.json", "w") as f:
        json.dump(results, f)

    # Print summary
    for model_type in results:
        print(f"{model_type}: Max Test Acc = {max(results[model_type]['test']):.2f}%")
