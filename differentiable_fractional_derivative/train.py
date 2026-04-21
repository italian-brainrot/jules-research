import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import BaselineMLP, DFDAugmentedMLP
import os

def get_data():
    args = get_dataset_args()
    args.num_samples = 10000
    data = make_dataset(args)

    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, lr, epochs=50, batch_size=128):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in dl_train:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        output = model(X_test)
        pred = output.argmax(dim=1)
        acc = (pred == y_test).float().mean().item()
    return acc

def objective(trial, model_type):
    X_train, y_train, X_test, y_test = get_data()
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    if model_type == "baseline":
        model = BaselineMLP()
    else:
        model = DFDAugmentedMLP(num_orders=4)

    return train_model(model, X_train, y_train, X_test, y_test, lr, epochs=20)

def run_experiment():
    X_train, y_train, X_test, y_test = get_data()

    results = {}

    for model_type in ["baseline", "dfd"]:
        print(f"Tuning {model_type}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_type), n_trials=15)

        best_lr = study.best_params["lr"]
        print(f"Best LR for {model_type}: {best_lr}")

        accs = []
        for seed in range(5):
            torch.manual_seed(seed)
            if model_type == "baseline":
                model = BaselineMLP()
            else:
                model = DFDAugmentedMLP(num_orders=4)

            acc = train_model(model, X_train, y_train, X_test, y_test, best_lr, epochs=50)
            accs.append(acc)
            print(f"Seed {seed}, {model_type} Accuracy: {acc:.4f}")

            if model_type == "dfd":
                print(f"Learned orders: {model.dfd.orders.detach().cpu().numpy()}")

        results[model_type] = {
            "mean": np.mean(accs),
            "std": np.std(accs),
            "accs": accs,
            "best_lr": best_lr
        }

    # Save results
    with open("differentiable_fractional_derivative/results.txt", "w") as f:
        for model_type, res in results.items():
            f.write(f"{model_type}: {res['mean']:.4f} +- {res['std']:.4f} (Best LR: {res['best_lr']:.6f})\n")
            f.write(f"Accuracies: {res['accs']}\n\n")

    # Plot results
    plt.figure(figsize=(8, 6))
    model_names = ["Baseline MLP", "DFD-Augmented MLP"]
    means = [results["baseline"]["mean"], results["dfd"]["mean"]]
    stds = [results["baseline"]["std"], results["dfd"]["std"]]

    plt.bar(model_names, means, yerr=stds, capsize=5, color=['gray', 'blue'], alpha=0.7)
    plt.ylabel("Accuracy")
    plt.title("MNIST-1D Classification Results")
    plt.ylim(0, 1.0)
    for i, v in enumerate(means):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
    plt.savefig("differentiable_fractional_derivative/comparison.png")

if __name__ == "__main__":
    run_experiment()
