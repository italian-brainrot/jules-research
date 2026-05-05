import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from mnist1d.data import get_dataset, get_dataset_args
from mnist1d.utils import set_seed
from model import BaselineMLP, DDMDNet, DDMDAugmentedMLP
import matplotlib.pyplot as plt
import os

def get_mnist1d_data():
    args = get_dataset_args()
    dataset = get_dataset(args)
    x_train = torch.tensor(dataset['x'], dtype=torch.float32)
    y_train = torch.tensor(dataset['y'], dtype=torch.long)
    x_test = torch.tensor(dataset['x_test'], dtype=torch.float32)
    y_test = torch.tensor(dataset['y_test'], dtype=torch.long)
    return x_train, y_train, x_test, y_test

def train_model(model, x_train, y_train, x_test, y_test, lr, epochs=30, batch_size=128):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(x_train.size(0))
        for i in range(0, x_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = x_train[indices], y_train[indices]

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        output = model(x_test)
        pred = output.argmax(dim=1)
        acc = (pred == y_test).float().mean().item()
    return acc

def objective(trial, model_type):
    x_train, y_train, x_test, y_test = get_mnist1d_data()
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    if model_type == "baseline":
        model = BaselineMLP()
    elif model_type == "ddmd":
        model = DDMDNet()
    else:
        model = DDMDAugmentedMLP()

    acc = train_model(model, x_train, y_train, x_test, y_test, lr, epochs=20)
    return acc

def main():
    x_train, y_train, x_test, y_test = get_mnist1d_data()
    results = {}

    for model_type in ["baseline", "ddmd", "augmented"]:
        print(f"Tuning {model_type}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_type), n_trials=5)

        best_lr = study.best_params["lr"]
        print(f"Best LR for {model_type}: {best_lr}")

        # Train final models with best LR
        accs = []
        for seed in range(3):
            set_seed(seed)
            if model_type == "baseline":
                model = BaselineMLP()
            elif model_type == "ddmd":
                model = DDMDNet()
            else:
                model = DDMDAugmentedMLP()

            acc = train_model(model, x_train, y_train, x_test, y_test, best_lr, epochs=30)
            accs.append(acc)

        results[model_type] = {
            "mean": np.mean(accs),
            "std": np.std(accs),
            "best_lr": best_lr
        }

    # Print results
    print("\nFinal Results:")
    for model_type, metrics in results.items():
        print(f"{model_type}: {metrics['mean']:.4f} +/- {metrics['std']:.4f} (LR: {metrics['best_lr']:.2e})")

    # Save results to file
    with open("differentiable_dmd_experiment/results.txt", "w") as f:
        for model_type, metrics in results.items():
            f.write(f"{model_type}: {metrics['mean']:.4f} +/- {metrics['std']:.4f}\n")

    # Plot results
    labels = list(results.keys())
    means = [results[l]["mean"] for l in labels]
    stds = [results[l]["std"] for l in labels]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, means, yerr=stds, capsize=5, color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylabel("Accuracy")
    plt.title("MNIST-1D Classification Results with DDMD Features")
    plt.ylim(0, 1.0)
    plt.savefig("differentiable_dmd_experiment/results.png")
    plt.close()

if __name__ == "__main__":
    main()
