import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import BaselineMLP, DLCAMLP, DLCAAugmentedMLP
import os

def get_data():
    args = get_dataset_args()
    args.num_samples = 10000
    data = make_dataset(args)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data['x_test']).float(), torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, lr, epochs=100, batch_size=128):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        out_test = model(X_test)
        preds = out_test.argmax(dim=1)
        acc = (preds == y_test).float().mean().item()
    return acc

def objective(trial, model_class, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    # Use a subset of data for faster HPO
    X_train_hpo = X_train[:2000]
    y_train_hpo = y_train[:2000]

    if model_class == BaselineMLP:
        model = model_class(input_dim=40)
    else:
        model = model_class(input_dim=40, num_levels=10)

    acc = train_model(model, X_train_hpo, y_train_hpo, X_test, y_test, lr, epochs=50)
    return acc

def run_experiment():
    X_train, y_train, X_test, y_test = get_data()

    results = {}
    model_classes = [BaselineMLP, DLCAMLP, DLCAAugmentedMLP]
    model_names = ["BaselineMLP", "DLCAMLP", "DLCAAugmentedMLP"]

    best_lrs = {}

    for model_class, name in zip(model_classes, model_names):
        print(f"Tuning {name}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_class, X_train, y_train, X_test, y_test), n_trials=20)
        best_lrs[name] = study.best_params["lr"]
        print(f"Best LR for {name}: {best_lrs[name]}")

        # Multiple seeds evaluation
        accs = []
        for seed in range(3):
            torch.manual_seed(seed)
            if model_class == BaselineMLP:
                model = model_class(input_dim=40)
            else:
                model = model_class(input_dim=40, num_levels=10)
            acc = train_model(model, X_train, y_train, X_test, y_test, lr=best_lrs[name], epochs=100)
            accs.append(acc)
            print(f"Seed {seed} accuracy: {acc:.4f}")

        results[name] = {
            "mean": np.mean(accs),
            "std": np.std(accs)
        }
        print(f"{name} mean accuracy: {results[name]['mean']:.4f} +/- {results[name]['std']:.4f}")

    # Plot results
    names = list(results.keys())
    means = [results[n]["mean"] for n in names]
    stds = [results[n]["std"] for n in names]

    plt.figure(figsize=(10, 6))
    plt.bar(names, means, yerr=stds, capsize=5)
    plt.ylabel("Accuracy")
    plt.title("MNIST-1D Classification with DLCA Features")
    plt.savefig("differentiable_level_crossing_analysis/results.png")

    with open("differentiable_level_crossing_analysis/results.txt", "w") as f:
        for name in names:
            f.write(f"{name}: {results[name]['mean']:.4f} +/- {results[name]['std']:.4f} (Best LR: {best_lrs[name]})\n")

if __name__ == "__main__":
    run_experiment()
