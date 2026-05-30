import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from model import BaselineMLP, DDSRNet, DDSAugmentedMLP
import os

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, lr, epochs=30, batch_size=64):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for x, y in dl_train:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(X_test)
        preds = torch.argmax(out, dim=1)
        acc = (preds == y_test).float().mean().item()
    return acc

def objective(trial, model_name):
    X_train, y_train, X_test, y_test = get_data()
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    if model_name == "baseline":
        model = BaselineMLP()
    elif model_name == "dds":
        model = DDSRNet()
    elif model_name == "dds_augmented":
        model = DDSAugmentedMLP()

    # Use a subset for faster tuning
    acc = train_model(model, X_train[:2000], y_train[:2000], X_test, y_test, lr, epochs=15)
    return acc

def run_experiment():
    X_train, y_train, X_test, y_test = get_data()
    models = ["baseline", "dds", "dds_augmented"]
    best_lrs = {}

    for name in models:
        print(f"Tuning {name}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, name), n_trials=10)
        best_lrs[name] = study.best_params["lr"]
        print(f"Best LR for {name}: {best_lrs[name]}")

    results = {name: [] for name in models}
    seeds = [42, 43, 44]

    for seed in seeds:
        torch.manual_seed(seed)
        for name in models:
            print(f"Evaluating {name} with seed {seed}...")
            if name == "baseline":
                model = BaselineMLP()
            elif name == "dds":
                model = DDSRNet()
            elif name == "dds_augmented":
                model = DDSAugmentedMLP()

            acc = train_model(model, X_train, y_train, X_test, y_test, best_lrs[name], epochs=30)
            results[name].append(acc)

    # Print summary
    with open("results.txt", "w") as f:
        for name in models:
            mean_acc = np.mean(results[name])
            std_acc = np.std(results[name])
            res_str = f"{name}: {mean_acc:.4f} +/- {std_acc:.4f}"
            print(res_str)
            f.write(res_str + "\n")

    # Plot results
    plt.figure(figsize=(10, 6))
    for name in models:
        plt.bar(name, np.mean(results[name]), yerr=np.std(results[name]), capsize=10)
    plt.ylabel("Accuracy")
    plt.title("MNIST-1D Model Comparison (DDS)")
    plt.savefig("comparison.png")
    plt.close()

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    if args.tune:
        X_train, y_train, X_test, y_test = get_data()
        models = ["baseline", "dds", "dds_augmented"]
        best_lrs = {}
        for name in models:
            print(f"Tuning {name}...")
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(trial, name), n_trials=10)
            best_lrs[name] = study.best_params["lr"]
            print(f"Best LR for {name}: {best_lrs[name]}")
        import json
        with open("best_lrs.json", "w") as f:
            json.dump(best_lrs, f)

    if args.eval:
        import json
        with open("best_lrs.json", "r") as f:
            best_lrs = json.load(f)

        X_train, y_train, X_test, y_test = get_data()
        models = ["baseline", "dds", "dds_augmented"]
        results = {name: [] for name in models}
        seeds = [42, 43, 44]

        for seed in seeds:
            torch.manual_seed(seed)
            for name in models:
                print(f"Evaluating {name} with seed {seed}...")
                if name == "baseline":
                    model = BaselineMLP()
                elif name == "dds":
                    model = DDSRNet()
                elif name == "dds_augmented":
                    model = DDSAugmentedMLP()

                acc = train_model(model, X_train, y_train, X_test, y_test, best_lrs[name], epochs=30)
                results[name].append(acc)

        # Print summary
        with open("results.txt", "w") as f:
            for name in models:
                mean_acc = np.mean(results[name])
                std_acc = np.std(results[name])
                res_str = f"{name}: {mean_acc:.4f} +/- {std_acc:.4f}"
                print(res_str)
                f.write(res_str + "\n")

        # Plot results
        plt.figure(figsize=(10, 6))
        for name in models:
            plt.bar(name, np.mean(results[name]), yerr=np.std(results[name]), capsize=10)
        plt.ylabel("Accuracy")
        plt.title("MNIST-1D Model Comparison (DDS)")
        plt.savefig("comparison.png")
        plt.close()

    if not args.tune and not args.eval:
        run_experiment()
